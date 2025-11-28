use crate::{
    ai::AI,
    cli::{Cli, Commands},
    data::{database::VDB, ingest},
    qa::answer_query,
    utils::get_current_working_dir,
};
use clap::Parser;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Terminal,
};
use std::{error::Error, sync::Arc, time::Duration};
use tokio::{
    io::{self, AsyncBufReadExt, BufReader},
    time::timeout,
};

fn render_startup_tui() {
    if let Err(e) = draw_startup_tui() {
        let _ = disable_raw_mode();
        eprintln!("warning: could not render the startup TUI: {}", e);
        println!("Commands: ask <query>, remember <text>, upload <path>, list --start 0 --limit 10, forget --content-id <id> or --all");
    }
}

fn draw_startup_tui() -> Result<(), Box<dyn Error>> {
    use std::io::stdout;

    enable_raw_mode()?;
    let mut stdout = stdout();
    let backend = CrosstermBackend::new(&mut stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    terminal.draw(|f| {
        let size = f.size();
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(4),
                Constraint::Min(7),
                Constraint::Length(4),
            ])
            .split(size);

        let header = Paragraph::new(vec![
            Line::from(Span::styled(
                "rag-me server",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )),
            Line::from("Your REPL for asking, ingesting, and managing context."),
        ])
        .block(Block::default().borders(Borders::ALL).title("Welcome"));
        f.render_widget(header, sections[0]);

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(sections[1]);

        let commands = vec![
            ListItem::new(Line::from(vec![
                Span::styled(
                    "ask <query>",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  Ask a question against your indexed notes."),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled(
                    "remember <text>",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  Store a short note in the vector DB."),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled(
                    "upload <path>",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  Ingest a txt/pdf file for retrieval."),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled(
                    "list --start 0 --limit 10",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  Peek at stored content."),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled(
                    "forget (--content-id <id> | --all)",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
                Span::raw("  Remove one item or wipe everything."),
            ])),
        ];
        let commands_widget =
            List::new(commands).block(Block::default().borders(Borders::ALL).title("Commands"));
        f.render_widget(commands_widget, body[0]);

        let status_items = vec![
            ListItem::new(Line::from(vec![
                Span::styled(
                    "Vector DB",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(": ready"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled(
                    "AI workers",
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(": warmed up"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled(
                    "Uploads",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(": txt/pdf accepted"),
            ])),
        ];
        let status_widget =
            List::new(status_items).block(Block::default().borders(Borders::ALL).title("Status"));
        f.render_widget(status_widget, body[1]);

        let footer = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Tip:", Style::default().fg(Color::Yellow)),
                Span::raw(" Multi-word queries work best inside quotes."),
            ]),
            Line::from("Press Ctrl+C to exit the REPL; commands echo as they run."),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Getting started"),
        )
        .wrap(Wrap { trim: true });
        f.render_widget(footer, sections[2]);

        f.set_cursor(2, sections[2].y + sections[2].height.saturating_sub(1));
    })?;

    disable_raw_mode()?;
    println!();
    Ok(())
}

pub async fn run_repl(vdb: Arc<VDB>, ai: Arc<AI>) -> Result<(), Box<dyn Error>> {
    render_startup_tui();

    let stdin = io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let trimmed_line = line.trim();
        if trimmed_line.is_empty() {
            continue;
        }
        println!("receiving cmd: {}", trimmed_line);

        let args = match shell_words::split(trimmed_line) {
            Ok(tokens) => tokens,
            Err(e) => {
                eprintln!("error parsing commands: {}", e);
                continue;
            }
        };

        let mut cli_args = vec!["ragme"];
        cli_args.extend(args.iter().map(String::as_str));
        match Cli::try_parse_from(cli_args) {
            Ok(cli) => match cli.command {
                Commands::Ask { query } => {
                    let query_str = query.join(" ");
                    eprintln!("query = {}", query_str);
                    let response = answer_query(&query_str, &vdb, &ai).await;
                    match response {
                        Ok(answer) => {
                            println!("answer: {}", answer);
                        }
                        Err(e) => {
                            eprintln!("error asking question: {}", e);
                        }
                    }
                }
                Commands::Remember { content } => {
                    eprintln!("content = {}", content);
                }
                Commands::List { start, limit } => {
                    if let Ok(content) = vdb.get_all_content(start, limit).await {
                        for c in content {
                            println!("content: {:?}", c);
                        }
                    }
                }
                Commands::Forget { content_id, all } => {
                    eprintln!("forget called; content_id={:?}, all={}", content_id, all);
                }
                Commands::Upload { path } => {
                    eprintln!("path = {:?}", path);
                    let cwd = get_current_working_dir().unwrap();
                    let path = cwd.join(path);

                    let result =
                        match timeout(Duration::from_secs(5), tokio::fs::metadata(&path)).await {
                            Ok(result) => result,
                            Err(e) => {
                                eprintln!("Failed to get metadata for path due to: {:?}", e);
                                continue;
                            }
                        };
                    let metadata = match result {
                        Ok(metadata) => metadata,
                        Err(e) => {
                            eprintln!("Failed to get metadata for path: {:?}", e);
                            continue;
                        }
                    };
                    if !metadata.is_file() {
                        eprintln!("Path is not a file: {:?}", path);
                        continue;
                    };

                    if let Some(extension) = path.extension() {
                        match extension.to_str() {
                            Some("txt") => {
                                eprintln!("ingesting txt file");
                                ingest::ingest_via_txt(&vdb, &path).await.unwrap();
                            }
                            Some("pdf") => {
                                eprintln!("ingesting pdf file");
                                ingest::ingest_via_pdf(&vdb, &path).await.unwrap();
                            }
                            _ => {
                                eprintln!("Unsupported file type: {:?}", extension);
                            }
                        }
                    } else {
                        eprintln!("File has no extension: {:?}", path);
                    };
                }
            },
            Err(e) => {
                eprintln!("error parsing commands: {}", e);
            }
        }
    }

    Ok(())
}
