use crate::{
    ai::AI,
    data::{
        database::{Content, VDB},
        ingest,
    },
    qa::answer_query,
    utils::get_current_working_dir,
};
use anyhow::{anyhow, bail};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Terminal,
};
use serde_json::json;
use std::{collections::VecDeque, error::Error, io::stdout, sync::Arc, time::Instant};
use tokio::{sync::mpsc, task::spawn_blocking};

#[derive(Debug, Clone, Copy)]
enum Focus {
    Ask,
    Remember,
    Upload,
    List,
    Help,
}

struct App {
    focus: Focus,
    ask_input: String,
    remember_input: String,
    upload_input: String,
    start: u16,
    limit: u16,
    contents: Vec<Content>,
    answer: String,
    logs: VecDeque<String>,
    status: String,
    should_quit: bool,
}

impl App {
    fn new() -> Self {
        Self {
            focus: Focus::Ask,
            ask_input: String::new(),
            remember_input: String::new(),
            upload_input: String::new(),
            start: 0,
            limit: 10,
            contents: Vec::new(),
            answer: String::new(),
            logs: VecDeque::new(),
            status: "Ready".into(),
            should_quit: false,
        }
    }

    fn cycle_focus(&mut self, backwards: bool) {
        use Focus::*;
        self.focus = match (self.focus, backwards) {
            (Ask, false) => Remember,
            (Remember, false) => Upload,
            (Upload, false) => List,
            (List, false) => Help,
            (Help, false) => Ask,
            (Ask, true) => Help,
            (Remember, true) => Ask,
            (Upload, true) => Remember,
            (List, true) => Upload,
            (Help, true) => List,
        };
    }

    fn push_log<S: Into<String>>(&mut self, msg: S) {
        self.logs.push_back(msg.into());
        if self.logs.len() > 200 {
            let excess = self.logs.len() - 200;
            for _ in 0..excess {
                self.logs.pop_front();
            }
        }
    }
}

pub async fn run_repl(vdb: Arc<VDB>, ai: Arc<AI>) -> Result<(), Box<dyn Error>> {
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new();

    // Prime the content list once at startup.
    app.status = "Loading content…".into();
    if let Err(e) = refresh_content(&mut app, &vdb).await {
        app.push_log(format!("failed to load content: {e}"));
    } else {
        app.status = "Ready".into();
    }

    let (tx, mut rx) = mpsc::channel(32);
    tokio::task::spawn_blocking(move || loop {
        if let Ok(ev) = event::read() {
            if tx.blocking_send(ev).is_err() {
                break;
            }
        } else {
            break;
        }
    });

    while !app.should_quit {
        terminal.draw(|f| draw_ui(f, &app))?;
        match rx.recv().await {
            Some(ev) => {
                if let Err(e) = handle_event(ev, &mut app, &vdb, &ai).await {
                    app.push_log(format!("error: {e}"));
                    app.status = "Error".into();
                }
            }
            None => break,
        }
    }

    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;
    Ok(())
}

async fn handle_event(
    ev: Event,
    app: &mut App,
    vdb: &Arc<VDB>,
    ai: &Arc<AI>,
) -> Result<(), Box<dyn Error>> {
    if let Event::Key(key) = ev {
        if key.kind != KeyEventKind::Press {
            return Ok(());
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                app.should_quit = true;
            }
            KeyCode::Tab => app.cycle_focus(false),
            KeyCode::BackTab => app.cycle_focus(true),
            KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.status = "Refreshing content…".into();
                refresh_content(app, vdb).await?;
                app.status = "Ready".into();
            }
            KeyCode::Char(c) => {
                match app.focus {
                    Focus::Ask => app.ask_input.push(c),
                    Focus::Remember => app.remember_input.push(c),
                    Focus::Upload => app.upload_input.push(c),
                    Focus::List => {
                        // adjust paging quickly
                        if c == '+' {
                            app.start = app.start.saturating_add(app.limit);
                            refresh_content(app, vdb).await?;
                        } else if c == '-' {
                            app.start = app.start.saturating_sub(app.limit);
                            refresh_content(app, vdb).await?;
                        }
                    }
                    Focus::Help => {}
                }
            }
            KeyCode::Backspace => match app.focus {
                Focus::Ask => {
                    app.ask_input.pop();
                }
                Focus::Remember => {
                    app.remember_input.pop();
                }
                Focus::Upload => {
                    app.upload_input.pop();
                }
                _ => {}
            },
            KeyCode::Enter => match app.focus {
                Focus::Ask => {
                    if app.ask_input.trim().is_empty() {
                        app.status = "Query is empty".into();
                    } else {
                        let query = app.ask_input.trim().to_string();
                        app.status = "Thinking…".into();
                        let started = Instant::now();
                        let resp = answer_query(&query, vdb, ai).await?;
                        app.answer = format!("{resp}");
                        app.status = format!("Done in {:.2?}", started.elapsed());
                        app.push_log(format!("ask: {}", query));
                    }
                }
                Focus::Remember => {
                    if app.remember_input.trim().is_empty() {
                        app.status = "Note is empty".into();
                    } else {
                        let note = app.remember_input.trim().to_string();
                        ingest_note(vdb, &note).await?;
                        app.status = "Saved note".into();
                        app.push_log("remember: saved note");
                        refresh_content(app, vdb).await?;
                        app.remember_input.clear();
                    }
                }
                Focus::Upload => {
                    if app.upload_input.trim().is_empty() {
                        app.status = "Path is empty".into();
                    } else {
                        app.status = "Uploading…".into();
                        let result = ingest_path(vdb, &app.upload_input).await;
                        match result {
                            Ok(()) => {
                                app.status = "Uploaded".into();
                                app.push_log(format!("uploaded {}", app.upload_input));
                                refresh_content(app, vdb).await?;
                                app.upload_input.clear();
                            }
                            Err(e) => {
                                app.status = "Upload failed".into();
                                app.push_log(format!("upload error: {e}"));
                            }
                        }
                    }
                }
                Focus::List => {
                    refresh_content(app, vdb).await?;
                    app.status = "Refreshed content".into();
                }
                Focus::Help => {}
            },
            _ => {}
        }
    }
    Ok(())
}

async fn ingest_note(vdb: &Arc<VDB>, note: &str) -> Result<(), Box<dyn Error>> {
    vdb.process_content(
        "note",
        note,
        serde_json::json!({"source": "note", "kind": "remember"}),
    )
    .await?;
    Ok(())
}

async fn ingest_path(vdb: &Arc<VDB>, path_str: &str) -> Result<(), Box<dyn Error>> {
    let cwd = get_current_working_dir()?;
    let path = cwd.join(path_str);
    let metadata = tokio::fs::metadata(&path).await?;
    if !metadata.is_file() {
        panic!("Path is not a file");
    }
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext {
            "txt" => ingest::ingest_via_txt(vdb, &path).await?,
            "pdf" => ingest::ingest_via_pdf(vdb, &path).await?,
            _ => panic!("Unsupported file type: {}", ext),
        }
    } else {
        panic!("File has no extension");
    }
    Ok(())
}

async fn refresh_content(app: &mut App, vdb: &Arc<VDB>) -> Result<(), Box<dyn Error>> {
    let content = vdb.get_all_content(app.start, app.limit).await?;
    app.contents = content;
    Ok(())
}

fn draw_ui(f: &mut ratatui::Frame, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(6),
        ])
        .split(f.size());

    draw_header(f, layout[0], app);

    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(layout[1]);

    draw_left(f, main[0], app);
    draw_right(f, main[1], app);

    draw_footer(f, layout[2], app);
}

fn draw_header(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let focus = match app.focus {
        Focus::Ask => "ASK",
        Focus::Remember => "REMEMBER",
        Focus::Upload => "UPLOAD",
        Focus::List => "LIST",
        Focus::Help => "HELP",
    };
    let header = Paragraph::new(vec![Line::from(vec![
        Span::styled(
            "Cortex Console ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!("status: {} | focus: {}", app.status, focus)),
    ])])
    .block(Block::default().borders(Borders::ALL).title("Status"));
    f.render_widget(header, area);
}

fn draw_left(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Length(5),
            Constraint::Length(5),
            Constraint::Min(3),
        ])
        .split(area);

    let ask = Paragraph::new(app.ask_input.as_str())
        .block(input_block("Ask", matches!(app.focus, Focus::Ask)));
    f.render_widget(ask, chunks[0]);

    let remember = Paragraph::new(app.remember_input.as_str()).block(input_block(
        "Remember note",
        matches!(app.focus, Focus::Remember),
    ));
    f.render_widget(remember, chunks[1]);

    let upload = Paragraph::new(app.upload_input.as_str()).block(input_block(
        "Upload path (.txt/.pdf)",
        matches!(app.focus, Focus::Upload),
    ));
    f.render_widget(upload, chunks[2]);

    let answer = Paragraph::new(app.answer.as_str())
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("Answer"));
    f.render_widget(answer, chunks[3]);
}

fn draw_right(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(6)])
        .split(area);

    let items: Vec<ListItem> = app
        .contents
        .iter()
        .map(|c| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    c.title.clone(),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::raw(format!("{}", c.id)),
            ]))
        })
        .collect();
    let x = &format!("Content (start {} | limit {})", app.start, app.limit);
    let list = List::new(items)
        .block(input_block(x, matches!(app.focus, Focus::List)))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    f.render_widget(list, chunks[0]);

    let logs: Vec<Line> = app
        .logs
        .iter()
        .rev()
        .take(5)
        .rev()
        .map(|l| Line::from(l.clone()))
        .collect();
    let log_view = Paragraph::new(logs)
        .block(Block::default().borders(Borders::ALL).title("Logs"))
        .wrap(Wrap { trim: true });
    f.render_widget(log_view, chunks[1]);
}

fn draw_footer(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let help = Paragraph::new(vec![
        Line::from(
            "Tab/Shift-Tab: switch focus | Enter: run action | r: refresh list | q/esc: quit",
        ),
        Line::from(
            "Ask: type question -> Enter | Remember: type note -> Enter | Upload: path -> Enter",
        ),
        Line::from("List: +/- to page (start +=/-= limit) | Ready state: minimal key hints."),
    ])
    .block(input_block("Help", matches!(app.focus, Focus::Help)))
    .wrap(Wrap { trim: true });
    f.render_widget(help, area);
}

fn input_block<'a>(title: &'a str, focused: bool) -> Block<'a> {
    let style = if focused {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };
    Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(style)
}
