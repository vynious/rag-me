use crate::{
    ai::AI,
    data::{
        database::{Content, VDB},
        ingest,
    },
    qa::answer_query,
    utils::get_current_working_dir,
};
use anyhow::Result;
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
use std::{
    collections::VecDeque,
    error::Error,
    io::stdout,
    sync::Arc,
};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Copy)]
enum Focus {
    Ask,
    Remember,
    Upload,
    List,
    Help,
}

enum AppEvent {
    Input(Event),
    Tick,
    Answered {
        query: String,
        result: Result<String>,
    },
    ContentLoaded(Vec<Content>),
    Log(String),
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

    // event bus
    let (ev_tx, mut ev_rx) = mpsc::channel::<AppEvent>(128);

    // input read -> appevent inputs
    {
        let tx = ev_tx.clone();
        tokio::task::spawn_blocking(move || {
            // read events from terminal
            while let Ok(ev) = event::read() {
                // send app event and check if error
                if tx.blocking_send(AppEvent::Input(ev)).is_err() {
                    break;
                }
            }
        });
    }

    // load content without blocking UI
    {
        let tx = ev_tx.clone();
        let vdb = vdb.clone();
        tokio::spawn(async move {
            let res = vdb.get_all_content(0, 10).await;
            match res {
                Ok(items) => {
                    let _ = tx.send(AppEvent::ContentLoaded(items)).await;
                }
                Err(e) => {
                    let _ = tx.send(AppEvent::Log(format!("failed to load {e}"))).await;
                }
            }
        });
    }

    while !app.should_quit {
        terminal.draw(|f| draw_ui(f, &app))?;

        match ev_rx.recv().await {
            Some(AppEvent::Input(ev)) => handle_input(ev, &mut app, &ev_tx, &vdb, &ai).await?,
            Some(AppEvent::Tick) => { /* optional: animations/timeouts */ }
            Some(AppEvent::ContentLoaded(items)) => {
                app.contents = items;
                app.status = "Ready".into();
            }
            Some(AppEvent::Answered { query, result }) => {
                match result {
                    Ok(ans) => app.answer = ans,
                    Err(e) => app.answer = format!("error: {e}"),
                }
                app.status = "Ready".into();
                app.push_log(format!("ask: {query}"));
            }
            Some(AppEvent::Log(msg)) => app.push_log(msg),
            None => break,
        }
    }

    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;
    Ok(())
}

async fn handle_input(
    ev: Event,
    app: &mut App,
    ev_tx: &mpsc::Sender<AppEvent>,
    vdb: &Arc<VDB>,
    ai: &Arc<AI>,
) -> Result<(), Box<dyn Error>> {
    if let Event::Key(key) = ev {
        if key.kind != KeyEventKind::Press {
            return Ok(());
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
            KeyCode::Tab => app.cycle_focus(false),
            KeyCode::BackTab => app.cycle_focus(true),

            KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.status = "Refreshing content…".into();
                let tx = ev_tx.clone();
                let vdb = vdb.clone();
                let start = app.start;
                let limit = app.limit;
                tokio::spawn(async move {
                    let res = vdb.get_all_content(start, limit).await;
                    let _ = match res {
                        Ok(items) => tx.send(AppEvent::ContentLoaded(items)).await,
                        Err(e) => tx.send(AppEvent::Log(format!("refresh error: {e}"))).await,
                    };
                });
            }

            KeyCode::Enter => match app.focus {
                Focus::Ask => {
                    let query = app.ask_input.trim().to_string();
                    if query.is_empty() {
                        app.status = "Query is empty".into();
                    } else {
                        app.status = "Thinking…".into();
                        let tx = ev_tx.clone();
                        let vdb = vdb.clone();
                        let ai = ai.clone();
                        tokio::spawn(async move {
                            let res = answer_query(&query, &vdb, &ai).await;
                            let _ = tx
                                .send(AppEvent::Answered {
                                    query,
                                    result: res.map(|r| r.0).map_err(|e| e.into()),
                                })
                                .await;
                        });
                    }
                }
                Focus::Remember => {
                    let note = app.remember_input.trim().to_string();
                    if note.is_empty() {
                        app.status = "Note is empty".into();
                    } else {
                        app.status = "Saving…".into();
                        let tx = ev_tx.clone();
                        let vdb = vdb.clone();
                        tokio::spawn(async move {
                            let res = ingest_note(&vdb, &note).await;
                            let _ = tx.send(AppEvent::Log(format!("remember: {res:?}"))).await;
                            if res.is_ok() {
                                let _ = tx
                                    .send(AppEvent::ContentLoaded(
                                        vdb.get_all_content(0, 10).await.unwrap_or_default(),
                                    ))
                                    .await;
                            }
                        });
                    }
                }
                Focus::Upload => {
                    let path = app.upload_input.trim().to_string();
                    if path.is_empty() {
                        app.status = "Path is empty".into();
                    } else {
                        app.status = "Uploading…".into();
                        let tx = ev_tx.clone();
                        let vdb = vdb.clone();
                        tokio::spawn(async move {
                            let res = ingest_path(&vdb, &path).await;
                            let _ = tx.send(AppEvent::Log(format!("upload: {res:?}"))).await;
                            if res.is_ok() {
                                let _ = tx
                                    .send(AppEvent::ContentLoaded(
                                        vdb.get_all_content(0, 10).await.unwrap_or_default(),
                                    ))
                                    .await;
                            }
                        });
                    }
                }
                Focus::List => {
                    let tx = ev_tx.clone();
                    let vdb = vdb.clone();
                    let start = app.start;
                    let limit = app.limit;
                    tokio::spawn(async move {
                        let res = vdb.get_all_content(start, limit).await;
                        let _ = match res {
                            Ok(items) => tx.send(AppEvent::ContentLoaded(items)).await,
                            Err(e) => tx.send(AppEvent::Log(format!("refresh error: {e}"))).await,
                        };
                    });
                }
                Focus::Help => {}
            },

            KeyCode::Char(c) => match app.focus {
                Focus::Ask => app.ask_input.push(c),
                Focus::Remember => app.remember_input.push(c),
                Focus::Upload => app.upload_input.push(c),
                Focus::List => {
                    if c == '+' {
                        app.start = app.start.saturating_add(app.limit);
                        let _ = ev_tx.send(AppEvent::Log("paging +".into())).await;
                    } else if c == '-' {
                        app.start = app.start.saturating_sub(app.limit);
                        let _ = ev_tx.send(AppEvent::Log("paging -".into())).await;
                    }
                }
                Focus::Help => {}
            },

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

            _ => {}
        }
    }
    Ok(())
}

async fn ingest_note(vdb: &Arc<VDB>, note: &str) -> anyhow::Result<()> {
    vdb.process_content(
        "note",
        note,
        serde_json::json!({"source": "note", "kind": "remember"}),
    )
    .await?;
    Ok(())
}

async fn ingest_path(vdb: &Arc<VDB>, path_str: &str) -> anyhow::Result<()> {
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
