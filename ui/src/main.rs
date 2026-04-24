mod app;
mod repo;
mod telemetry;
mod view;

use std::io;
use std::time::{Duration, Instant};

use app::App;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

const TICK_RATE: Duration = Duration::from_millis(100);

fn main() -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut app = App::new()?;
    let mut last_tick = Instant::now();

    loop {
        let timeout = TICK_RATE.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('r') => app.reset(),
                        KeyCode::Char('c') => {
                            app.reasoning_log.push_back("[CONFIG] UI layout optimized for btop-density.".to_string());
                        }
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        _ => {}
                    }
                }
            }
        }

        if last_tick.elapsed() >= TICK_RATE {
            app.refresh();
            terminal.draw(|frame| view::draw(frame, &app))?;
            last_tick = Instant::now();
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
