use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use crate::state::UnpackedHUD;

pub fn render(frame: &mut Frame, area: Rect, hud: &UnpackedHUD) {
    let block = Block::default()
        .title(" TOPOLOGY: CORE AFFINITY ")
        .title_style(Style::default().fg(Color::DarkGray))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Meta details at top
    let meta_area = Rect::new(inner.x, inner.y, inner.width, 2);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled("Widget: Affinity Grid", Style::default().fg(Color::Gray))),
            Line::from(Span::styled("type: Grid, cores: 4", Style::default().fg(Color::DarkGray))),
        ]),
        meta_area,
    );

    let grid_area = Rect::new(inner.x, inner.y + 3, inner.width, inner.height.saturating_sub(3));
    let rows = Layout::vertical([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)]).split(grid_area);
    
    let top_cols = Layout::horizontal([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)]).split(rows[0]);
    let bot_cols = Layout::horizontal([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)]).split(rows[1]);

    let cores = [
        ("Core 0", top_cols[0], hud.features[0]),
        ("Core 1", top_cols[1], hud.features[1]),
        ("Core 2", bot_cols[0], hud.features[2]),
        ("Core 3", bot_cols[1], hud.features[3]),
    ];

    for (name, col_area, val) in cores {
        render_core(frame, col_area, name, val);
    }
}

fn render_core(frame: &mut Frame, area: Rect, name: &str, val: u64) {
    let block = Block::default()
        .title(name)
        .title_style(Style::default().add_modifier(Modifier::BOLD))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let load = (val as f64 / 100.0).clamp(0.0, 1.0);
    let bar_len = ((inner.width.saturating_sub(12)) as f64 * load) as usize;
    let bar = "█".repeat(bar_len) + &"░".repeat((inner.width.saturating_sub(12) as usize).saturating_sub(bar_len));
    
    let color = if load > 0.8 { Color::Rgb(255, 100, 100) } else { Color::Rgb(100, 200, 100) };

    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(bar, Style::default().fg(color)),
                Span::raw(format!(" {:>3}% P100, 2139", (load * 100.0) as u32)),
            ]),
            Line::from(vec![
                Span::styled("██████░░░", Style::default().fg(Color::DarkGray)),
                Span::raw("  8% P105, 2139"),
            ]),
        ]),
        inner,
    );
}
