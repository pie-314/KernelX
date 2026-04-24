use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

pub fn render(frame: &mut Frame, area: Rect) {
    let rows = Layout::vertical([
        Constraint::Ratio(1, 3), // mem
        Constraint::Ratio(1, 3), // disks
        Constraint::Ratio(1, 3), // net
    ]).split(area);

    render_mem(frame, rows[0]);
    render_disks(frame, rows[1]);
    render_net(frame, rows[2]);
}

fn render_mem(frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .title(" mem ")
        .title_style(Style::default().fg(Color::DarkGray))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![Span::raw("Total:"), Span::raw(" ".repeat(inner.width.saturating_sub(15) as usize)), Span::styled("16.0 GiB", Style::default().fg(Color::White))]),
            Line::from(vec![Span::raw("Used: "), Span::raw(" ".repeat(inner.width.saturating_sub(14) as usize)), Span::styled("5.66 GiB", Style::default().fg(Color::White))]),
            Line::from(Span::styled("35% ██████████░░░░░░░░░░░░░░░░░░░", Style::default().fg(Color::DarkGray))),
            Line::from(""),
            Line::from(vec![Span::raw("Available:"), Span::raw(" ".repeat(inner.width.saturating_sub(18) as usize)), Span::styled("10.3 GiB", Style::default().fg(Color::White))]),
            Line::from(Span::styled("65% ███████████████████░░░░░░░░░", Style::default().fg(Color::DarkGray))),
        ]),
        inner,
    );
}

fn render_disks(frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .title(" disks ")
        .title_style(Style::default().fg(Color::DarkGray))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![Span::raw("root"), Span::raw(" ".repeat(inner.width.saturating_sub(12) as usize)), Span::styled("460 GiB", Style::default().fg(Color::White))]),
            Line::from(Span::styled("Used: 60% ████████████░░░░░░", Style::default().fg(Color::Rgb(255, 100, 100)))),
            Line::from(vec![Span::raw("swap"), Span::raw(" ".repeat(inner.width.saturating_sub(13) as usize)), Span::styled("2.44 GiB", Style::default().fg(Color::White))]),
            Line::from(Span::styled("Used: 31% ██████░░░░░░░░░░░", Style::default().fg(Color::Rgb(255, 100, 100)))),
        ]),
        inner,
    );
}

fn render_net(frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .title(" net 192.168.1.4 ")
        .title_style(Style::default().fg(Color::DarkGray))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Use safe Layout splitting instead of manual Rect math
    let cols = Layout::horizontal([
        Constraint::Percentage(50), // spacer or graph placeholder
        Constraint::Percentage(50), // text area
    ]).split(inner);

    let text_area = cols[1];

    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled("download", Style::default().fg(Color::Gray))),
            Line::from(vec![Span::raw("▼ 5.98 KiB/s"), Span::raw(" (47.8 Kbps)")]),
            Line::from(vec![Span::raw("▼ Top:      "), Span::raw(" (47.8 Kbps)")]),
            Line::from(vec![Span::raw("▼ Total:    "), Span::raw(" 2.86 MiB")]),
            Line::from(""),
            Line::from(Span::styled("upload", Style::default().fg(Color::Gray))),
            Line::from(vec![Span::raw("▲ 14.9 KiB/s"), Span::raw(" (119 Kbps)")]),
        ]),
        text_area,
    );
}
