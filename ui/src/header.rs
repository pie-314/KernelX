use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use ratatui::widgets::Paragraph;
use ratatui::Frame;

pub fn render(frame: &mut Frame, area: Rect) {
    let cols = Layout::horizontal([
        Constraint::Length(10), // cpu
        Constraint::Length(10), // mem
        Constraint::Length(10), // net
        Constraint::Fill(1),    // clock
        Constraint::Length(12), // battery
        Constraint::Length(10), // latency
    ])
    .split(area);

    let dim = Style::default().fg(Color::DarkGray);
    let active = Style::default().fg(Color::White);

    frame.render_widget(Paragraph::new(Span::styled("cpu", active)).alignment(Alignment::Center), cols[0]);
    frame.render_widget(Paragraph::new(Span::styled("■ mem", active)).alignment(Alignment::Center), cols[1]);
    frame.render_widget(Paragraph::new(Span::styled("■ net", active)).alignment(Alignment::Center), cols[2]);

    frame.render_widget(
        Paragraph::new(Span::styled("13:55:45", Style::default().fg(Color::White))).alignment(Alignment::Center),
        cols[3],
    );

    frame.render_widget(
        Paragraph::new(Span::styled("BAT 99% █", Style::default().fg(Color::Green))).alignment(Alignment::Right),
        cols[4],
    );

    frame.render_widget(
        Paragraph::new(Span::styled("2000ms", Style::default().fg(Color::White))).alignment(Alignment::Right),
        cols[5],
    );
}
