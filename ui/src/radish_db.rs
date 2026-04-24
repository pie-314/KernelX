use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

pub fn render(frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .title(" RADISH-DB: WAL STATUS ")
        .title_style(Style::default().fg(Color::DarkGray))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(inner);

    // Left Side: BufferMap and Stats
    let left = cols[0];
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled("Widget: BufferMap", Style::default().fg(Color::Gray))),
            Line::from(Span::styled("source: /dev/shm/kernelx_state", Style::default().fg(Color::DarkGray))),
            Line::from(""),
            Line::from(Span::styled("Fields", Style::default().fg(Color::Gray))),
            Line::from(vec![
                Span::raw("Dirty Pages: "),
                Span::styled("128 MB (9%) ", Style::default().fg(Color::White)),
                Span::styled("█░░░░░░░░░", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![
                Span::raw("Commit Index: "),
                Span::styled("0x0A284C10", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::raw("Disk IO:     "),
                Span::styled("1.5 GB/s", Style::default().fg(Color::White)),
            ]),
            Line::from(""),
            Line::from(Span::styled("█".repeat(left.width as usize), Style::default().fg(Color::DarkGray))),
            Line::from(vec![
                Span::styled("████", Style::default().fg(Color::Rgb(255, 150, 100))),
                Span::styled("████████", Style::default().fg(Color::Rgb(100, 200, 100))),
                Span::styled("████", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("[X] Dirty  [] Free  [-] Pinned", Style::default().fg(Color::DarkGray)),
            ]),
        ]),
        left,
    );

    // Right Side: M2 Ultra stats
    let right = cols[1];
    let mut right_lines = vec![
        Line::from(Span::styled(" M2 Ultra", Style::default().fg(Color::Gray))),
    ];
    
    let percentages = [16, 32, 25, 15, 10, 9, 14, 1, 16, 7, 26, 7];
    for (i, p) in percentages.iter().enumerate() {
        let bar_len = (p * 15) / 100;
        let bar = "█".repeat(bar_len) + &"░".repeat(15 - bar_len);
        right_lines.push(Line::from(vec![
            Span::raw(format!("C{:<2} ", i)),
            Span::styled(bar, Style::default().fg(Color::DarkGray)),
            Span::raw(format!(" {:>3}%", p)),
        ]));
    }
    
    right_lines.push(Line::from(Span::raw("  Load avg: 2.50 2.66 2.67")));

    frame.render_widget(Paragraph::new(right_lines), right);
}
