use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph};
use ratatui::Frame;

use crate::state::AppState;

const MAGENTA: Color = Color::Rgb(255, 0, 255);

pub fn render(frame: &mut Frame, area: Rect, app: &AppState) {
    let block = Block::default()
        .title(" WORLD MODEL: PREDICTION ERROR ")
        .title_style(Style::default().fg(Color::DarkGray))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Meta details
    let meta_area = Rect::new(inner.x, inner.y, inner.width, 2);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Widget: LineChart", Style::default().fg(Color::Gray)),
                Span::raw(" ".repeat(inner.width.saturating_sub(35) as usize)),
                Span::styled("ms: 13.52 ms", Style::default().fg(MAGENTA)),
            ]),
            Line::from(Span::styled("metric: mse_loss", Style::default().fg(Color::DarkGray))),
        ]),
        meta_area,
    );

    let chart_area = Rect::new(inner.x, inner.y + 3, inner.width, inner.height.saturating_sub(3));

    let data: Vec<(f64, f64)> = app
        .drift_history
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v as f64))
        .collect();

    let dataset = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(MAGENTA))
        .data(&data);

    let max_val = data.iter().map(|d| d.1).fold(100.0, f64::max);

    let chart = Chart::new(vec![dataset])
        .x_axis(
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, 100.0])
                .labels(vec![Span::from("0.76ms"), Span::from("338ms")]),
        )
        .y_axis(
            Axis::default()
                .title(Span::styled("Prediction Drift (ms)", Style::default().fg(Color::DarkGray)))
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, max_val])
                .labels(vec![
                    Span::from("20"),
                    Span::from("40"),
                    Span::from("70"),
                    Span::from("90"),
                    Span::from("100"),
                ]),
        );

    frame.render_widget(chart, chart_area);
}
