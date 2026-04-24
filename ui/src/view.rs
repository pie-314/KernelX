use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::prelude::{Alignment, Color, Line, Style, Stylize};
use ratatui::widgets::{
    Block, Borders, Cell, Gauge, List, ListItem, Paragraph, Row, Sparkline,
    Table, BorderType,
};
use ratatui::Frame;

use crate::app::App;

// Btop-inspired High-Contrast Palette
const THEME_BG: Color = Color::Rgb(15, 15, 15);
const THEME_TEXT: Color = Color::Rgb(220, 220, 220);
const THEME_CYAN: Color = Color::Rgb(0, 255, 255);
const THEME_MAGENTA: Color = Color::Rgb(255, 0, 255);
const THEME_LIME: Color = Color::Rgb(50, 255, 50);
const THEME_ORANGE: Color = Color::Rgb(255, 165, 0);
const THEME_GRAY: Color = Color::Rgb(60, 60, 60);

pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    
    // Main background clear
    frame.render_widget(Block::default().bg(THEME_BG), area);

    let main_layout = Layout::vertical([
        Constraint::Length(1), // Top bar
        Constraint::Min(0),    // Main body
        Constraint::Length(1), // Bottom info
    ])
    .split(area);

    render_top_bar(frame, main_layout[0], app);
    render_main_body(frame, main_layout[1], app);
    render_bottom_bar(frame, main_layout[2], app);
}

fn render_top_bar(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::horizontal([
        Constraint::Percentage(20),
        Constraint::Percentage(60),
        Constraint::Percentage(20),
    ]).split(area);

    let left = Paragraph::new(format!(" KERNELX [v0.1.0]")).fg(THEME_CYAN).bold();
    let center = Paragraph::new(format!(" EPOCH: {:06} | MODE: {} ", app.tick, app.live_mode_label()))
        .alignment(Alignment::Center)
        .fg(THEME_TEXT);
    let right = Paragraph::new(format!("{}", app.now.format("%H:%M:%S")))
        .alignment(Alignment::Right)
        .fg(THEME_GRAY);

    frame.render_widget(left, chunks[0]);
    frame.render_widget(center, chunks[1]);
    frame.render_widget(right, chunks[2]);
}

fn render_main_body(frame: &mut Frame, area: Rect, app: &App) {
    let outer_cols = Layout::horizontal([
        Constraint::Percentage(30), // Left: CPU / Sentinel
        Constraint::Percentage(40), // Mid: Brain / Process Warp
        Constraint::Percentage(30), // Right: Infra / Evidence
    ]).split(area);

    render_sentinel_panel(frame, outer_cols[0], app);
    render_brain_warp_panel(frame, outer_cols[1], app);
    render_infra_evidence_panel(frame, outer_cols[2], app);
}

/* --- LEFT PANEL: SENTINEL & TELEMETRY --- */
fn render_sentinel_panel(frame: &mut Frame, area: Rect, app: &App) {
    let sections = Layout::vertical([
        Constraint::Length(12), // Core Usage Bars
        Constraint::Length(8),  // Pain Meter
        Constraint::Min(0),     // Feature Grid
    ]).split(area);

    // Cores Section (Btop Style)
    let mut core_lines = Vec::new();
    for i in 0..4 {
        let heat = app.telemetry.core_heat[i];
        let bar_width = (heat * 20.0) as usize;
        let bar = "█".repeat(bar_width).fg(if heat > 0.8 { THEME_MAGENTA } else { THEME_LIME });
        let empty = "░".repeat(20 - bar_width).fg(THEME_GRAY);
        core_lines.push(Line::from(vec![
            format!(" Core{} ", i).fg(THEME_TEXT),
            bar,
            empty,
            format!(" {:>3.0}%", heat * 100.0).fg(THEME_TEXT),
        ]));
    }
    frame.render_widget(Paragraph::new(core_lines).block(btop_block("CPU_CORES")), sections[0]);

    // Pain Meter
    let wait = app.telemetry.p99_wait_us;
    let wait_ratio = (wait.min(1000) as f64 / 1000.0).clamp(0.0, 1.0);
    let color = if wait_ratio > 0.7 { THEME_MAGENTA } else if wait_ratio > 0.4 { THEME_ORANGE } else { THEME_CYAN };
    
    let gauge = Gauge::default()
        .block(btop_block("P99_LATENCY_WAIT"))
        .gauge_style(Style::default().fg(color).bg(THEME_GRAY))
        .ratio(wait_ratio)
        .label(format!("{} μs", wait));
    frame.render_widget(gauge, sections[1]);

    // 24D Feature Radar (Dense Sparklines)
    let features: Vec<ListItem> = app.telemetry.features.iter().enumerate().take(12).map(|(i, &v)| {
        let bar_len = (v.min(100) / 10) as usize;
        let bar = "■".repeat(bar_len).fg(THEME_CYAN);
        ListItem::new(Line::from(vec![
            format!("F{:02} ", i).fg(THEME_GRAY),
            bar,
            format!(" {}", v).fg(THEME_TEXT),
        ]))
    }).collect();
    frame.render_widget(List::new(features).block(btop_block("24D_TELEMETRY")), sections[2]);
}

/* --- CENTER PANEL: BRAIN & WARP TABLE --- */
fn render_brain_warp_panel(frame: &mut Frame, area: Rect, app: &App) {
    let sections = Layout::vertical([
        Constraint::Length(7),  // Brain Decision (Gauge + Stats)
        Constraint::Min(0),     // Warp Table (The "Process" list)
        Constraint::Length(8),  // CoT Scrolling Log
    ]).split(area);

    // Decision Logic
    let action = app.telemetry.current_action;
    let conf = app.telemetry.model_confidence;
    let action_color = if action < 0.0 { THEME_CYAN } else { THEME_MAGENTA };
    
    let decision = Paragraph::new(vec![
        Line::from(vec!["NUDGE_ACTION: ".fg(THEME_GRAY), format!("{:.4}", action).fg(action_color).bold()]),
        Line::from(vec!["CONFIDENCE:   ".fg(THEME_GRAY), format!("{:.1}%", conf * 100.0).fg(THEME_LIME)]),
        Line::from(vec!["TARGET_PID:   ".fg(THEME_GRAY), format!("{}", app.telemetry.active_pid).fg(THEME_ORANGE)]),
    ]).block(btop_block("STRATEGIST_BRAIN"));
    frame.render_widget(decision, sections[0]);

    // Warp Table (btop-style process list)
    let rows: Vec<Row> = app.nudged_processes.iter().map(|p| {
        let (nudge_label, color) = if p.nudge < -0.3 {
            ("PROMOTED", THEME_LIME)
        } else if p.nudge > 0.3 {
            ("THROTTLED", THEME_MAGENTA)
        } else {
            ("STABLE", THEME_GRAY)
        };
        
        Row::new(vec![
            Cell::from(p.pid.to_string()).fg(THEME_TEXT),
            Cell::from(p.name.clone()).fg(THEME_CYAN).bold(),
            Cell::from(format!("{}μs", p.wait_us)).fg(THEME_TEXT),
            Cell::from(format!("{:.2}", p.nudge)).fg(color),
            Cell::from(nudge_label).fg(color),
        ])
    }).collect();

    let table = Table::new(rows, [
        Constraint::Length(6),
        Constraint::Min(10),
        Constraint::Length(10),
        Constraint::Length(8),
        Constraint::Length(10),
    ])
    .header(Row::new(vec!["PID", "PROC", "WAIT", "NUDGE", "STATUS"]).fg(THEME_GRAY).bold())
    .block(btop_block("ACTIVE_WARP_TUNNEL"))
    .row_highlight_style(Style::default().bg(THEME_GRAY));
    frame.render_widget(table, sections[1]);

    // Scrolling Logic
    let reasoning: Vec<ListItem> = app.reasoning_log.iter().rev().take(6).map(|s| {
        ListItem::new(Line::from(format!("> {}", s)).fg(THEME_TEXT))
    }).collect();
    frame.render_widget(List::new(reasoning).block(btop_block("CHAIN_OF_THOUGHT")), sections[2]);
}

/* --- RIGHT PANEL: INFRA & EVIDENCE --- */
fn render_infra_evidence_panel(frame: &mut Frame, area: Rect, app: &App) {
    let sections = Layout::vertical([
        Constraint::Length(10), // Cumulative Reward Graph
        Constraint::Length(10), // World Model Drift
        Constraint::Min(0),     // RadishDB / Auditor logs
    ]).split(area);

    // Reward Sparkline (Vibrant Lime)
    let reward_data: Vec<u64> = app.reward_history.iter().map(|&r| (r + 1000).max(0) as u64).collect();
    let reward_spark = Sparkline::default()
        .block(btop_block("EVIDENCE: REWARD_CURVE"))
        .data(&reward_data)
        .style(Style::default().fg(THEME_LIME));
    frame.render_widget(reward_spark, sections[0]);

    // Drift Sparkline (Vibrant Orange)
    let drift_data: Vec<u64> = app.drift_history.iter().map(|&d| (d * 1000.0) as u64).collect();
    let drift_spark = Sparkline::default()
        .block(btop_block("EVIDENCE: MODEL_DRIFT"))
        .data(&drift_data)
        .style(Style::default().fg(THEME_ORANGE));
    frame.render_widget(drift_spark, sections[1]);

    // Infra Health Section
    let logs: Vec<ListItem> = app.safety_log.iter().rev().map(|s| {
        ListItem::new(Line::from(s.as_str()).fg(THEME_MAGENTA))
    }).collect();
    
    let radish_health = Paragraph::new(vec![
        Line::from(vec!["WAL_FILL:  ".fg(THEME_GRAY), format!("{:.1}%", app.telemetry.radish_wal_fill * 100.0).fg(THEME_CYAN)]),
        Line::from(vec!["DIRTY_PG:  ".fg(THEME_GRAY), format!("{}", app.telemetry.radish_dirty_pages).fg(THEME_ORANGE)]),
        Line::from(vec!["IO_LOCKS:  ".fg(THEME_GRAY), "NONE".fg(THEME_LIME)]),
    ]).block(btop_block("RADISHDB_STATUS"));
    
    let infra_layout = Layout::vertical([Constraint::Length(5), Constraint::Min(0)]).split(sections[2]);
    frame.render_widget(radish_health, infra_layout[0]);
    frame.render_widget(List::new(logs).block(btop_block("AUDITOR_ALERTS")), infra_layout[1]);
}

fn render_bottom_bar(frame: &mut Frame, area: Rect, _app: &App) {
    let text = Paragraph::new(Line::from(vec![
        " [Q] QUIT ".fg(THEME_BG).bg(THEME_GRAY),
        " [R] RESET_STATS ".fg(THEME_BG).bg(THEME_GRAY),
        " [C] CONFIG ".fg(THEME_BG).bg(THEME_GRAY),
        "  SYSTEM_STATUS: OK ".fg(THEME_LIME),
    ])).alignment(Alignment::Left);
    frame.render_widget(text, area);
}

// Custom btop-style block helper
fn btop_block(title: &str) -> Block<'_> {
    Block::default()
        .title(format!(" {} ", title))
        .title_style(Style::default().fg(THEME_TEXT).bold())
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(THEME_GRAY))
}
