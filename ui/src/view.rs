use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::prelude::{Alignment, Color, Line, Span, Style, Stylize};
use ratatui::widgets::{
    Block, BorderType, Borders, Gauge, ListItem, Paragraph, Row, Sparkline, Table, Cell, List
};
use ratatui::Frame;

use crate::app::{App, Screen};
use crate::telemetry::PolicyMode;

// Btop-inspired High-Contrast Palette
const THEME_BG: Color = Color::Rgb(15, 15, 15);
const THEME_TEXT: Color = Color::Rgb(220, 220, 220);
const THEME_CYAN: Color = Color::Rgb(0, 255, 255);
const THEME_MAGENTA: Color = Color::Rgb(255, 0, 255);
const THEME_LIME: Color = Color::Rgb(50, 255, 50);
const THEME_ORANGE: Color = Color::Rgb(255, 165, 0);
const THEME_GRAY: Color = Color::Rgb(60, 60, 60);
const THEME_RED: Color = Color::Rgb(255, 60, 60);
const THEME_YELLOW: Color = Color::Rgb(255, 255, 0);
const THEME_DIM: Color = Color::Rgb(100, 100, 100);

/// Color-code latency: green (<10us), yellow (10-100us), red (>100us)
fn latency_color(wait_us: u64) -> Color {
    if wait_us < 10 {
        THEME_LIME
    } else if wait_us < 100 {
        THEME_YELLOW
    } else {
        THEME_RED
    }
}

pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    frame.render_widget(Block::default().bg(THEME_BG), area);

    let main_layout = Layout::vertical([
        Constraint::Length(1), // Top bar
        Constraint::Min(0),   // Main body
        Constraint::Length(1), // Bottom info
    ])
    .split(area);

    render_top_bar(frame, main_layout[0], app);

    match app.screen {
        Screen::Dashboard => render_dashboard(frame, main_layout[1], app),
    }

    render_bottom_bar(frame, main_layout[2], app);
}

fn render_top_bar(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::horizontal([
        Constraint::Percentage(25),
        Constraint::Percentage(50),
        Constraint::Percentage(25),
    ])
    .split(area);

    let left = Paragraph::new(format!(" KERNELX [v0.2.0]"))
        .fg(THEME_CYAN)
        .bold();

    // Connection status indicators
    let shm_dot = if app.connections.shm_connected { Span::styled("SHM", Style::default().fg(THEME_LIME)) } else { Span::styled("SHM", Style::default().fg(THEME_RED)) };
    let bridge_dot = if app.connections.bridge_active { Span::styled("BRG", Style::default().fg(THEME_LIME)) } else { Span::styled("BRG", Style::default().fg(THEME_RED)) };
    let brain_dot = if app.connections.brain_active { Span::styled("BRN", Style::default().fg(THEME_LIME)) } else { Span::styled("BRN", Style::default().fg(THEME_RED)) };

    let center = Paragraph::new(Line::from(vec![
        "MISSION_CONTROL ".fg(THEME_TEXT).bold(),
        "| ".fg(THEME_GRAY),
        format!("E:{:06} ", app.tick).fg(THEME_DIM),
        "| ".fg(THEME_GRAY),
        format!("{} ", app.live_mode_label()).fg(if app.live_mode_label() == "LIVE SHM" { THEME_LIME } else { THEME_ORANGE }),
        "| ".fg(THEME_GRAY),
        shm_dot, " ".into(), bridge_dot, " ".into(), brain_dot,
    ]))
    .alignment(Alignment::Center);

    // Policy mode on the right
    let policy_color = match app.telemetry.model_status.policy_mode {
        PolicyMode::Heuristic => THEME_ORANGE,
        PolicyMode::TrainedV1 => THEME_LIME,
        PolicyMode::TrainedV2 => THEME_CYAN,
    };
    let policy_label = match app.telemetry.model_status.policy_mode {
        PolicyMode::Heuristic => "HEURISTIC",
        PolicyMode::TrainedV1 => "TRAINED v1",
        PolicyMode::TrainedV2 => "TRAINED v2",
    };
    let right = Paragraph::new(Line::from(vec![
        Span::styled(policy_label, Style::default().fg(policy_color).bold()),
        " ".into(),
        app.now.format("%H:%M:%S").to_string().fg(THEME_GRAY),
        " ".into(),
    ]))
    .alignment(Alignment::Right);

    frame.render_widget(left, chunks[0]);
    frame.render_widget(center, chunks[1]);
    frame.render_widget(right, chunks[2]);
}

fn render_dashboard(frame: &mut Frame, area: Rect, app: &App) {
    let outer_cols = Layout::horizontal([
        Constraint::Percentage(30),
        Constraint::Percentage(40),
        Constraint::Percentage(30),
    ])
    .split(area);

    render_sentinel_panel(frame, outer_cols[0], app);
    render_brain_warp_panel(frame, outer_cols[1], app);
    render_infra_evidence_panel(frame, outer_cols[2], app);
}

fn render_sentinel_panel(frame: &mut Frame, area: Rect, app: &App) {
    let sections = Layout::vertical([
        Constraint::Length(12), // Core Usage Bars
        Constraint::Length(8),  // Pain Meter
        Constraint::Min(0),     // Feature Grid
    ])
    .split(area);

    let cpu_cores = &app.telemetry.system.cpu_usage_per_core;
    let mut core_lines = Vec::new();
    let core_count = cpu_cores.len().min(8); 
    for i in 0..core_count.max(4) {
        let heat = if i < cpu_cores.len() {
            cpu_cores[i] / 100.0
        } else {
            app.telemetry.core_heat.get(i).copied().unwrap_or(0.0)
        };
        let heat = heat.clamp(0.0, 1.0);
        let bar_width = (heat * 20.0) as usize;
        let color = if heat > 0.8 { THEME_RED } else if heat > 0.5 { THEME_ORANGE } else { THEME_LIME };
        let bar = Span::styled("█".repeat(bar_width), Style::default().fg(color));
        let empty = Span::styled("░".repeat(20 - bar_width), Style::default().fg(THEME_GRAY));
        core_lines.push(Line::from(vec![
            format!(" C{:<2}", i).fg(THEME_DIM),
            bar,
            empty,
            format!(" {:>3.0}%", heat * 100.0).fg(THEME_TEXT),
        ]));
    }
    frame.render_widget(Paragraph::new(core_lines).block(btop_block("CPU_CORES")), sections[0]);

    let wait = app.telemetry.p99_wait_us;
    let wait_ratio = (wait.min(1000) as f64 / 1000.0).clamp(0.0, 1.0);
    let color = latency_color(wait);
    let gauge = Gauge::default()
        .block(btop_block("P99_LATENCY"))
        .gauge_style(Style::default().fg(color).bg(THEME_GRAY))
        .ratio(wait_ratio)
        .label(format!("{} us", wait));
    frame.render_widget(gauge, sections[1]);

    let feature_short = ["cpu", "pri", "spr", "npr", "exe", "vrt", "mig", "cpu", "rq ", "gld", "u10", "u11", "csw", "pm0", "pm1", "pm2", "pm3", "pm4", "pm5", "pm6", "pm7", "pm8", "pm9", "wt "];
    let features: Vec<ListItem> = app.telemetry.features.iter().enumerate().map(|(i, &v)| {
        let color = if v == 0 { THEME_GRAY } else if i == 23 { latency_color(v) } else if v > 1_000_000 { THEME_MAGENTA } else { THEME_CYAN };
        ListItem::new(Line::from(vec![format!("{} ", feature_short[i]).fg(THEME_GRAY), format!(" {}", compact_number(v)).fg(color)]))
    }).collect();
    frame.render_widget(List::new(features).block(btop_block("24D_TELEMETRY")), sections[2]);
}

fn render_brain_warp_panel(frame: &mut Frame, area: Rect, app: &App) {
    let sections = Layout::vertical([
        Constraint::Length(9),  
        Constraint::Min(0),     
        Constraint::Length(8),  
    ])
    .split(area);

    let action = app.telemetry.current_action;
    let conf = app.telemetry.model_confidence;
    let action_color = if action < -0.3 { THEME_LIME } else if action > 0.3 { THEME_RED } else { THEME_CYAN };
    let ms = &app.telemetry.model_status;
    let policy_color = match ms.policy_mode {
        PolicyMode::Heuristic => THEME_ORANGE,
        PolicyMode::TrainedV1 => THEME_LIME,
        PolicyMode::TrainedV2 => THEME_CYAN,
    };

    let decision = Paragraph::new(vec![
        Line::from(vec!["ACTION:     ".fg(THEME_GRAY), format!("{:.4}", action).fg(action_color).bold(), "  ".into(), if action < -0.1 { "BOOST".fg(THEME_LIME) } else if action > 0.1 { "DEMOTE".fg(THEME_RED) } else { "HOLD".fg(THEME_GRAY) }]),
        Line::from(vec!["CONFIDENCE: ".fg(THEME_GRAY), format!("{:.1}%", conf * 100.0).fg(THEME_LIME)]),
        Line::from(vec!["TARGET_PID: ".fg(THEME_GRAY), format!("{}", app.telemetry.active_pid).fg(THEME_ORANGE)]),
        Line::from(vec!["POLICY:     ".fg(THEME_GRAY), Span::styled(app.policy_label(), Style::default().fg(policy_color))]),
        Line::from(vec!["INF_LATENCY:".fg(THEME_GRAY), if ms.inference_latency_ms > 0.0 { format!(" {}ms", ms.inference_latency_ms as u32).fg(if ms.inference_latency_ms < 50.0 { THEME_LIME } else { THEME_ORANGE }) } else { " N/A".fg(THEME_GRAY) }]),
    ]).block(btop_block("STRATEGIST_BRAIN"));
    frame.render_widget(decision, sections[0]);

    let rows: Vec<Row> = app.nudged_processes.iter().map(|p| {
        let wait_color = latency_color(p.wait_us);
        let (nudge_label, nudge_color) = if p.nudge < -0.3 { ("PROMOTED", THEME_LIME) } else if p.nudge > 0.3 { ("THROTTLED", THEME_RED) } else { ("STABLE", THEME_GRAY) };
        Row::new(vec![Cell::from(p.pid.to_string()).fg(THEME_TEXT), Cell::from(p.name.clone()).fg(THEME_CYAN).bold(), Cell::from(format!("{}us", p.wait_us)).fg(wait_color), Cell::from(format!("{:.2}", p.nudge)).fg(nudge_color), Cell::from(nudge_label).fg(nudge_color)])
    }).collect();

    let table = Table::new(rows, [Constraint::Length(6), Constraint::Min(10), Constraint::Length(10), Constraint::Length(8), Constraint::Length(10)])
        .header(Row::new(vec!["PID", "PROC", "WAIT", "NUDGE", "STATUS"]).fg(THEME_GRAY).bold())
        .block(btop_block("ACTIVE_WARP_TUNNEL"));
    frame.render_widget(table, sections[1]);

    let reasoning: Vec<ListItem> = app.reasoning_log.iter().rev().take(6).map(|s| ListItem::new(Line::from(format!("> {}", s)).fg(THEME_TEXT))).collect();
    frame.render_widget(List::new(reasoning).block(btop_block("CHAIN_OF_THOUGHT")), sections[2]);
}

fn render_infra_evidence_panel(frame: &mut Frame, area: Rect, app: &App) {
    let sections = Layout::vertical([
        Constraint::Length(10), 
        Constraint::Length(10), 
        Constraint::Length(6),  
        Constraint::Min(0),     
    ])
    .split(area);

    let reward_data: Vec<u64> = app.reward_history.iter().map(|&r| (r + 1000).max(0) as u64).collect();
    let reward_spark = Sparkline::default().block(btop_block("REWARD_CURVE")).data(&reward_data).style(Style::default().fg(THEME_LIME));
    frame.render_widget(reward_spark, sections[0]);

    let drift_data: Vec<u64> = app.drift_history.iter().map(|&d| (d * 1000.0) as u64).collect();
    let drift_spark = Sparkline::default().block(btop_block("MODEL_DRIFT")).data(&drift_data).style(Style::default().fg(THEME_ORANGE));
    frame.render_widget(drift_spark, sections[1]);

    let mem_pct = app.telemetry.system.memory_pct;
    let mem_color = if mem_pct > 0.8 { THEME_RED } else if mem_pct > 0.6 { THEME_ORANGE } else { THEME_CYAN };
    let mem_gauge = Gauge::default().block(btop_block("MEMORY")).gauge_style(Style::default().fg(mem_color).bg(THEME_GRAY)).ratio(mem_pct as f64).label(format!("{}/{}MB ({:.0}%)", app.telemetry.system.used_memory_mb, app.telemetry.system.total_memory_mb, mem_pct * 100.0));
    frame.render_widget(mem_gauge, sections[2]);

    let logs: Vec<ListItem> = app.safety_log.iter().rev().map(|s| ListItem::new(Line::from(s.as_str()).fg(THEME_MAGENTA))).collect();
    let radish_health = Paragraph::new(vec![Line::from(vec!["WAL_FILL:  ".fg(THEME_GRAY), format!("{:.1}%", app.telemetry.radish_wal_fill * 100.0).fg(THEME_CYAN)]), Line::from(vec!["DIRTY_PG:  ".fg(THEME_GRAY), format!("{}", app.telemetry.radish_dirty_pages).fg(THEME_ORANGE)]), Line::from(vec!["IO_LOCKS:  ".fg(THEME_GRAY), "NONE".fg(THEME_LIME)])]).block(btop_block("RADISHDB"));
    let infra_layout = Layout::vertical([Constraint::Length(5), Constraint::Min(0)]).split(sections[3]);
    frame.render_widget(radish_health, infra_layout[0]);
    frame.render_widget(List::new(logs).block(btop_block("AUDITOR_ALERTS")), infra_layout[1]);
}

fn render_bottom_bar(frame: &mut Frame, area: Rect, app: &App) {
    let status_color = if app.connections.shm_connected && app.connections.bridge_active { THEME_LIME } else if app.connections.shm_connected || app.connections.bridge_active { THEME_YELLOW } else { THEME_RED };
    let status_text = if app.connections.shm_connected && app.connections.bridge_active { "ALL_CONNECTED" } else if app.telemetry.source == crate::telemetry::TelemetrySource::Mock { "MOCK_MODE" } else { "PARTIAL" };
    let text = Paragraph::new(Line::from(vec![" [R] RESET ".fg(THEME_BG).bg(THEME_GRAY), " [Q] QUIT ".fg(THEME_BG).bg(THEME_GRAY), format!("  STATUS: {} ", status_text).fg(status_color)])).alignment(Alignment::Left);
    frame.render_widget(text, area);
}

fn btop_block(title: &str) -> Block<'_> {
    Block::default().title(format!(" {} ", title)).title_style(Style::default().fg(THEME_TEXT).bold()).borders(Borders::ALL).border_type(BorderType::Rounded).border_style(Style::default().fg(THEME_GRAY))
}

fn compact_number(n: u64) -> String {
    if n >= 1_000_000_000_000 { format!("{:.1}T", n as f64 / 1e12) } else if n >= 1_000_000_000 { format!("{:.1}G", n as f64 / 1e9) } else if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) } else if n >= 1_000 { format!("{:.1}K", n as f64 / 1e3) } else { format!("{}", n) }
}
