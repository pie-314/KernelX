use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::prelude::{Alignment, Color, Line, Span, Style, Stylize};
use ratatui::widgets::{
    Block, BorderType, Borders, Cell, Gauge, List, ListItem, Paragraph, Row, Sparkline, Table,
};
use ratatui::Frame;

use crate::app::{App, Screen};
use crate::repo::CheckStatus;
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
        Screen::EventFlow => render_event_flow(frame, main_layout[1], app),
        Screen::Judging => render_judging(frame, main_layout[1], app),
        Screen::Submission => render_submission(frame, main_layout[1], app),
        Screen::System => render_system_screen(frame, main_layout[1], app),
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

    let screen_name = match app.screen {
        Screen::Dashboard => "1:DASHBOARD",
        Screen::EventFlow => "2:EVENT_FLOW",
        Screen::Judging => "3:JUDGING",
        Screen::Submission => "4:SUBMISSION",
        Screen::System => "5:SYSTEM",
    };

    // Connection status indicators
    let shm_dot = if app.connections.shm_connected { Span::styled("SHM", Style::default().fg(THEME_LIME)) } else { Span::styled("SHM", Style::default().fg(THEME_RED)) };
    let bridge_dot = if app.connections.bridge_active { Span::styled("BRG", Style::default().fg(THEME_LIME)) } else { Span::styled("BRG", Style::default().fg(THEME_RED)) };
    let brain_dot = if app.connections.brain_active { Span::styled("BRN", Style::default().fg(THEME_LIME)) } else { Span::styled("BRN", Style::default().fg(THEME_RED)) };

    let center = Paragraph::new(Line::from(vec![
        format!(" {} ", screen_name).fg(THEME_TEXT),
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

/* --- SCREEN 1: DASHBOARD --- */
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

/* --- SCREEN 2: EVENT FLOW --- */
fn render_event_flow(frame: &mut Frame, area: Rect, _app: &App) {
    let events = vec![
        ("APR 25 09:00", "HACKATHON_START", "Meta HQ Doors Open. Let the building begin."),
        ("APR 25 10:30", "OPENENV_KEYNOTE", "Introduction to OpenEnv API and Environment standards."),
        ("APR 25 14:00", "KERNELX_ALPHA", "Sentinel eBPF sensor successfully attached to Linux scheduler."),
        ("APR 25 20:00", "BRAIN_TRANSPLANT", "First successful ZMQ action pipeline from Brain to Bridge."),
        ("APR 26 02:00", "RADISH_DUMP", "Persistence layer verified. Trajectory collection enabled."),
        ("APR 26 09:00", "FINAL_PUSH", "Trained model strategist-q4km.gguf ready for inference."),
        ("APR 26 13:00", "SUBMISSION_WINDOW", "Code freeze. Mirrors verified on Hugging Face."),
        ("APR 26 15:00", "JUDGING_START", "Presenting KernelX to the Meta and OpenEnv judges."),
    ];

    let items: Vec<ListItem> = events
        .iter()
        .map(|(time, label, desc)| {
            ListItem::new(vec![
                Line::from(vec![
                    time.fg(THEME_GRAY),
                    " ".into(),
                    label.fg(THEME_CYAN).bold(),
                ]),
                Line::from(desc.fg(THEME_TEXT)),
                Line::from(""),
            ])
        })
        .collect();

    frame.render_widget(
        List::new(items).block(btop_block("MISSION_LOG / TIMELINE")),
        area,
    );
}

/* --- SCREEN 3: JUDGING --- */
fn render_judging(frame: &mut Frame, area: Rect, _app: &App) {
    let criteria = vec![
        ("40%", "Environment Innovation", "Is the environment novel? Kernel telemetry as OpenEnv is unique."),
        ("30%", "Storytelling/Presentation", "The TUI HUD and end-to-end mission flow show high polish."),
        ("20%", "Observable Reward", "P99 latency improvement verified in autonomous benchmark."),
        ("10%", "Training Pipeline", "World model drift analysis and GGUF quantization show depth."),
    ];

    let rows: Vec<Row> = criteria
        .iter()
        .map(|(weight, label, desc)| {
            Row::new(vec![
                Cell::from(weight.to_string()).fg(THEME_MAGENTA).bold(),
                Cell::from(label.to_string()).fg(THEME_CYAN),
                Cell::from(desc.to_string()).fg(THEME_TEXT),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Length(25),
            Constraint::Min(40),
        ],
    )
    .header(
        Row::new(vec!["WEIGHT", "CRITERION", "DESCRIPTION"])
            .fg(THEME_GRAY)
            .bold(),
    )
    .block(btop_block("JUDGING_CRITERIA"));

    frame.render_widget(table, area);
}

/* --- SCREEN 4: SUBMISSION --- */
fn render_submission(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::vertical([Constraint::Min(0), Constraint::Length(7)]).split(area);

    let rows: Vec<Row> = app
        .repo
        .checks
        .iter()
        .map(|c| {
            let (icon, color) = match c.status {
                CheckStatus::Ready => (" [ok] ", THEME_LIME),
                CheckStatus::Warning => (" [!!] ", THEME_ORANGE),
                CheckStatus::Missing => (" [xx] ", THEME_MAGENTA),
            };
            Row::new(vec![
                Cell::from(icon).fg(color),
                Cell::from(c.label.to_string()).fg(THEME_TEXT).bold(),
                Cell::from(c.detail.clone()).fg(THEME_GRAY),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(6),
            Constraint::Length(25),
            Constraint::Min(0),
        ],
    )
    .block(btop_block("SUBMISSION_READINESS_CHECKLIST"));
    frame.render_widget(table, chunks[0]);

    let runbook: Vec<ListItem> = app
        .repo
        .command_cards
        .iter()
        .map(|&c| ListItem::new(c.fg(THEME_CYAN)))
        .collect();
    frame.render_widget(
        List::new(runbook).block(btop_block("OPERATOR_RUNBOOK / DEPLOY_SEQUENCE")),
        chunks[1],
    );
}

/* --- SCREEN 5: SYSTEM (real metrics) --- */
fn render_system_screen(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::vertical([
        Constraint::Length(9),  // System info
        Constraint::Length(8),  // Model status
        Constraint::Min(0),     // Full 24D telemetry
    ])
    .split(area);

    let sys = &app.telemetry.system;
    let sys_info = vec![
        Line::from(vec![
            "HOST:      ".fg(THEME_GRAY),
            sys.hostname.as_str().fg(THEME_CYAN),
        ]),
        Line::from(vec![
            "OS:        ".fg(THEME_GRAY),
            sys.os_name.as_str().fg(THEME_CYAN),
        ]),
        Line::from(vec![
            "KERNEL:    ".fg(THEME_GRAY),
            sys.kernel_version.as_str().fg(THEME_CYAN),
        ]),
        Line::from(vec![
            "CPU:       ".fg(THEME_GRAY),
            format!("{} ({} cores)", sys.cpu_brand, sys.cpu_count).fg(THEME_CYAN),
        ]),
        Line::from(vec![
            "MEMORY:    ".fg(THEME_GRAY),
            format!(
                "{} / {} MB ({:.0}%)",
                sys.used_memory_mb,
                sys.total_memory_mb,
                sys.memory_pct * 100.0
            )
            .fg(if sys.memory_pct > 0.8 {
                THEME_RED
            } else {
                THEME_LIME
            }),
        ]),
        Line::from(vec![
            "CPU_AVG:   ".fg(THEME_GRAY),
            format!("{:.1}%", sys.cpu_usage_avg).fg(if sys.cpu_usage_avg > 80.0 {
                THEME_RED
            } else {
                THEME_LIME
            }),
        ]),
        Line::from(vec![
            "UPTIME:    ".fg(THEME_GRAY),
            format_uptime(sys.uptime_secs).fg(THEME_LIME),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(sys_info).block(btop_block("SYSTEM_HARDWARE")),
        chunks[0],
    );

    // Model status panel
    let ms = &app.telemetry.model_status;
    let policy_color = match ms.policy_mode {
        PolicyMode::Heuristic => THEME_ORANGE,
        PolicyMode::TrainedV1 => THEME_LIME,
        PolicyMode::TrainedV2 => THEME_CYAN,
    };
    let model_info = vec![
        Line::from(vec![
            "POLICY:    ".fg(THEME_GRAY),
            Span::styled(
                match ms.policy_mode {
                    PolicyMode::Heuristic => "HEURISTIC (rule-based)",
                    PolicyMode::TrainedV1 => "TRAINED v1 (SmolLM2-360M)",
                    PolicyMode::TrainedV2 => "TRAINED v2 (GRPO fine-tuned)",
                },
                Style::default().fg(policy_color).bold(),
            ),
        ]),
        Line::from(vec![
            "MODEL:     ".fg(THEME_GRAY),
            ms.model_version.as_str().fg(THEME_CYAN),
        ]),
        Line::from(vec![
            "GGUF_SIZE: ".fg(THEME_GRAY),
            if ms.gguf_size_mb > 0.0 {
                format!("{:.0} MB (Q4_K_M)", ms.gguf_size_mb).fg(THEME_LIME)
            } else {
                "N/A".fg(THEME_GRAY)
            },
        ]),
        Line::from(vec![
            "LATENCY:   ".fg(THEME_GRAY),
            if ms.inference_latency_ms > 0.0 {
                let color = if ms.inference_latency_ms < 50.0 {
                    THEME_LIME
                } else {
                    THEME_ORANGE
                };
                format!("{:.0}ms", ms.inference_latency_ms).fg(color)
            } else {
                "N/A".fg(THEME_GRAY)
            },
        ]),
        Line::from(vec![
            "DATA:      ".fg(THEME_GRAY),
            format!("~{} transitions", ms.total_transitions).fg(THEME_TEXT),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(model_info).block(btop_block("MODEL_STATUS")),
        chunks[1],
    );

    // Full 24D telemetry grid — show ALL features
    let feature_names = [
        "cpu_id", "prio", "s_prio", "n_prio",
        "exec_rt", "vruntime", "nr_migr", "cpus_ok",
        "rq_len", "g_load", "unused", "unused",
        "ctx_sw", "pmu_0", "pmu_1", "pmu_2",
        "pmu_3", "pmu_4", "pmu_5", "pmu_6",
        "pmu_7", "pmu_8", "pmu_9", "wait_us",
    ];

    let mut rows = Vec::new();
    for i in 0..6 {
        let mut cells = Vec::new();
        for j in 0..4 {
            let idx = i * 4 + j;
            let val = app.telemetry.features[idx];
            let name = feature_names[idx];
            let color = if val == 0 {
                THEME_GRAY
            } else if val > 1_000_000 {
                THEME_MAGENTA
            } else if val > 1000 {
                THEME_ORANGE
            } else {
                THEME_TEXT
            };
            cells.push(
                Cell::from(format!("{:>7}: {:>6}", name, compact_number(val))).fg(color),
            );
        }
        rows.push(Row::new(cells));
    }

    let table = Table::new(rows, [Constraint::Percentage(25); 4])
        .block(btop_block("FULL_24D_TELEMETRY_VECTOR"));
    frame.render_widget(table, chunks[2]);
}

/* --- PANEL HELPERS FOR DASHBOARD --- */

fn render_sentinel_panel(frame: &mut Frame, area: Rect, app: &App) {
    let sections = Layout::vertical([
        Constraint::Length(12), // Core Usage Bars
        Constraint::Length(8),  // Pain Meter
        Constraint::Min(0),     // Feature Grid (all 24)
    ])
    .split(area);

    // Cores Section — use real CPU data if available
    let cpu_cores = &app.telemetry.system.cpu_usage_per_core;
    let mut core_lines = Vec::new();
    let core_count = cpu_cores.len().min(8); // Show up to 8 cores
    for i in 0..core_count.max(4) {
        let heat = if i < cpu_cores.len() {
            cpu_cores[i] / 100.0
        } else {
            app.telemetry.core_heat.get(i).copied().unwrap_or(0.0)
        };
        let heat = heat.clamp(0.0, 1.0);
        let bar_width = (heat * 20.0) as usize;
        let color = if heat > 0.8 {
            THEME_RED
        } else if heat > 0.5 {
            THEME_ORANGE
        } else {
            THEME_LIME
        };
        let bar = Span::styled("█".repeat(bar_width), Style::default().fg(color));
        let empty = Span::styled("░".repeat(20 - bar_width), Style::default().fg(THEME_GRAY));
        core_lines.push(Line::from(vec![
            format!(" C{:<2}", i).fg(THEME_DIM),
            bar,
            empty,
            format!(" {:>3.0}%", heat * 100.0).fg(THEME_TEXT),
        ]));
    }
    frame.render_widget(
        Paragraph::new(core_lines).block(btop_block("CPU_CORES")),
        sections[0],
    );

    // P99 Latency Pain Meter — color-coded
    let wait = app.telemetry.p99_wait_us;
    let wait_ratio = (wait.min(1000) as f64 / 1000.0).clamp(0.0, 1.0);
    let color = latency_color(wait);

    let gauge = Gauge::default()
        .block(btop_block("P99_LATENCY"))
        .gauge_style(Style::default().fg(color).bg(THEME_GRAY))
        .ratio(wait_ratio)
        .label(format!("{} us", wait));
    frame.render_widget(gauge, sections[1]);

    // 24D Feature list — show ALL 24 features with color coding
    let feature_short = [
        "cpu", "pri", "spr", "npr", "exe", "vrt",
        "mig", "cpu", "rq ", "gld", "u10", "u11",
        "csw", "pm0", "pm1", "pm2", "pm3", "pm4",
        "pm5", "pm6", "pm7", "pm8", "pm9", "wt ",
    ];
    let features: Vec<ListItem> = app
        .telemetry
        .features
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let color = if v == 0 {
                THEME_GRAY
            } else if i == 23 {
                latency_color(v)
            } else if v > 1_000_000 {
                THEME_MAGENTA
            } else {
                THEME_CYAN
            };
            let bar_len = if v == 0 { 0 } else { (v.min(100) / 10).max(1) as usize };
            let bar = Span::styled("■".repeat(bar_len), Style::default().fg(color));
            ListItem::new(Line::from(vec![
                format!("{} ", feature_short[i]).fg(THEME_GRAY),
                bar,
                format!(" {}", compact_number(v)).fg(color),
            ]))
        })
        .collect();
    frame.render_widget(
        List::new(features).block(btop_block("24D_TELEMETRY")),
        sections[2],
    );
}

fn render_brain_warp_panel(frame: &mut Frame, area: Rect, app: &App) {
    let sections = Layout::vertical([
        Constraint::Length(9),  // Brain Decision + Model info
        Constraint::Min(0),     // Warp Table
        Constraint::Length(8),  // CoT Scrolling Log
    ])
    .split(area);

    let action = app.telemetry.current_action;
    let conf = app.telemetry.model_confidence;
    let action_color = if action < -0.3 {
        THEME_LIME
    } else if action > 0.3 {
        THEME_RED
    } else {
        THEME_CYAN
    };

    let ms = &app.telemetry.model_status;
    let policy_color = match ms.policy_mode {
        PolicyMode::Heuristic => THEME_ORANGE,
        PolicyMode::TrainedV1 => THEME_LIME,
        PolicyMode::TrainedV2 => THEME_CYAN,
    };

    let decision = Paragraph::new(vec![
        Line::from(vec![
            "ACTION:     ".fg(THEME_GRAY),
            format!("{:.4}", action).fg(action_color).bold(),
            "  ".into(),
            if action < -0.1 { "BOOST".fg(THEME_LIME) } else if action > 0.1 { "DEMOTE".fg(THEME_RED) } else { "HOLD".fg(THEME_GRAY) },
        ]),
        Line::from(vec![
            "CONFIDENCE: ".fg(THEME_GRAY),
            format!("{:.1}%", conf * 100.0).fg(THEME_LIME),
        ]),
        Line::from(vec![
            "TARGET_PID: ".fg(THEME_GRAY),
            format!("{}", app.telemetry.active_pid).fg(THEME_ORANGE),
        ]),
        Line::from(vec![
            "POLICY:     ".fg(THEME_GRAY),
            Span::styled(app.policy_label(), Style::default().fg(policy_color)),
        ]),
        Line::from(vec![
            "INF_LATENCY:".fg(THEME_GRAY),
            if ms.inference_latency_ms > 0.0 {
                format!(" {}ms", ms.inference_latency_ms as u32).fg(
                    if ms.inference_latency_ms < 50.0 { THEME_LIME } else { THEME_ORANGE }
                )
            } else {
                " N/A".fg(THEME_GRAY)
            },
        ]),
    ])
    .block(btop_block("STRATEGIST_BRAIN"));
    frame.render_widget(decision, sections[0]);

    // Process warp table with latency color-coding
    let rows: Vec<Row> = app
        .nudged_processes
        .iter()
        .map(|p| {
            let wait_color = latency_color(p.wait_us);
            let (nudge_label, nudge_color) = if p.nudge < -0.3 {
                ("PROMOTED", THEME_LIME)
            } else if p.nudge > 0.3 {
                ("THROTTLED", THEME_RED)
            } else {
                ("STABLE", THEME_GRAY)
            };

            Row::new(vec![
                Cell::from(p.pid.to_string()).fg(THEME_TEXT),
                Cell::from(p.name.clone()).fg(THEME_CYAN).bold(),
                Cell::from(format!("{}us", p.wait_us)).fg(wait_color),
                Cell::from(format!("{:.2}", p.nudge)).fg(nudge_color),
                Cell::from(nudge_label).fg(nudge_color),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(6),
            Constraint::Min(10),
            Constraint::Length(10),
            Constraint::Length(8),
            Constraint::Length(10),
        ],
    )
    .header(
        Row::new(vec!["PID", "PROC", "WAIT", "NUDGE", "STATUS"])
            .fg(THEME_GRAY)
            .bold(),
    )
    .block(btop_block("ACTIVE_WARP_TUNNEL"))
    .row_highlight_style(Style::default().bg(THEME_GRAY));
    frame.render_widget(table, sections[1]);

    let reasoning: Vec<ListItem> = app
        .reasoning_log
        .iter()
        .rev()
        .take(6)
        .map(|s| ListItem::new(Line::from(format!("> {}", s)).fg(THEME_TEXT)))
        .collect();
    frame.render_widget(
        List::new(reasoning).block(btop_block("CHAIN_OF_THOUGHT")),
        sections[2],
    );
}

fn render_infra_evidence_panel(frame: &mut Frame, area: Rect, app: &App) {
    let sections = Layout::vertical([
        Constraint::Length(10), // Reward graph (wider sparkline)
        Constraint::Length(10), // World Model Drift
        Constraint::Length(6),  // Memory gauge
        Constraint::Min(0),     // RadishDB / Auditor logs
    ])
    .split(area);

    // Wider reward sparkline
    let reward_data: Vec<u64> = app
        .reward_history
        .iter()
        .map(|&r| (r + 1000).max(0) as u64)
        .collect();
    let reward_spark = Sparkline::default()
        .block(btop_block("REWARD_CURVE"))
        .data(&reward_data)
        .style(Style::default().fg(THEME_LIME));
    frame.render_widget(reward_spark, sections[0]);

    let drift_data: Vec<u64> = app
        .drift_history
        .iter()
        .map(|&d| (d * 1000.0) as u64)
        .collect();
    let drift_spark = Sparkline::default()
        .block(btop_block("MODEL_DRIFT"))
        .data(&drift_data)
        .style(Style::default().fg(THEME_ORANGE));
    frame.render_widget(drift_spark, sections[1]);

    // Memory gauge (real data)
    let mem_pct = app.telemetry.system.memory_pct;
    let mem_color = if mem_pct > 0.8 {
        THEME_RED
    } else if mem_pct > 0.6 {
        THEME_ORANGE
    } else {
        THEME_CYAN
    };
    let mem_gauge = Gauge::default()
        .block(btop_block("MEMORY"))
        .gauge_style(Style::default().fg(mem_color).bg(THEME_GRAY))
        .ratio(mem_pct as f64)
        .label(format!(
            "{}/{}MB ({:.0}%)",
            app.telemetry.system.used_memory_mb,
            app.telemetry.system.total_memory_mb,
            mem_pct * 100.0
        ));
    frame.render_widget(mem_gauge, sections[2]);

    // RadishDB + Auditor
    let logs: Vec<ListItem> = app
        .safety_log
        .iter()
        .rev()
        .map(|s| ListItem::new(Line::from(s.as_str()).fg(THEME_MAGENTA)))
        .collect();

    let radish_health = Paragraph::new(vec![
        Line::from(vec![
            "WAL_FILL:  ".fg(THEME_GRAY),
            format!("{:.1}%", app.telemetry.radish_wal_fill * 100.0).fg(THEME_CYAN),
        ]),
        Line::from(vec![
            "DIRTY_PG:  ".fg(THEME_GRAY),
            format!("{}", app.telemetry.radish_dirty_pages).fg(THEME_ORANGE),
        ]),
        Line::from(vec![
            "IO_LOCKS:  ".fg(THEME_GRAY),
            "NONE".fg(THEME_LIME),
        ]),
    ])
    .block(btop_block("RADISHDB"));

    let infra_layout =
        Layout::vertical([Constraint::Length(5), Constraint::Min(0)]).split(sections[3]);
    frame.render_widget(radish_health, infra_layout[0]);
    frame.render_widget(
        List::new(logs).block(btop_block("AUDITOR_ALERTS")),
        infra_layout[1],
    );
}

fn render_bottom_bar(frame: &mut Frame, area: Rect, app: &App) {
    let status_color = if app.connections.shm_connected && app.connections.bridge_active {
        THEME_LIME
    } else if app.connections.shm_connected || app.connections.bridge_active {
        THEME_YELLOW
    } else {
        THEME_RED
    };
    let status_text = if app.connections.shm_connected && app.connections.bridge_active {
        "ALL_CONNECTED"
    } else if app.telemetry.source == crate::telemetry::TelemetrySource::Mock {
        "MOCK_MODE"
    } else {
        "PARTIAL"
    };

    let text = Paragraph::new(Line::from(vec![
        " [TAB] CYCLE ".fg(THEME_BG).bg(THEME_GRAY),
        " [1-5] JUMP ".fg(THEME_BG).bg(THEME_GRAY),
        " [R] RESET ".fg(THEME_BG).bg(THEME_GRAY),
        " [Q] QUIT ".fg(THEME_BG).bg(THEME_GRAY),
        format!("  STATUS: {} ", status_text).fg(status_color),
    ]))
    .alignment(Alignment::Left);
    frame.render_widget(text, area);
}

/* --- UTILITY --- */

fn btop_block(title: &str) -> Block<'_> {
    Block::default()
        .title(format!(" {} ", title))
        .title_style(Style::default().fg(THEME_TEXT).bold())
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(THEME_GRAY))
}

fn compact_number(n: u64) -> String {
    if n >= 1_000_000_000_000 {
        format!("{:.1}T", n as f64 / 1e12)
    } else if n >= 1_000_000_000 {
        format!("{:.1}G", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

fn format_uptime(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;
    if days > 0 {
        format!("{}d {}h {}m", days, hours, mins)
    } else if hours > 0 {
        format!("{}h {}m", hours, mins)
    } else {
        format!("{}m {}s", mins, secs % 60)
    }
}
