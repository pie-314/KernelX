use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders, Cell, Row, Table};
use ratatui::Frame;

pub fn render(frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .title(" proc filter ")
        .title_style(Style::default().fg(Color::DarkGray))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let header = Row::new(vec![
        "Pid", "Program", "Command", "Threads", "User", "MemB", "Cpu%",
    ])
    .style(Style::default().fg(Color::White))
    .bottom_margin(0);

    let rows = vec![
        Row::new(vec!["33428", "Brave Browser", "/Applications/Brave Browser", "23", "subh", "155M", "0.0"]),
        Row::new(vec!["42712", "WhatsApp", "/Applications/WhatsApp.app", "27", "subh", "223M", "0.2"]),
        Row::new(vec!["38216", "Brave Browser", "/Applications/Brave Browser", "19", "subh", "152M", "0.0"]),
        Row::new(vec!["47787", "Brave Browser", "/Applications/Brave Browser", "19", "subh", "302M", "0.0"]),
        Row::new(vec!["48876", "screencaptureui", "/System/Library/CoreServices", "5", "subh", "32M", "0.0"]),
        Row::new(vec!["33424", "Brave Browser", "/Applications/Brave Browser", "20", "subh", "133M", "0.0"]),
        Row::new(vec!["464", "loginwindow", "/System/Library/CoreServices", "6", "subh", "52M", "0.2"]),
        Row::new(vec!["22684", "Discord Helper", "/Applications/Discord.app", "51", "subh", "207M", "0.8"]),
        Row::new(vec!["42703", "Brave Browser", "/Applications/Brave Browser", "28", "subh", "152M", "0.9"]),
        Row::new(vec!["42589", "screencapture", "/usr/sbin/screencapture", "5", "subh", "75M", "0.5"]),
        Row::new(vec!["47695", "Brave Browser", "/Applications/Brave Browser", "23", "subh", "153M", "0.0"]),
        Row::new(vec!["42788", "Antigravity", "/Applications/Antigravity", "32", "subh", "152M", "0.0"]),
        Row::new(vec!["42817", "rust-analyzer", "/Users/subh/.antigravity/", "33", "subh", "43M", "0.0"]),
        Row::new(vec!["42722", "deleted", "/System/Library/PrivateFramew", "2", "subh", "25M", "0.6"]),
        Row::new(vec!["14836", "Stats", "/Applications/Stats.app", "15", "subh", "73M", "0.5"]),
        Row::new(vec!["33414", "Brave Browser", "/Applications/Brave Browser", "49", "subh", "184M", "0.0"]),
        Row::new(vec!["42978", "AudioComponent", "/System/Library/Frameworks", "3", "subh", "9.4M", "0.0"]),
        Row::new(vec!["41851", "language_serve", "/Applications/Antigravity", "16", "subh", "174M", "0.1"]),
    ];

    let table = Table::new(
        rows,
        &[
            Constraint::Length(6),
            Constraint::Length(16),
            Constraint::Length(30),
            Constraint::Length(8),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(5),
        ],
    )
    .header(header)
    .block(block)
    .style(Style::default().fg(Color::DarkGray));

    frame.render_widget(table, area);
}
