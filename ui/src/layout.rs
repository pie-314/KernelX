use ratatui::layout::{Constraint, Layout};
use ratatui::Frame;

use crate::state::{AppState, UnpackedHUD};
use crate::{header, proc, radish_db, system, topology, world_model};

pub fn draw(frame: &mut Frame, hud: &UnpackedHUD, app: &AppState) {
    let area = frame.area();

    let main_rows = Layout::vertical([
        Constraint::Length(1),    // Minimal top bar
        Constraint::Ratio(35, 100), // Top row: Topology, World Model, Radish
        Constraint::Ratio(65, 100), // Bottom row: System (mem, disks, net), Proc list
    ])
    .split(area);

    // --- Header ---
    header::render(frame, main_rows[0]);

    // --- Top Row (3 columns) ---
    let top_cols = Layout::horizontal([
        Constraint::Ratio(30, 100), // Topology
        Constraint::Ratio(40, 100), // World Model
        Constraint::Ratio(30, 100), // Radish DB
    ])
    .split(main_rows[1]);

    topology::render(frame, top_cols[0], hud);
    world_model::render(frame, top_cols[1], app);
    radish_db::render(frame, top_cols[2]);

    // --- Bottom Row (2 columns) ---
    let bot_cols = Layout::horizontal([
        Constraint::Ratio(35, 100), // System stats
        Constraint::Ratio(65, 100), // Proc list
    ])
    .split(main_rows[2]);

    system::render(frame, bot_cols[0]);
    proc::render(frame, bot_cols[1]);
}
