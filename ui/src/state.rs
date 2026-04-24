//! Data layer: HUDState (shared-memory contract) and AppState (application state).

use bytemuck::{Pod, Zeroable};
use rand::Rng;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Shared-memory struct
// ---------------------------------------------------------------------------

#[repr(C, packed)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HUDState {
    pub features: [u64; 24],
    pub current_action: f32,
    pub active_pid: u32,
    pub is_clamped: u32,
    pub reasoning: [u8; 128],
    pub p99_wait_us: u64,
}

#[derive(Clone)]
pub struct UnpackedHUD {
    pub features: [u64; 24],
    pub current_action: f32,
    pub active_pid: u32,
    pub is_clamped: u32,
    pub reasoning: [u8; 128],
    pub p99_wait_us: u64,
}

impl HUDState {
    pub fn unpack(&self) -> UnpackedHUD {
        let features = { self.features };
        let current_action = { self.current_action };
        let active_pid = { self.active_pid };
        let is_clamped = { self.is_clamped };
        let reasoning = { self.reasoning };
        let p99_wait_us = { self.p99_wait_us };
        UnpackedHUD {
            features,
            current_action,
            active_pid,
            is_clamped,
            reasoning,
            p99_wait_us,
        }
    }
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

pub struct AppState {
    pub tick: u64,
    pub drift_history: VecDeque<u64>,
    pub mem_history: VecDeque<u64>,
    pub net_down_history: VecDeque<u64>,
    pub net_up_history: VecDeque<u64>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            tick: 0,
            drift_history: VecDeque::with_capacity(100),
            mem_history: VecDeque::with_capacity(100),
            net_down_history: VecDeque::with_capacity(60),
            net_up_history: VecDeque::with_capacity(60),
        }
    }

    pub fn update(&mut self, hud: &UnpackedHUD) {
        self.tick += 1;
        let mut rng = rand::thread_rng();

        // Simulate MSE loss (drift)
        let drift: u64 = hud.features.iter().take(8).sum::<u64>() / 8;
        if self.drift_history.len() >= 100 {
            self.drift_history.pop_front();
        }
        self.drift_history.push_back(drift);

        // Simulate network
        if self.net_down_history.len() >= 60 {
            self.net_down_history.pop_front();
            self.net_up_history.pop_front();
        }
        self.net_down_history.push_back(rng.gen_range(0..20000));
        self.net_up_history.push_back(rng.gen_range(0..5000));
    }
}

pub fn read_hud_state(tick: u64) -> HUDState {
    #[cfg(target_os = "linux")]
    {
        if let Ok(hud) = try_read_shm() {
            return hud;
        }
    }
    generate_mock_state(tick)
}

#[cfg(target_os = "linux")]
fn try_read_shm() -> Result<HUDState, Box<dyn std::error::Error>> {
    use memmap2::MmapOptions;
    use std::fs::File;

    let file = File::open("/dev/shm/kernelx_state")?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    if mmap.len() < std::mem::size_of::<HUDState>() {
        return Err("shm too small".into());
    }
    let state: &HUDState = bytemuck::from_bytes(&mmap[..std::mem::size_of::<HUDState>()]);
    Ok(*state)
}

fn generate_mock_state(tick: u64) -> HUDState {
    let mut rng = rand::thread_rng();
    let mut features = [0u64; 24];
    for (i, f) in features.iter_mut().enumerate() {
        let base = 50.0 + 30.0 * ((tick as f64 * 0.1 + i as f64 * 0.5).sin());
        *f = (base + rng.gen_range(-5.0..5.0)) as u64;
    }

    HUDState {
        features,
        current_action: 0.0,
        active_pid: 1000 + (tick % 50) as u32,
        is_clamped: 0,
        reasoning: [0; 128],
        p99_wait_us: 100,
    }
}
