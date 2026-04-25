use std::fs::File;
use std::path::Path;

use bytemuck::{Pod, Zeroable};

const SHM_PATH: &str = "/dev/shm/kernelx_state";

#[repr(C, packed)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HUDState {
    pub features: [u64; 24],
    pub current_action: f32,
    pub active_pid: u32,
    pub is_clamped: u32,
    pub reasoning: [u8; 128],
    pub p99_wait_us: u64,
    
    // Real Infra Fields
    pub core_heat: [f32; 4],
    pub model_confidence: f32,
    pub world_model_drift: f32,
    pub radish_wal_size: u64,
    pub radish_dirty_pages: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TelemetrySource {
    Live,
    Mock,
}

#[derive(Clone)]
pub struct TelemetrySnapshot {
    pub source: TelemetrySource,
    pub features: [u64; 24],
    pub current_action: f32,
    pub active_pid: u32,
    pub is_clamped: bool,
    pub reasoning: String,
    pub p99_wait_us: u64,
    
    // Extended Metrics for the High-Fidelity HUD
    pub model_confidence: f32,
    pub world_model_drift: f32,
    pub radish_wal_fill: f32,
    pub radish_dirty_pages: u32,
    pub core_heat: [f32; 4],
    pub safety_logs: Vec<String>,
}

pub fn read(tick: u64) -> TelemetrySnapshot {
    #[cfg(target_os = "linux")]
    if let Ok(state) = read_live() {
        // Use real fields from SHM if they are being populated by the bridge
        // We still use mock_extended_metrics for safety_logs if they aren't in SHM
        let mock_ext = mock_extended_metrics(tick, state.p99_wait_us, state.is_clamped != 0);
        
        return TelemetrySnapshot {
            source: TelemetrySource::Live,
            features: state.features,
            current_action: state.current_action,
            active_pid: state.active_pid,
            is_clamped: state.is_clamped != 0,
            reasoning: parse_reasoning(&state.reasoning),
            p99_wait_us: state.p99_wait_us,
            // Prefer SHM fields if they are non-zero (indicating bridge is updating them)
            model_confidence: if state.model_confidence > 0.0 { state.model_confidence } else { mock_ext.model_confidence },
            world_model_drift: if state.world_model_drift > 0.0 { state.world_model_drift } else { mock_ext.world_model_drift },
            radish_wal_fill: (state.radish_wal_size as f32 / 10_000_000.0).min(1.0), // Scale 10MB to 100%
            radish_dirty_pages: state.radish_dirty_pages,
            core_heat: state.core_heat,
            safety_logs: mock_ext.safety_logs,
        };
    }

    mock_snapshot(tick)
}

#[cfg(target_os = "linux")]
fn read_live() -> Result<HUDState, Box<dyn std::error::Error>> {
    use memmap2::MmapOptions;

    let path = Path::new(SHM_PATH);
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    if mmap.len() < std::mem::size_of::<HUDState>() {
        return Err("shared memory region is smaller than HUDState".into());
    }
    let state: &HUDState = bytemuck::from_bytes(&mmap[..std::mem::size_of::<HUDState>()]);
    Ok(*state)
}

fn mock_snapshot(tick: u64) -> TelemetrySnapshot {
    let mut features = [0u64; 24];
    for (index, slot) in features.iter_mut().enumerate() {
        let wave = ((tick as f64 / 15.0) + index as f64 * 0.65).sin();
        let baseline = 40.0 + (index as f64 * 1.5);
        *slot = (baseline + wave * 25.0 + ((tick + index as u64) % 5) as f64) as u64;
    }

    let wait = 100 + ((tick as f64 / 10.0).sin().abs() * 800.0) as u64;
    let action = (tick as f32 / 12.0).sin();
    let pid = 1400 + ((tick * 13) % 47) as u32;
    let clamp = action.abs() > 0.8 && wait > 600;
    
    let reasoning = if clamp {
        format!("[SAFETY] Action {:.2} for PID {} exceeds risk threshold. Auditor clamped to 0.80.", action, pid)
    } else if action < -0.4 {
        format!("[ANALYSIS] PID {} exhibiting I/O starvation. Promoting to clear synchronous queue.", pid)
    } else if action > 0.4 {
        format!("[STRATEGY] Throttling PID {} to protect interactive latency. Model predicts {}% gain.", pid, (action * 100.0) as i32)
    } else {
        format!("[IDLE] System entropy within bounds. Maintaining nominal weights for PID {}.", pid)
    };

    let extended = mock_extended_metrics(tick, wait, clamp);

    TelemetrySnapshot {
        source: TelemetrySource::Mock,
        features,
        current_action: action,
        active_pid: pid,
        is_clamped: clamp,
        reasoning,
        p99_wait_us: wait,
        model_confidence: extended.model_confidence,
        world_model_drift: extended.world_model_drift,
        radish_wal_fill: extended.radish_wal_fill,
        radish_dirty_pages: extended.radish_dirty_pages,
        core_heat: extended.core_heat,
        safety_logs: extended.safety_logs,
    }
}

fn mock_extended_metrics(tick: u64, wait: u64, is_clamped: bool) -> TelemetrySnapshot {
    // Generate some interesting looking mock data for the deep infra sections
    let confidence = 0.85 + ((tick as f32 / 20.0).cos() * 0.12);
    let drift = 0.05 + ((tick as f32 / 15.0).sin().abs() * 0.1);
    let wal = (tick % 100) as f32 / 100.0;
    
    let mut core_heat = [0.0f32; 4];
    for i in 0..4 {
        core_heat[i] = 0.3 + ((tick as f32 / (10.0 + i as f32)).sin().abs() * 0.6);
    }

    let mut logs = Vec::new();
    if is_clamped {
        logs.push(format!("[SAFETY] AI requested nudge for PID 1. Auditor clamped to -1.0. Rule: PROTECT_INIT."));
    }
    if tick % 50 == 0 {
        logs.push(format!("[RADISH] AOF Rewrite triggered. {} keys persisted in 4ms.", 100 + (tick % 1000)));
    }
    if wait > 700 {
        logs.push(format!("[WARN] Latency anomaly detected: {}us. Transitioning to safe mode.", wait));
    }

    // This is a partial snapshot just for the extended fields
    TelemetrySnapshot {
        source: TelemetrySource::Mock, // placeholder
        features: [0; 24],
        current_action: 0.0,
        active_pid: 0,
        is_clamped: false,
        reasoning: String::new(),
        p99_wait_us: 0,
        model_confidence: confidence,
        world_model_drift: drift,
        radish_wal_fill: wal,
        radish_dirty_pages: (tick % 40) as u32,
        core_heat,
        safety_logs: logs,
    }
}

fn parse_reasoning(bytes: &[u8; 128]) -> String {
    let end = bytes.iter().position(|byte| *byte == 0).unwrap_or(bytes.len());
    let text = String::from_utf8_lossy(&bytes[..end]).trim().to_string();
    if text.is_empty() {
        "Initializing policy..." .to_string()
    } else {
        text
    }
}
