use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Memory-mapped structure matching the C header for eBPF telemetry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Pod, Zeroable)]
pub struct KernelXEvent {
    pub features: [u64; 24],
    pub timestamp: u64,
    pub pid: u32,
    pub cpu: u32,
}

impl KernelXEvent {
    /// Convenience method to get wait latency from index 23
    pub fn wait_us(&self) -> u64 {
        self.features[23]
    }
}

/// Represents a single state transition for RL training.
#[derive(Serialize)]
struct Transition {
    state_t: KernelXEvent,
    action: f32,
    reward: i64,
    state_t_next: KernelXEvent,
}

pub struct TrajectoryManager {
    /// Stores the "last seen" event for each PID to calculate transitions.
    history: HashMap<u32, KernelXEvent>,
    /// Buffered writer for high-frequency JSONL logging.
    writer: BufWriter<File>,
}

impl TrajectoryManager {
    /// Initializes a new manager and opens the trajectory file in append mode.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .context("Failed to open trajectories.json")?;

        Ok(Self {
            history: HashMap::new(),
            writer: BufWriter::new(file),
        })
    }

    /// Records a state transition and persists it to disk if it meets filtering criteria.
    pub fn record_transition(&mut self, current_event: KernelXEvent) -> Result<()> {
        let pid = current_event.pid;

        // 1. Check if we have a previous state for this PID
        if let Some(old_event) = self.history.get(&pid) {
            
            // 2. Calculate Reward: Improvement in wait time (delta wait_us)
            let reward = (old_event.wait_us() as i64) - (current_event.wait_us() as i64);

            // 3. Apply Filtering Logic
            let high_pain = current_event.wait_us() > 500;
            let random_sample = rand::random::<f64>() < 0.1;

            if high_pain || random_sample {
                let transition = Transition {
                    state_t: *old_event,
                    action: 0.0,
                    reward,
                    state_t_next: current_event,
                };

                let json = serde_json::to_string(&transition)
                    .context("Failed to serialize transition")?;
                
                writeln!(self.writer, "{}", json).context("Failed to write to trajectory log")?;
            }
        }

        // 5. Update the history
        self.history.insert(pid, current_event);

        Ok(())
    }

    /// Ensures all buffered data is written to disk.
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().context("Failed to flush trajectory writer")
    }
}
