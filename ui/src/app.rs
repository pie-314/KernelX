use std::collections::VecDeque;
use std::io;
use std::path::PathBuf;

use chrono::{DateTime, Local};

use crate::repo::{scan, RepoScan};
use crate::telemetry::{read, TelemetrySnapshot, TelemetrySource};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Screen {
    Dashboard,
}

pub struct NudgedProcess {
    pub pid: u32,
    pub name: String,
    pub wait_us: u64,
    pub nudge: f32,
    pub status: &'static str,
}

pub struct App {
    pub root: PathBuf,
    pub tick: u64,
    pub screen: Screen,
    pub now: DateTime<Local>,
    pub telemetry: TelemetrySnapshot,
    pub repo: RepoScan,
    pub latency_history: VecDeque<u64>,
    pub reward_history: VecDeque<i64>,
    pub drift_history: VecDeque<f32>,
    pub confidence_history: VecDeque<f32>,
    pub reasoning_log: VecDeque<String>,
    pub safety_log: VecDeque<String>,
    pub nudged_processes: Vec<NudgedProcess>,
}

impl App {
    pub fn new() -> io::Result<Self> {
        let root = std::env::current_dir()?;
        let now = Local::now();
        let telemetry = read(0);
        Ok(Self {
            repo: scan(&root),
            root,
            tick: 0,
            screen: Screen::Dashboard,
            now,
            telemetry,
            latency_history: VecDeque::from([0]),
            reward_history: VecDeque::from([0]),
            drift_history: VecDeque::from([0.0]),
            confidence_history: VecDeque::from([0.0]),
            reasoning_log: VecDeque::new(),
            safety_log: VecDeque::new(),
            nudged_processes: Vec::new(),
        })
    }

    pub fn refresh(&mut self) {
        self.tick += 1;
        self.now = Local::now();
        self.telemetry = read(self.tick);
        
        // Histories
        self.push_latency(self.telemetry.p99_wait_us);
        self.push_reward_delta();
        self.push_drift(self.telemetry.world_model_drift);
        self.push_confidence(self.telemetry.model_confidence);
        
        // Logs
        if !self.telemetry.reasoning.is_empty() {
            if self.reasoning_log.back() != Some(&self.telemetry.reasoning) {
                if self.reasoning_log.len() >= 20 { self.reasoning_log.pop_front(); }
                self.reasoning_log.push_back(self.telemetry.reasoning.clone());
            }
        }
        
        for log in &self.telemetry.safety_logs {
            if self.safety_log.len() >= 10 { self.safety_log.pop_front(); }
            self.safety_log.push_back(log.clone());
        }

        // Active Nudges simulation/tracking
        self.update_nudged_processes();

        if self.tick % 80 == 0 { // Scan repo less frequently now at 10Hz
            self.repo = scan(&self.root);
        }
    }

    pub fn live_mode_label(&self) -> &'static str {
        match self.telemetry.source {
            TelemetrySource::Live => "LIVE SHM",
            TelemetrySource::Mock => "MOCK DEMO",
        }
    }

    fn push_latency(&mut self, wait: u64) {
        if self.latency_history.len() >= 100 {
            self.latency_history.pop_front();
        }
        self.latency_history.push_back(wait);
    }

    fn push_reward_delta(&mut self) {
        let previous = self
            .latency_history
            .iter()
            .rev()
            .nth(1)
            .copied()
            .unwrap_or(self.telemetry.p99_wait_us);
        let delta = previous as i64 - self.telemetry.p99_wait_us as i64;
        if self.reward_history.len() >= 100 {
            self.reward_history.pop_front();
        }
        self.reward_history.push_back(delta);
    }

    fn push_drift(&mut self, drift: f32) {
        if self.drift_history.len() >= 100 { self.drift_history.pop_front(); }
        self.drift_history.push_back(drift);
    }

    fn push_confidence(&mut self, conf: f32) {
        if self.confidence_history.len() >= 100 { self.confidence_history.pop_front(); }
        self.confidence_history.push_back(conf);
    }

    fn update_nudged_processes(&mut self) {
        // Keep the list stable but update values
        let names = ["postgres", "rustc", "hyprland", "docker", "python", "node", "go", "cargo"];
        let pids = [1737, 7429, 1687, 8821, 5501, 3320, 9912, 1205];
        
        self.nudged_processes.clear();
        for i in 0..4 {
            let pid = pids[(self.tick as usize / 50 + i) % pids.len()];
            let name = names[(self.tick as usize / 50 + i) % names.len()];
            
            let nudge = if pid == self.telemetry.active_pid {
                self.telemetry.current_action
            } else {
                ((self.tick + pid as u64) % 20) as f32 / 10.0 - 1.0
            };

            let status = if nudge < -0.3 { "PROMOTED" } else if nudge > 0.3 { "DEMOTED" } else { "NEUTRAL" };
            
            self.nudged_processes.push(NudgedProcess {
                pid,
                name: name.to_string(),
                wait_us: 100 + (pid as u64 % 1000) + (self.tick % 200),
                nudge,
                status,
            });
        }
    }

    pub fn reset(&mut self) {
        self.tick = 0;
        self.latency_history.clear();
        self.reward_history.clear();
        self.drift_history.clear();
        self.confidence_history.clear();
        self.reasoning_log.clear();
        self.safety_log.clear();
        self.latency_history.push_back(0);
        self.reward_history.push_back(0);
        self.drift_history.push_back(0.0);
        self.confidence_history.push_back(0.0);
    }
}
