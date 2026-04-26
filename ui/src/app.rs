use std::collections::VecDeque;
use std::io;
use std::path::PathBuf;

use chrono::{DateTime, Local};
use sysinfo::System;

use crate::repo::{scan, RepoScan};
use crate::telemetry::{read, ConnectionStatus, TelemetrySnapshot, TelemetrySource};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Screen {
    Dashboard,
    EventFlow,
    Judging,
    Submission,
    System,
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
    pub connections: ConnectionStatus,
    sys: System,
}

impl App {
    pub fn new() -> io::Result<Self> {
        let root = std::env::current_dir()?;
        let now = Local::now();
        let mut sys = System::new_all();
        sys.refresh_all();
        let telemetry = read(0, &mut sys);
        let connections = telemetry.connections.clone();
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
            connections,
            sys,
        })
    }

    pub fn refresh(&mut self) {
        self.tick += 1;
        self.now = Local::now();

        // Refresh sysinfo every 10 ticks (1 second at 10Hz)
        if self.tick % 10 == 0 {
            self.sys.refresh_cpu_all();
            self.sys.refresh_memory();
        }

        self.telemetry = read(self.tick, &mut self.sys);

        // Update cached connection status (checked every 50 ticks / 5 seconds)
        if self.tick % 50 == 0 {
            self.connections = self.telemetry.connections.clone();
        }

        // Histories
        self.push_latency(self.telemetry.p99_wait_us);
        self.push_reward_delta();
        self.push_drift(self.telemetry.world_model_drift);
        self.push_confidence(self.telemetry.model_confidence);

        // Logs
        if !self.telemetry.reasoning.is_empty()
            && self.reasoning_log.back() != Some(&self.telemetry.reasoning)
        {
            if self.reasoning_log.len() >= 20 {
                self.reasoning_log.pop_front();
            }
            self.reasoning_log
                .push_back(self.telemetry.reasoning.clone());
        }

        for log in &self.telemetry.safety_logs {
            if self.safety_log.len() >= 10 {
                self.safety_log.pop_front();
            }
            self.safety_log.push_back(log.clone());
        }

        // Active Nudges tracking
        self.update_nudged_processes();

        if self.tick % 80 == 0 {
            self.repo = scan(&self.root);
        }
    }

    pub fn live_mode_label(&self) -> &'static str {
        match self.telemetry.source {
            TelemetrySource::Live => "LIVE SHM",
            TelemetrySource::Mock => "MOCK DEMO",
        }
    }

    pub fn policy_label(&self) -> &str {
        &self.telemetry.model_status.model_version
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
        if self.drift_history.len() >= 100 {
            self.drift_history.pop_front();
        }
        self.drift_history.push_back(drift);
    }

    fn push_confidence(&mut self, conf: f32) {
        if self.confidence_history.len() >= 100 {
            self.confidence_history.pop_front();
        }
        self.confidence_history.push_back(conf);
    }

    fn update_nudged_processes(&mut self) {
        let active_pid = self.telemetry.active_pid;
        let mut processes = Vec::new();

        // 1. Always include the active target if it exists
        if active_pid > 0 {
            let name = get_process_name(active_pid).unwrap_or_else(|| "unknown".to_string());
            processes.push(NudgedProcess {
                pid: active_pid,
                name,
                wait_us: self.telemetry.p99_wait_us,
                nudge: self.telemetry.current_action,
                status: if self.telemetry.current_action < -0.3 {
                    "PROMOTED"
                } else if self.telemetry.current_action > 0.3 {
                    "THROTTLED"
                } else {
                    "STABLE"
                },
            });
        }

        // 2. Read real processes on Linux, fall back to mock
        #[cfg(target_os = "linux")]
        {
            self.sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
            for (pid, proc_info) in self.sys.processes() {
                if processes.len() >= 6 {
                    break;
                }
                if pid.as_u32() == active_pid {
                    continue;
                }
                // Only show processes with notable CPU usage
                if proc_info.cpu_usage() > 1.0 {
                    let nudge = ((self.tick + pid.as_u32() as u64) % 15) as f32 / 50.0 - 0.1;
                    processes.push(NudgedProcess {
                        pid: pid.as_u32(),
                        name: proc_info.name().to_string_lossy().to_string(),
                        wait_us: 10 + (pid.as_u32() as u64 % 500) + (self.tick % 50),
                        nudge,
                        status: "NEUTRAL",
                    });
                }
            }
        }

        // Mock fallback (non-Linux or no real processes found)
        if processes.len() <= 1 {
            let candidates = [
                ("postgres", 1024),
                ("rustc", 2048),
                ("docker", 3072),
                ("python3", 4096),
                ("node", 5120),
                ("cargo", 6144),
                ("systemd", 1),
                ("kswapd0", 50),
            ];

            for (name, base_pid) in candidates {
                if processes.len() >= 6 {
                    break;
                }
                if base_pid == active_pid as i32 {
                    continue;
                }
                let nudge = ((self.tick + base_pid as u64) % 15) as f32 / 50.0 - 0.1;
                processes.push(NudgedProcess {
                    pid: base_pid as u32,
                    name: name.to_string(),
                    wait_us: 10 + (base_pid as u64 % 500) + (self.tick % 50),
                    nudge,
                    status: "NEUTRAL",
                });
            }
        }

        self.nudged_processes = processes;
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

fn get_process_name(pid: u32) -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let comm_path = format!("/proc/{}/comm", pid);
        if let Ok(name) = fs::read_to_string(comm_path) {
            return Some(name.trim().to_string());
        }
    }
    let _ = pid;
    None
}
