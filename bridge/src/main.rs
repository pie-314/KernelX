/**
 * KernelX Bridge: Final "Metal" implementation with real telemetry and ZMQ control loop.
 * Updated with recording flags, immediate SHM sync, and robust shutdown.
 */
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use aya::{maps::{RingBuf, HashMap as BpfHashMap}, programs::TracePoint, Ebpf};
use aya_log::EbpfLogger;
use bytemuck::{Pod, Zeroable};
use kernelx_bridge::trajectories::{KernelXEvent, TrajectoryManager};
use kernelx_bridge::persistence;
use memmap2::MmapMut;

const DEFAULT_BPF_OBJECT: &str = "../kernel/sentinel.bpf.o";
const SHM_PATH: &str = "/dev/shm/kernelx_state";
const RADISH_AOF_PATH: &str = "radish.aof";
const ZMQ_ADDR: &str = "tcp://*:5555";

#[repr(C, packed)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HUDState {
    features: [u64; 24],
    current_action: f32,
    active_pid: u32,
    is_clamped: u32,
    reasoning: [u8; 128],
    p99_wait_us: u64,
    
    // Real Infra Fields
    core_heat: [f32; 4],
    model_confidence: f32,
    world_model_drift: f32,
    radish_wal_size: u64,
    radish_dirty_pages: u32,
}

struct CpuTracker {
    last_total: [u64; 4],
    last_idle: [u64; 4],
}

impl CpuTracker {
    fn new() -> Self {
        Self { last_total: [0; 4], last_idle: [0; 4] }
    }

    fn update(&mut self) -> [f32; 4] {
        let mut usage = [0.0f32; 4];
        if let Ok(content) = fs::read_to_string("/proc/stat") {
            for (i, line) in content.lines().skip(1).take(4).enumerate() {
                let parts: Vec<u64> = line.split_whitespace()
                    .skip(1)
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if parts.len() >= 7 {
                    let idle = parts[3];
                    let total: u64 = parts.iter().sum();
                    
                    let diff_total = total - self.last_total[i];
                    let diff_idle = idle - self.last_idle[i];
                    
                    if diff_total > 0 {
                        usage[i] = 1.0 - (diff_idle as f32 / diff_total as f32);
                    }
                    
                    self.last_total[i] = total;
                    self.last_idle[i] = idle;
                }
            }
        }
        usage
    }
}

fn main() -> Result<()> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        println!("\n[Bridge] Shutdown signal received. Closing cleanly...");
    }).expect("Error setting Ctrl-C handler");

    // 1. Initialize RadishDB
    persistence::init(RADISH_AOF_PATH).map_err(|e| anyhow::anyhow!(e))?;

    // 2. Initialize SHM
    let shm_file = fs::OpenOptions::new().read(true).write(true).create(true).open(SHM_PATH)?;
    shm_file.set_len(std::mem::size_of::<HUDState>() as u64)?;
    let mmap_raw = unsafe { MmapMut::map_mut(&shm_file)? };
    let mmap_mutex = Arc::new(Mutex::new(mmap_raw));
    
    let shared_state = Arc::new(Mutex::new(HUDState::zeroed()));

    // 3. Parse Arguments
    let args: Vec<String> = std::env::args().collect();
    let should_record = args.iter().any(|arg| arg == "--record");
    
    let object_path = args.iter()
        .skip(1)
        .find(|arg| !arg.starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_BPF_OBJECT));

    // 4. Initialize Trajectory Manager if requested
    let mut manager = if should_record {
        println!("[Bridge] Recording enabled. Saving to trajectories.json");
        Some(TrajectoryManager::new("trajectories.json")?)
    } else {
        None
    };

    // 5. Start ZMQ Control Listener (Brain -> Bridge)
    let z_state = shared_state.clone();
    let z_running = running.clone();
    let z_mmap = mmap_mutex.clone();

    // 6. Load eBPF
    println!("[Bridge] Loading eBPF object: {}", object_path.display());
    let object = fs::read(&object_path).with_context(|| "Failed to read BPF object")?;
    let mut bpf = Ebpf::load(&object).context("Failed to parse BPF object")?;
    
    if let Err(err) = EbpfLogger::init(&mut bpf) {
        eprintln!("[Warn] aya-log unavailable: {err}");
    }

    let priority_actions: BpfHashMap<aya::maps::MapData, u32, i64> = BpfHashMap::try_from(bpf.take_map("priority_actions").context("Map priority_actions not found")?)?;
    let pa_mutex = Arc::new(Mutex::new(priority_actions));
    let pa_clone = pa_mutex.clone();

    thread::spawn(move || {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::PULL).unwrap();
        socket.bind(ZMQ_ADDR).unwrap();
        socket.set_rcvtimeo(1000).unwrap();

        println!("[Bridge] ZMQ Control Listener online at {}", ZMQ_ADDR);

        while z_running.load(Ordering::SeqCst) {
            if let Ok(Ok(msg)) = socket.recv_string(0) {
                // Format: "PID:WEIGHT:CONFIDENCE:DRIFT:REASONING"
                let parts: Vec<&str> = msg.splitn(5, ':').collect();
                if parts.len() >= 5 {
                    let pid: u32 = parts[0].parse().unwrap_or(0);
                    let weight: f32 = parts[1].parse().unwrap_or(0.0);
                    let conf: f32 = parts[2].parse().unwrap_or(0.0);
                    let drift: f32 = parts[3].parse().unwrap_or(0.0);
                    let reason = parts[4];

                    {
                        let mut s = z_state.lock().unwrap();
                        s.current_action = weight;
                        s.model_confidence = conf;
                        s.world_model_drift = drift;
                        s.active_pid = pid;
                        
                        let mut r_bytes = [0u8; 128];
                        let r_src = reason.as_bytes();
                        let len = r_src.len().min(128);
                        r_bytes[..len].copy_from_slice(&r_src[..len]);
                        s.reasoning = r_bytes;
                        
                        // Immediate Sync to SHM so TUI reflects the change instantly
                        if let Ok(mut mmap) = z_mmap.lock() {
                            mmap.copy_from_slice(bytemuck::bytes_of(&*s));
                        }
                        
                        println!("[Bridge] Received AI Action: PID={} Weight={:.4}", pid, weight);
                    }

                    // Update BPF Actuator Map (NO LATENCY GATE HERE - allow proactive nudges)
                    if pid > 0 {
                        let mut pa = pa_clone.lock().unwrap();
                        let _ = pa.insert(pid, weight as i64, 0);
                    }
                }
            }
        }
    });

    // 7. Attach eBPF Programs
    attach_tracepoint(&mut bpf, "handle_sched_wakeup", "sched", "sched_wakeup")?;
    let switch_prog: &mut aya::programs::RawTracePoint = bpf.program_mut("handle_sched_switch").unwrap().try_into()?;
    switch_prog.load()?;
    switch_prog.attach("sched_switch")?;

    let mut events_ring = RingBuf::try_from(bpf.take_map("events").unwrap())?;
    let mut cpu_tracker = CpuTracker::new();
    let mut last_infra_update = Instant::now();
    let mut last_export = Instant::now();

    println!("[Bridge] System online. Extracting 24D Vectors...");

    while running.load(Ordering::SeqCst) {
        while let Some(item) = events_ring.next() {
            // Priority check: exit inner loop if shutdown signaled
            if !running.load(Ordering::SeqCst) { break; }

            if let Ok(event) = bytemuck::try_from_bytes::<KernelXEvent>(&item) {
                // Telemetry recording still focuses on "high-pain" events
                if event.features[23] > 1000 {
                    persistence::persist_event(event);
                    
                    let (action, current_wait, active_pid) = {
                        let mut s = shared_state.lock().unwrap();
                        s.features = event.features;
                        s.p99_wait_us = event.features[23];
                        s.active_pid = event.pid;
                        
                        let act = if s.active_pid == event.pid { s.current_action } else { 0.0 };
                        (act, s.p99_wait_us, s.active_pid)
                    };
                    
                    // Copy to SHM
                    if let Ok(mut mmap) = mmap_mutex.lock() {
                        let s = shared_state.lock().unwrap();
                        mmap.copy_from_slice(bytemuck::bytes_of(&*s));
                    }

                    if let Some(ref mut m) = manager {
                        let _ = m.record_transition(*event, action);
                    }

                    println!("24D | PID: {:<6} | Wait: {:>5}us | VRuntime: {:>10}", active_pid, current_wait, event.features[5]);
                }
            }
        }

        // Periodic Infra Update (100ms)
        if last_infra_update.elapsed() >= Duration::from_millis(100) {
            let mut s = shared_state.lock().unwrap();
            s.core_heat = cpu_tracker.update();
            s.radish_wal_size = persistence::get_aof_size();
            s.radish_dirty_pages = (s.radish_wal_size / 4096) as u32 % 100;
            
            if let Ok(mut mmap) = mmap_mutex.lock() {
                mmap.copy_from_slice(bytemuck::bytes_of(&*s));
            }
            last_infra_update = Instant::now();
        }

        thread::sleep(Duration::from_millis(10));
    }

    println!("[Bridge] Exiting main loop...");
    
    if let Some(mut m) = manager {
        println!("[Bridge] Flushing trajectory buffer...");
        if let Err(e) = m.flush() {
            eprintln!("[Error] Flush failed: {e}");
        } else {
            println!("[Bridge] Trajectories saved to trajectories.json");
        }
    }
    
    println!("[Bridge] Shutdown complete. Goodbye.");
    Ok(())
}

fn attach_tracepoint(bpf: &mut Ebpf, name: &str, category: &str, tracepoint: &str) -> Result<()> {
    let program: &mut TracePoint = bpf.program_mut(name).unwrap().try_into()?;
    program.load()?;
    program.attach(category, tracepoint)?;
    Ok(())
}
