/**
 * KernelX Bridge: The "Nervous System"
 * 
 * This Rust application acts as the high-speed bridge between the Linux Kernel (eBPF)
 * and the Reinforcement Learning Intelligence (Python). It polls the kernel ring
 * buffer and exposes telemetry data.
 */

use std::{
    fs, 
    path::PathBuf, 
    sync::atomic::{AtomicBool, Ordering}, 
    sync::Arc, 
    thread, 
    time::Duration
};

use anyhow::{Context, Result};
use aya::{
    maps::RingBuf,
    programs::TracePoint,
    Ebpf,
};
use aya_log::EbpfLogger;
use bytemuck::{Pod, Zeroable};

const DEFAULT_BPF_OBJECT: &str = "../kernel/sentinel.bpf.o";

/// Memory-mapped structure matching the kernel's `latency_event`.
/// We use `Pod` and `Zeroable` for safe zero-copy casting from raw bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LatencyEvent {
    scheduled_ns: u64,
    ready_ns: u64,
    latency_us: u64,
    pid: u32,
    cpu: u32,
}

fn main() -> Result<()> {
    // 1. Setup graceful shutdown handler (SIGINT/Ctrl+C)
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        println!("\n[Bridge] Shutdown signal received. Cleaning up...");
    }).expect("Error setting Ctrl-C handler");

    // 2. Resolve BPF object path
    let object_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_BPF_OBJECT));

    // 3. Load the BPF bytecode into the kernel
    println!("[Bridge] Loading eBPF object: {}", object_path.display());
    let object = fs::read(&object_path)
        .with_context(|| format!("Failed to read BPF object at {}", object_path.display()))?;
    
    let mut bpf = Ebpf::load(&object).context("Failed to parse BPF object")?;

    // 4. Initialize kernel logging (bpf_printk support)
    if let Err(err) = EbpfLogger::init(&mut bpf) {
        eprintln!("[Warn] aya-log unavailable: {err}");
    }

    // 5. Attach probes to the Linux Scheduler tracepoints
    attach_tracepoint(&mut bpf, "handle_sched_wakeup", "sched", "sched_wakeup")?;
    attach_tracepoint(&mut bpf, "handle_sched_wakeup_new", "sched", "sched_wakeup_new")?;
    attach_tracepoint(&mut bpf, "handle_sched_switch", "sched", "sched_switch")?;

    // 6. Bind to the kernel-to-user ring buffer
    let mut events_ring = RingBuf::try_from(
        bpf.take_map("events")
            .context("BPF ring buffer map 'events' not found")?,
    )
    .context("Failed to initialize ring buffer")?;

    println!("[Bridge] System online. Monitoring micro-latency...");

    // 7. Main Telemetry Loop
    while running.load(Ordering::SeqCst) {
        // Poll for new events from the kernel
        while let Some(item) = events_ring.next() {
            match parse_event(&item) {
                Ok(event) => {
                    // TODO: In Phase 2, this will send data to RadishDB and Python IPC
                    println!(
                        "EVENT | pid: {:<6} | cpu: {:<2} | wait: {:>5} us",
                        event.pid, event.cpu, event.latency_us
                    );
                }
                Err(e) => eprintln!("[Error] Corrupt event: {e}"),
            }
        }

        // Slight sleep to prevent 100% CPU usage on the bridge itself
        thread::sleep(Duration::from_millis(10));
    }

    println!("[Bridge] Successfully detached. Goodbye.");
    Ok(())
}

/// Helper to attach a BPF program to a kernel tracepoint.
fn attach_tracepoint(bpf: &mut Ebpf, name: &str, category: &str, tracepoint: &str) -> Result<()> {
    let program: &mut TracePoint = bpf
        .program_mut(name)
        .with_context(|| format!("Program '{name}' not found in BPF object"))?
        .try_into()
        .context("Program is not a tracepoint")?;

    program.load().context(format!("Failed to load {name}"))?;
    program.attach(category, tracepoint)
        .context(format!("Failed to attach {name} to {category}:{tracepoint}"))?;

    Ok(())
}

/// Safely casts a byte slice into our structured LatencyEvent.
fn parse_event(bytes: &[u8]) -> Result<LatencyEvent> {
    bytemuck::try_from_bytes::<LatencyEvent>(bytes)
        .copied()
        .map_err(|_| anyhow::anyhow!("Data layout mismatch in ring buffer packet"))
}
