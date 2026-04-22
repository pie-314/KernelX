/**
 * KernelX Bridge: Updated for 24D Telemetry
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

/// Mirror of the 24D packet from kernel/sentinel_event.h
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct KernelXEvent {
    features: [u64; 24],
    timestamp: u64,
    pid: u32,
    cpu: u32,
}

fn main() -> Result<()> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        println!("\n[Bridge] Shutdown signal received.");
    }).expect("Error setting Ctrl-C handler");

    let object_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_BPF_OBJECT));

    let object = fs::read(&object_path)
        .with_context(|| format!("Failed to read BPF object at {}", object_path.display()))?;
    
    let mut bpf = Ebpf::load(&object).context("Failed to parse BPF object")?;

    if let Err(err) = EbpfLogger::init(&mut bpf) {
        eprintln!("[Warn] aya-log unavailable: {err}");
    }

    attach_tracepoint(&mut bpf, "handle_sched_wakeup", "sched", "sched_wakeup")?;
    
    // Attach sched_switch as a RawTracePoint
    let switch_prog: &mut aya::programs::RawTracePoint = bpf
        .program_mut("handle_sched_switch")
        .context("Missing handle_sched_switch")?
        .try_into()
        .context("Not a raw tracepoint")?;
    switch_prog.load()?;
    switch_prog.attach("sched_switch")?;

    let mut events_ring = RingBuf::try_from(
        bpf.take_map("events")
            .context("BPF ring buffer map 'events' not found")?,
    )
    .context("Failed to initialize ring buffer")?;

    println!("[Bridge] System online. Extracting 24D Vectors...");

    while running.load(Ordering::SeqCst) {
        while let Some(item) = events_ring.next() {
            match bytemuck::try_from_bytes::<KernelXEvent>(&item) {
                Ok(event) => {
                    // Log a summary of the 24D vector
                    println!(
                        "24D | PID: {:<6} | Wait: {:>5}us | VRuntime: {:>10}",
                        event.pid, 
                        event.features[23], // Our latency metric
                        event.features[5]   // VRuntime index
                    );
                }
                Err(e) => eprintln!("[Error] Data layout mismatch: {e}"),
            }
        }
        thread::sleep(Duration::from_millis(10));
    }

    Ok(())
}

fn attach_tracepoint(bpf: &mut Ebpf, name: &str, category: &str, tracepoint: &str) -> Result<()> {
    let program: &mut TracePoint = bpf
        .program_mut(name)
        .with_context(|| format!("Program '{name}' not found"))?
        .try_into()
        .context("Program is not a tracepoint")?;

    program.load().context(format!("Failed to load {name}"))?;
    program.attach(category, tracepoint)
        .context(format!("Failed to attach {name} to {category}:{tracepoint}"))?;

    Ok(())
}
