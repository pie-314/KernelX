/*
 * KernelX Sentinel: The "Metal" Layer
 *
 * This eBPF program measures "Ready Queue Latency" (the time a process spends 
 * waiting to be scheduled after it's ready to run). This is a critical metric
 * for Reinforcement Learning to optimize system responsiveness.
 */

#include "vmlinux.h"
#include "sentinel_event.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char LICENSE[] SEC("license") = "GPL";

/* 
 * Transient storage for wakeup timestamps. 
 * Key: PID, Value: Nanoseconds since boot.
 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, u32);
    __type(value, u64);
} start_times SEC(".maps");

/* 
 * High-performance Ring Buffer for zero-copy data transfer to the Rust bridge.
 */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24); // 16MB buffer
} events SEC(".maps");

/**
 * Record the exact moment a process enters the "Ready" state.
 * This happens when a process wakes up from sleep or I/O.
 */
SEC("tp/sched/sched_wakeup")
int handle_sched_wakeup(struct trace_event_raw_sched_wakeup_template *ctx) {
    u32 pid = ctx->pid;
    u64 ts = bpf_ktime_get_ns();
    
    // We overwrite existing entries (BPF_ANY) because the most recent 
    // wakeup is the one that matters for current latency.
    bpf_map_update_elem(&start_times, &pid, &ts, BPF_ANY);
    return 0;
}

/**
 * Record the start time for brand new processes.
 */
SEC("tp/sched/sched_wakeup_new")
int handle_sched_wakeup_new(struct trace_event_raw_sched_wakeup_template *ctx) {
    u32 pid = ctx->pid;
    u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&start_times, &pid, &ts, BPF_ANY);
    return 0;
}

/**
 * The "Actuator" logic: This is triggered every time the CPU switches tasks.
 * We calculate: Latency = Current Time (Scheduled) - Wakeup Time (Ready).
 */
SEC("tp/sched/sched_switch")
int handle_sched_switch(struct trace_event_raw_sched_switch *ctx) {
    u32 next_pid = ctx->next_pid;
    u64 now = bpf_ktime_get_ns();
    u64 *start_ts;

    // Check if the process being scheduled was tracked by our wakeup probes
    start_ts = bpf_map_lookup_elem(&start_times, &next_pid);

    if (start_ts) {
        struct latency_event *event;
        u64 delta_us = (now - *start_ts) / 1000;

        // Reserve space in the ring buffer for the telemetry packet
        event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
        if (event) {
            event->scheduled_ns = now;
            event->ready_ns = *start_ts;
            event->latency_us = delta_us;
            event->pid = next_pid;
            event->cpu = bpf_get_smp_processor_id();
            
            // Submit to user-space (Rust bridge)
            bpf_ringbuf_submit(event, 0);
        }

        // Clean up the hash map to prevent memory leaks in kernel-space
        bpf_map_delete_elem(&start_times, &next_pid);
    }

    return 0;
}
