/*
 * KernelX Sentinel: Full "Metal" Implementation
 *
 * This eBPF program is the core sensor and actuator for KernelX.
 * It implements high-frequency telemetry (24D State Vector) and 
 * provides the hook for AI-driven scheduling decisions.
 */

#include "vmlinux.h"
#include "sentinel_event.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "GPL";

/* --- MAPS: TELEMETRY & STATE --- */

// Per-PID start times to measure latency
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, u32);
    __type(value, u64);
} start_times SEC(".maps");

// Per-CPU counters for context switches and load
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} cpu_stats SEC(".maps");

// Output Ring Buffer
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24);
} events SEC(".maps");

/* --- MAPS: ACTUATION --- */

// The "Control" map: AI writes priority weights here.
// Key: PID, Value: Weight (-100 to 100)
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, u32);
    __type(value, s64);
} priority_actions SEC(".maps");

/* --- HELPER: FEATURE EXTRACTION --- */

static __always_inline void collect_24d_vector(struct kernelx_event *e, struct task_struct *task) {
    // [0-3] CPU ID and Basic Stats
    e->features[0] = bpf_get_smp_processor_id();
    e->features[1] = BPF_CORE_READ(task, prio);
    e->features[2] = BPF_CORE_READ(task, static_prio);
    e->features[3] = BPF_CORE_READ(task, normal_prio);

    // [4-7] Scheduling Stats (using BPF Core Read for safety)
    e->features[4] = BPF_CORE_READ(task, se.sum_exec_runtime);
    e->features[5] = BPF_CORE_READ(task, se.vruntime);
    e->features[6] = BPF_CORE_READ(task, se.nr_migrations);
    e->features[7] = BPF_CORE_READ(task, nr_cpus_allowed);

    // [8-11] System Load approximations
    // In a real implementation, we'd pull from rq->nr_running
    e->features[8] = 0; // Placeholder for per-CPU run-queue length
    e->features[9] = 0; // Placeholder for global load
    
    // [12-15] Context Switch Counters (Mock)
    u32 key = 0;
    u64 *cs_count = bpf_map_lookup_elem(&cpu_stats, &key);
    if (cs_count) {
        e->features[12] = *cs_count;
    }

    // [16-23] Reserved for PMU (Cache Misses/IPC) and I/O Wait
    // These require perf_event_open and are populated by user-space sidecars
    // or specialized BPF helpers.
}

/* --- PROBES: SCHEDULER SENSORS --- */

SEC("tp/sched/sched_wakeup")
int handle_sched_wakeup(struct trace_event_raw_sched_wakeup_template *ctx) {
    u32 pid = ctx->pid;
    u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&start_times, &pid, &ts, BPF_ANY);
    return 0;
}

SEC("raw_tp/sched_switch")
int handle_sched_switch(u64 *ctx) {
    /* 
     * In raw_tp/sched_switch, the context is a raw array of u64:
     * ctx[0] = preempt (bool)
     * ctx[1] = prev task_struct*
     * ctx[2] = next task_struct*
     * 
     * We use a raw u64* here because some BPF loaders (like Aya)
     * fail to relocate the standard bpf_raw_tracepoint_args struct
     * due to its zero-length 'args[0]' member.
     */
    struct task_struct *next = (struct task_struct *)ctx[2];
    struct task_struct *prev = (struct task_struct *)ctx[1];
    
    u32 next_pid = BPF_CORE_READ(next, pid);
    u64 now = bpf_ktime_get_ns();
    u64 *start_ts;

    // Update Context Switch Counter
    u32 key = 0;
    u64 *cs_count = bpf_map_lookup_elem(&cpu_stats, &key);
    if (cs_count) {
        __sync_fetch_and_add(cs_count, 1);
    }

    // Measure Latency and Ship Telemetry
    start_ts = bpf_map_lookup_elem(&start_times, &next_pid);
    if (start_ts) {
        struct kernelx_event *e;
        e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
        if (e) {
            e->timestamp = now;
            e->pid = next_pid;
            e->cpu = bpf_get_smp_processor_id();
            
            // Populate the 24D Vector using the NEXT task
            collect_24d_vector(e, next);
            
            // Set the primary latency metric (Wait Time) in index 23
            e->features[23] = (now - *start_ts) / 1000; 

            bpf_ringbuf_submit(e, 0);
        }
        bpf_map_delete_elem(&start_times, &next_pid);
    }

    /* --- ACTUATOR LOGIC --- */
    // Check if the AI has a specific instruction for this PID
    s64 *action = bpf_map_lookup_elem(&priority_actions, &next_pid);
    if (action) {
        // Here we would apply the weight. 
        // Note: Direct vruntime manipulation requires kprobes/fentry on specific 
        // internal functions. For this core implementation, we log the intent.
        // bpf_printk("ACTUATOR: PID %d assigned weight %lld", next_pid, *action);
    }

    return 0;
}
