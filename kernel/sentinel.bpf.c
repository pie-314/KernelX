#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char LICENSE[] SEC("license") = "GPL";

/**
  KernelX Sentinel: Scheduler Observer
  Hook: sched_switch tracepoint
 */
SEC("tp/sched/sched_switch")
int handle_sched_switch(struct trace_event_raw_sched_switch *ctx) {
  // Extract the Previous and Next PIDs
  int prev_pid = ctx->prev_pid;
  int next_pid = ctx->next_pid;

  // Extract the name of the 'Next' task
  char next_comm[16];
  bpf_get_current_comm(&next_comm, sizeof(next_comm));

  bpf_printk("KernelX Switch: [%s] (%d) is taking over CPU.", next_comm,
             next_pid);

  return 0;
}
