/* Shared wire format for scheduler latency events exported over ringbuf.
 * Layout is fixed at 32 bytes and mirrored in bridge/src/main.rs.
 */
struct latency_event {
  u64 scheduled_ns;
  u64 ready_ns;
  u64 latency_us;
  u32 pid;
  u32 cpu;
};
