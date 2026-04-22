#ifndef __SENTINEL_EVENT_H
#define __SENTINEL_EVENT_H

/* 
 * KernelX Unified Telemetry Packet
 * 
 * This structure is the "Handshake" between the Kernel and the Bridge.
 * Total size: 24*8 + 8 + 4 + 4 = 208 bytes.
 */
struct kernelx_event {
    /* The 24-dimensional state vector */
    unsigned long long features[24];
    
    /* Metadata for alignment and tracking */
    unsigned long long timestamp;
    unsigned int pid;
    unsigned int cpu;
};

#endif
