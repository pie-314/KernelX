# Variables
CLANG = clang
BPFTOOL = bpftool
SRC = sentinel.bpf.c
OBJ = sentinel.bpf.o
BPF_FS = /sys/fs/bpf/kernelx_sentinel

# Flags
# -O2 is required for BPF
# -target bpf tells clang to output BPF bytecode
# -g includes debug info for BTF
CFLAGS = -O2 -target bpf -g -c

.PHONY: all clean load unload trace vmlinux

# Default target: build and load, but don't start tracing
all: load

# Generate vmlinux.h from your current kernel
vmlinux:
	@echo "Generating vmlinux.h..."
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

# Compile the BPF program
$(OBJ): $(SRC) vmlinux.h sentinel_event.h
	@echo "Compiling $(SRC)..."
	$(CLANG) $(CFLAGS) $(SRC) -o $(OBJ)

# Load and auto-attach all programs from the object.
# `prog load` only loads a single section, which misses the sched_switch
# program that emits the trace output.
load: $(OBJ)
	@echo "Loading and attaching KernelX Sentinel..."
	sudo $(BPFTOOL) prog loadall $(OBJ) $(BPF_FS) autoattach
	@echo "Program loaded. Use 'make trace' to see logs."

# Unload/Unpin the program
unload:
	@echo "Unloading KernelX Sentinel..."
	@sudo rm -rf $(BPF_FS)
	@echo "Done."

# Stream the kernel trace pipe
trace:
	@echo "Listening to KernelX heartbeat (Ctrl+C to stop)..."
	sudo $(BPFTOOL) prog tracelog

# Clean build artifacts
clean:
	@echo "Cleaning up..."
	rm -f $(OBJ) vmlinux.h
