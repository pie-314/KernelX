# KernelX Sentinel Makefile
# Optimized for the "Metal Layer" testing

CLANG = clang
BPFTOOL = bpftool
SRC = sentinel.bpf.c
OBJ = sentinel.bpf.o
BPF_FS = /sys/fs/bpf/kernelx_sentinel

# Colors for better CLI UX
CYAN  = \033[0;36m
GREEN = \033[0;32m
RED   = \033[0;31m
NC    = \033[0m # No Color

# -O2 is required for BPF
# -target bpf tells clang to output BPF bytecode
# -g includes debug info for BTF (CO-RE support)
CFLAGS = -O2 -target bpf -g -c

.PHONY: all clean load unload reload trace vmlinux check_env

all: load

# Verification of environment
check_env:
	@if [ ! -f /sys/kernel/btf/vmlinux ]; then \
		echo "$(RED)Error: BTF not found at /sys/kernel/btf/vmlinux. Is your kernel compiled with CONFIG_DEBUG_INFO_BTF?$(NC)"; \
		exit 1; \
	fi

# Generate vmlinux.h from your current kernel
vmlinux: check_env
	@echo "$(CYAN)Generating vmlinux.h from active kernel...$(NC)"
	@$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

# Compile the BPF program
$(OBJ): $(SRC) sentinel_event.h vmlinux.h
	@echo "$(CYAN)Compiling $(SRC) -> $(OBJ)...$(NC)"
	@$(CLANG) $(CFLAGS) $(SRC) -o $(OBJ)
	@echo "$(GREEN)Compilation successful.$(NC)"

# Load and auto-attach all programs
load: $(OBJ)
	@echo "$(CYAN)Loading and attaching KernelX Sentinel to Scheduler...$(NC)"
	@# Ensure no stale pins exist
	@sudo rm -rf $(BPF_FS)
	@sudo mkdir -p $(BPF_FS)
	@sudo $(BPFTOOL) prog loadall $(OBJ) $(BPF_FS) autoattach
	@echo "$(GREEN)Sentinel online. Pins located at $(BPF_FS)$(NC)"

# Unload/Unpin the program
unload:
	@echo "$(RED)Unloading KernelX Sentinel...$(NC)"
	@sudo rm -rf $(BPF_FS)
	@echo "$(GREEN)Done.$(NC)"

# Combined reload
reload: unload load

# Stream the kernel trace pipe (for bpf_printk debugging)
trace:
	@echo "$(CYAN)Listening to Kernel Heartbeat (Ctrl+C to stop)...$(NC)"
	@sudo $(BPFTOOL) prog tracelog

# Clean build artifacts
clean:
	@echo "$(RED)Cleaning up build artifacts...$(NC)"
	@rm -f $(OBJ) vmlinux.h
