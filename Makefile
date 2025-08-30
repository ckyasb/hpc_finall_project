# Makefile for Winograd Project with CUTLASS

# --- 编译器和目标 ---
NVCC = nvcc
TARGET = winograd
CUTLASS_DIR = ./cutlass
SOURCES = main.cu naive_conv.cu winograd_conv.cu

NVCCFLAGS = -O3 -std=c++17 -arch=sm_70 -I$(CUTLASS_DIR)/include -lnccl

all: $(TARGET)

$(TARGET): $(SOURCES)
        @echo "===> Compiling and Linking all sources..."
        $(NVCC) $(NVCCFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)
        @echo "===> Build finished successfully: $(TARGET)"

clean:
        @echo "===> Cleaning build files..."
        rm -f $(TARGET)
        @echo "===> Clean complete."

.PHONY: all clean