# Compiler
NVCC = nvcc

# --- Paths ---
# Default CUDA Toolkit Path for core libraries like cublas
CUDA_PATH ?= /usr/local/cuda
# Custom cuDNN Path specified by user
CUDNN_PATH = /river/hpc101/2025/lab3/cudnn

# Add both include paths. The compiler will search them in order.
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDNN_PATH)/include

# Add both library paths. The linker will search them in order.
LIB_PATHS = -L$(CUDA_PATH)/lib64 -L$(CUDNN_PATH)/lib

# Compiler flags
CXXFLAGS = -O3 -std=c++17 -arch=sm_70 $(INCLUDES)

# Libraries to link
# FIX: Added -lcublasLt to resolve the missing dependency for cublas
LIBS = -lcublas -lcublasLt -lcudnn $(LIB_PATHS)

# Source files
SRCS = main.cu naive_conv.cu winograd_conv.cu cudnn_conv.cu

# Executable name
TARGET = winograd

# Default target
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

# Clean rule
clean:
	rm -f $(TARGET)

