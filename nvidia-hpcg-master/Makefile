# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# by default, "arch" is unknown, should be specified in the command line
# Example usage: make arch=CUDA_X86
arch = UNKNOWN

setup_file = setup/Make.$(arch)
include $(setup_file)

# --- FIX for linker error "cannot find -lmpi" ---
# The setup file might create a double -L (e.g., -L-L/path). 
# This line corrects it to a single -L.
HPCG_LIBS := $(subst -L-L,-L,$(HPCG_LIBS))
# --- END FIX ---

# --- Add NCCL flags if USE_NCCL=1 is set ---
# This ensures multi-GPU support is compiled in.
# It requires the NCCL_PATH environment variable to be set.
ifeq ($(USE_NCCL), 1)
	CXXFLAGS += -DUSE_NCCL -I$(NCCL_PATH)/include
	HPCG_LIBS += -L$(NCCL_PATH)/lib -lnccl
endif
# --- END NCCL FIX ---

bin_name = bin/xhpcg

# --- NEW, AUTOMATED FILE DETECTION ---
# Use wildcard to automatically find all .cpp and .cu source files in the src/ directory.
CPP_SOURCES = $(wildcard src/*.cpp)
CUDA_SOURCES = $(wildcard src/*.cu)

# Automatically generate the list of object files (.o) from the source file lists.
OBJS = $(CPP_SOURCES:.cpp=.o) $(CUDA_SOURCES:.cu=.o)
# --- END NEW DETECTION ---


# The final executable depends on ALL object files.
$(bin_name): $(OBJS)
	$(LINKER) $(LINKFLAGS) $(OBJS) -o $(bin_name) $(HPCG_LIBS)


# --- SIMPLIFIED COMPILATION RULES ---
# Generic rule to compile any .cpp file in src/ into a .o file in src/
src/%.o: src/%.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

# Generic rule to compile any .cu file in src/ into a .o file in src/
src/%.o: src/%.cu
	$(CXX) -c $< -o $@ $(CXXFLAGS)
# --- END SIMPLIFIED RULES ---


install:
	cp build/bin/xhpcg* bin/

clean:
	# Clean rule now uses the automatically generated object list.
	rm -f $(OBJS) $(bin_name)

.PHONY: clean


