CUDA_PATH       := /usr/local/cuda-5.0
CUDA_INC_PATH   := $(CUDA_PATH)/include
CUDA_BIN_PATH   := $(CUDA_PATH)/bin
CUDA_LIB_PATH   := $(CUDA_PATH)/lib64
NVCC            := $(CUDA_BIN_PATH)/nvcc
GCC             := g++

CPPFLAGS  :=
ifdef THREADS
CPPFLAGS += -DTHREADS=$(THREADS)
endif
ifdef BLOCKS
CPPFLAGS += -DBLOCKS=$(BLOCKS)
endif
ifdef SIZE
CPPFLAGS += -DSIZE=$(SIZE)
endif
ifdef EXP
CPPFLAGS += -DEXP=$(EXP)
endif
ifdef LIMB_COUNT
CPPFLAGS += -DLIMB_COUNT=$(LIMB_COUNT)
endif
ifdef PTX
CPPFLAGS += -Xptxas=-v
endif

LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
CCFLAGS   := -m64 -arch=sm_30 $(CPPFLAGS)
NVCCFLAGS := -m64 -arch=sm_30 $(CPPFLAGS)
INCLUDES  := -I$(CUDA_INC_PATH) -I. -I$(CUDA_PATH)/samples/common/inc

all: BigNumAdd prod Exp

BigNumAdd: BigNumAdd.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) -O3 -o $@ $<

prod: prod.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) -O3 -o $@ $<

Exp: Exp.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) -O3 -o $@ $<
