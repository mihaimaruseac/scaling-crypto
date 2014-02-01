#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef THREADS
#define THREADS 256
#endif
#ifndef BLOCKS
#define BLOCKS 64
#endif
#define TOTAL_THREADS (THREADS * BLOCKS)
#ifndef SIZE
#define SIZE 32
#endif
#ifndef BITS
#define BITS 768
#endif

#if BITS == 768
#define BASES 25
int bases[BASES] = {1073741827, 1073741831, 1073741833, 1073741839,
	1073741843, 1073741857, 1073741891, 1073741909, 1073741939,
	1073741953, 1073741969, 1073741971, 1073741987, 1073741993,
	1073742037, 1073742053, 1073742073, 1073742077, 1073742091,
	1073742113, 1073742169, 1073742203, 1073742209, 1073742223,
	1073742233, 1073743327};
#else
#error "Only 768 implemented"
#endif

#define INPUT_SIZE (BASES * SIZE * TOTAL_THREADS)
#define OUTPUT_SIZE (BASES * TOTAL_THREADS)
#define INPUT_STRIDE (SIZE * TOTAL_THREADS)
#define OUTPUT_STRIDE (TOTAL_THREADS)

__global__ void task(int *input, int *output, int items)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int ix;

	for (ix = tid; ix < OUTPUT_SIZE; ix += blockDim.x * gridDim.x)
		output[ix] = 1;

	for (ix = tid; ix < items; ix += blockDim.x * gridDim.x)
		output[tid + (ix - tid) / INPUT_STRIDE * OUTPUT_SIZE] *=
			input[ix] % bases[ix / INPUT_STRIDE];
}

int main(int argc, char *argv[])
{
	dim3 numThreads(THREADS, 1), numBlocks(BLOCKS, 1);
	int i, j, device = 0;
	cudaEvent_t start, stop;
	float msecTotal;

	int *gpuInput, *gpuOutput;
	int *cpuInput, *cpuOutput;

	cudaError_t err;

	/* init events and calloc vectors (calloc to ensure padding with 0) */
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cpuInput = (int *)calloc(INPUT_SIZE, 1);
	cpuOutput = (int *)calloc(OUTPUT_SIZE, 1);

	/* randomly fill up input vectors: use constant seed */
	srand48(42);
	for (i = 0; i < INPUT_SIZE; i++)
		input[i] = lrand48() % bases[j / INPUT_STRIDE];

	/* set up vectors on device */
	cudaSetDevice(device);
	cudaMalloc((void**)&gpuInput, size);
	cudaMalloc((void**)&gpuOutput, size);

	/* copy vectors to device */
	err = cudaMemcpy(gpuInput, cpuInput, INPUT_SIZE, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		printf("Copy input to device %s\n", cudaGetErrorString(err));

	/* GPU computation */
	cudaEventRecord(start, NULL);
	task<<<numBlocks, numThreads>>>(gpuInput, gpuOutput, INPUT_SIZE);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("cannot invoke task kernel! %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	printf("gpu time: %.3f ms\n", msecTotal);

	return 0;
}

