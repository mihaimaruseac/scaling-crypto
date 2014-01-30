/**
 * @author Mihai Maruseasc (Mihai.Maruseac001@umb.edu)
 *
 * @section DESCRIPTION
 * Bignum addition on each thread on CUDA.
 */
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
#define TOTAL_SIZE (SIZE * TOTAL_THREADS)

/* In multiples of 64 bits (minimum 2 for 128 bits numbers) */
#define NUMBER_SIZE 12

#define LIMB_COUNT 40
// TODO: check with 1 + NUMBER_SIZE
#define half_shift 32
#define full_shift 64
#define half_mask 0x00000000ffffffffll
#define full_mask 0xffffffffffffffffll
#define first_bit 0x8000000000000000ll

typedef size_t limb_t;
struct bn {
	size_t limbs_used;
	limb_t limbs[LIMB_COUNT];
};

__constant__ bn N;
__constant__ bn ZERO = {0};

__noinline__ __device__ void bn_add(bn *a, bn *b, bn *c)
{
	int i;
	limb_t tmp, carry = 0;

	/* implement an half adder on limbs */
#pragma unroll
	for (i = 0; i < LIMB_COUNT; i++) {
		carry += (a->limbs[i] & half_mask) + (b->limbs[i] & half_mask);
		tmp = carry & half_mask;
		carry >>= half_shift;
		carry += ((a->limbs[i] >> half_shift) & half_mask) + ((b->limbs[i] >> half_shift) & half_mask);
		tmp |= (carry & half_mask) << half_shift;
		carry >>= half_shift;
		c->limbs[i] = tmp;
	}

	/* get the maximum of operand's limbs without branching */
	c->limbs_used = __sad(a->limbs_used, b->limbs_used,
			a->limbs_used + b->limbs_used) >> 1;

	/* if no carry to the last limb reduce number of used limbs */
	c->limbs_used += !!c->limbs[c->limbs_used];
}

__noinline__ __device__ void bn_sub(bn *a, bn *b, bn *c)
{
	int i;
	limb_t tmp, borrow = 1;

#pragma unroll
	for (i = 0; i < LIMB_COUNT; i++) {
		borrow = (a->limbs[i] & half_mask) + half_mask - (b->limbs[i] & half_mask) + borrow;
		tmp = borrow & half_mask;
		borrow >>= half_shift;
		borrow = ((a->limbs[i] >> half_shift) & half_mask) + half_mask - ((b->limbs[i] >> half_shift) & half_mask) + borrow;
		tmp |= (borrow & half_mask) << half_shift;
		borrow >>= half_shift;
		c->limbs[i] = tmp;
	}

	/* get the maximum of operand's limbs without branching */
	c->limbs_used = __sad(a->limbs_used, b->limbs_used,
			a->limbs_used + b->limbs_used) >> 1;

	/* if b < a use one extra limb (though in theory all of them are 0xff..ff) */
	c->limbs_used += !!c->limbs[c->limbs_used];
}

__noinline__ __device__ int bn_cmp(bn *a, bn *b)
{
	int ret = 0, i;

	/* compare limb by limb (but touch all limbs) */
#pragma unroll
	for (i = LIMB_COUNT - 1; i >= 0; i--) {
		ret += (ret == 0) & (a->limbs[i] > b->limbs[i]);
		ret -= (ret == 0) & (a->limbs[i] < b->limbs[i]);
	}

	return ret;
}

/**
 * Assumes a is at most 2b.
 */
__noinline__ __device__ void bn_mod(bn *a, bn *b)
{
	int cmp = bn_cmp(a, b) > 0;
	bn *sub;

	/* need to do the same amount of operations so subtract 0 when below b */
	sub = cmp ? b : &ZERO;
	bn_sub(a, sub, a);
}

__noinline__ __device__ limb_t bn_llr(limb_t n0)
{
	limb_t s = 0ll, t = 1ll;
	limb_t alpha = -n0, beta = first_bit;
	int i;

#pragma unroll
	for (i = 0; i < full_shift; i++)
		if (((s & 1) == 0) && ((t & 1) == 0)) {
			s >>= 1;
			t >>= 1;
		} else {
			s = (s >> 1) + beta;
			t = (t - alpha) >> 1;
		}

	return s;
}

/**
 * Limb multiplication:
 *
 * m = a * b & full_mask;
 * q = a * b >>
 */
__noinline__ __device__ void bn_llm(limb_t a, limb_t b, limb_t *m, limb_t *q)
{
	limb_t z0, z1, z2, z3, M, Q;

	z0 = (a & half_mask) * (b & half_mask);
	z1 = ((a >> half_shift) & half_mask) * (b & half_mask);
	z2 = (a & half_mask) * ((b >> half_shift) & half_mask);
	z3 = ((a >> half_shift) & half_mask) * ((b >> half_shift) & half_mask);

	M = z0 & half_mask;
	Q = (z0 >> half_shift) & half_mask;

	Q += (z1 & half_mask) + (z2 & half_mask);
	M |= (Q & half_mask) << half_shift;
	Q >>= half_shift;
	Q += z3 + ((z1 >> half_shift) & half_mask) + ((z2 >> half_shift) & half_mask);

	*m = M;
	*q = Q;
}

/**
 * Multiplying with a single digit
 */
__noinline__ __device__ void bn_lm(bn *a, limb_t x, bn *b)
{
	limb_t m, q, carry = 0, tmp;
	int i;

#pragma unroll
	for (i = 0; i < LIMB_COUNT; i++) {
		bn_llm(a->limbs[i], x, &m, &q);
		carry += m & half_mask;
		tmp = carry & half_mask;
		carry >>= half_shift;
		carry += (m >> half_shift) & half_mask;
		tmp |= (carry & half_mask) << half_shift;
		carry >>= half_shift;
		b->limbs[i] = tmp;
		carry += q;
	}

	b->limbs_used = a->limbs_used;
	b->limbs_used += !!b->limbs[b->limbs_used];
}

/**
 * c = ab*((2^64)^{-||m||}) mod m
 */
__noinline__ __device__ void bn_montgomery(bn *a, bn *b, bn *m, bn *c)
{
	limb_t rho = bn_llr(m->limbs[0]), q, rem, carry;
	int i, j;
	bn A, Y;

	memset(&A, 0, sizeof(bn));
	for (i = 0; i < m->limbs_used; i++) {
		/* a->limbs[i] * b->limbs[0] = 2^64 * q + rem */
		bn_llm(a->limbs[i], b->limbs[0], &rem, &q);

		/* q = (A.limbs[0] + rem) mod 2^64 */
		carry = (A.limbs[0] & half_mask) + (rem & half_mask);
		q = carry & half_mask;
		carry >>= half_shift;
		carry += ((A.limbs[0] >> half_shift) & half_mask) + ((rem >> half_shift) & half_mask);
		q |= (carry & half_mask) << half_shift;

		/* q * rho = 2^64*rem + q */
		bn_llm(q, rho, &q, &rem);

		/* A += a->limbs[i] * b + q*m */
		bn_lm(b, a->limbs[i], &Y);
		bn_add(&A, &Y, &A);
		bn_lm(m, q, &Y);
		bn_add(&A, &Y, &A);

		/* A >>= 64 */
		for (j = 0; j <= m->limbs_used; j++)
			A.limbs[j] = A.limbs[j+1];
		A.limbs_used--;
	}

	*c = A;
}

/**
 * ab mod m = c
 */
__noinline__ __device__ void bn_mul(bn *a, bn *b, bn *m, bn *c)
{
	bn reductor;
	limb_t carry, old_carry;
	int i, j, k;

	bn_montgomery(a, b, m, c);

	/* reductor = (2^64)^(2*m->limbs_used) mod m */
	memset(&reductor, 0, sizeof(bn));
	reductor.limbs_used = m->limbs_used;
	reductor.limbs[1] = 1; /* reductor = 2^64 */
	for (i = 2 * m->limbs_used - 1; i > 0; i--)
		/* reductor = reductor * (2^64) mod m */
		for (j = 0; j < full_shift; j++) {
			old_carry = 0;
			for (k = 0; k <= reductor.limbs_used; k++) {
				carry = !!(reductor.limbs[k] & first_bit);
				reductor.limbs[k] <<= 1;
				reductor.limbs[k] |= old_carry;
				old_carry = carry;
			}
			reductor.limbs_used += !!reductor.limbs[reductor.limbs_used];
			bn_mod(&reductor, m);
		}

	bn_montgomery(c, &reductor, m, c);
}

/**
 * a^b mod N
 */
__noinline__ __device__ void bn_exp(bn *a, bn *b, bn *m, bn *c)
{
	bn r, base;
	limb_t exp;
	int i;

	memset(&r, 0, sizeof(bn));
	r.limbs_used = m->limbs_used;
	r.limbs[0] = 1;
	base = *a;

	for (i = 0; i < b->limbs_used; i++) {
		exp = b->limbs[i];
		while (exp) {
			if (exp & 2)
				bn_mul(&r, &base, m, &r);
			exp >>= 1;
			bn_mul(&base, &base, m, &base);
		}
	}

	*c = r;
}

#if 0
__global__ void bn_add_work(bn *a, bn *b, bn *c, int *d)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;

#if 0
	/* c = a + b - b mod N */
	bn_add(&a[ix], &b[ix], &c[ix]);
	bn_sub(&c[ix], &b[ix], &c[ix]);
	bn_mod(&c[ix], &N);
	d[ix] = bn_cmp(&a[ix], &N);
#endif
#if 1
	/* c = a * b mod N */
	bn_mul(&a[ix], &b[ix], &N, &c[ix]);
	d[ix] = N.limbs_used;
#endif
#if 0
	/* c = a ^ b mod N */
	bn_exp(&a[ix], &b[ix], &N, &c[ix]);
#endif
}
#endif

__global__ void pairwiseMultiply(bn *a, int *b, bn *c, int items)
{
	int ix;
	bn tmp;

	memset(&tmp, 0, sizeof(bn));
	tmp.limbs_used = 1; tmp.limbs[0] = 1;

#pragma unroll
	for (ix = threadIdx.x + blockIdx.x * blockDim.x;
			ix < items;
			ix += blockDim.x * gridDim.x) {
		bn_mul(&tmp, &a[ix], &N, &tmp);
#if 0
		memset(&tmp, 0, sizeof(bn));
		tmp.limbs_used = 1; tmp.limbs[0] = b[ix];
		bn_mul(&a_local[i++], &tmp, &N, &c[ix]);
#endif
#if 0
		tmp.limbs_used = 1; tmp.limbs[0] = a[ix].limbs[0];
		bn_exp(&a[ix], &tmp, &N, &c[ix]);
#endif
#if 0
		bn_lm(&a[ix], b[ix], &c[ix]);
		bn_mod(&c[ix], &N);
#endif
	}

	c[threadIdx.x + blockIdx.x * blockDim.x] = tmp;
}

__global__ void reduce(bn* input, bn* output, int size){
	unsigned int ix, tid;
	bn sum;

	memset(&sum, 0, sizeof(bn));

	tid = threadIdx.x;

#pragma unroll
	for (ix = threadIdx.x + blockIdx.x * blockDim.x;
			ix < size;
			ix += blockDim.x * gridDim.x) {
		bn_add(&sum, &input[ix], &sum);
		bn_mod(&sum, &N);
	}
	output[tid] = sum;
	__syncthreads();

	if (tid == 0) {
		memset(&sum, 0, sizeof(bn));
#pragma unroll
		for (ix = 0; ix < blockDim.x; ix++) {
			bn_add(&sum, &output[ix], &sum);
			bn_mod(&sum, &N);
		}

		output[0] = sum;
	}
}

static void cpu_bn_add(bn *a, bn *b, bn *c)
{
	int i;
	limb_t tmp, carry = 0;

	/* implement an half adder on limbs */
#pragma unroll
	for (i = 0; i < LIMB_COUNT; i++) {
		carry += (a->limbs[i] & half_mask) + (b->limbs[i] & half_mask);
		tmp = carry & half_mask;
		carry >>= half_shift;
		carry += ((a->limbs[i] >> half_shift) & half_mask) + ((b->limbs[i] >> half_shift) & half_mask);
		tmp |= (carry & half_mask) << half_shift;
		carry >>= half_shift;
		c->limbs[i] = tmp;
	}

	/* get the maximum of operand's limbs without branching */
	c->limbs_used = a->limbs_used;
	if (c->limbs_used < b->limbs_used)
		c->limbs_used = b->limbs_used;

	/* if no carry to the last limb reduce number of used limbs */
	c->limbs_used += !!c->limbs[c->limbs_used];
}

static void cpu_bn_sub(bn *a, bn *b, bn *c)
{
	int i;
	limb_t tmp, borrow = 1;

#pragma unroll
	for (i = 0; i < LIMB_COUNT; i++) {
		borrow = (a->limbs[i] & half_mask) + half_mask - (b->limbs[i] & half_mask) + borrow;
		tmp = borrow & half_mask;
		borrow >>= half_shift;
		borrow = ((a->limbs[i] >> half_shift) & half_mask) + half_mask - ((b->limbs[i] >> half_shift) & half_mask) + borrow;
		tmp |= (borrow & half_mask) << half_shift;
		borrow >>= half_shift;
		c->limbs[i] = tmp;
	}

	/* get the maximum of operand's limbs without branching */
	c->limbs_used = a->limbs_used;
	if (c->limbs_used < b->limbs_used)
		c->limbs_used = b->limbs_used;

	/* if b < a use one extra limb (though in theory all of them are 0xff..ff) */
	c->limbs_used += !!c->limbs[c->limbs_used];
}

static int cpu_bn_cmp(bn *a, bn *b)
{
	int ret = 0, i;

	/* compare limb by limb (but touch all limbs) */
#pragma unroll
	for (i = LIMB_COUNT - 1; i >= 0; i--) {
		ret += (ret == 0) & (a->limbs[i] > b->limbs[i]);
		ret -= (ret == 0) & (a->limbs[i] < b->limbs[i]);
	}

	return ret;
}

/**
 * Assumes a is at most 2b.
 */
static void cpu_bn_mod(bn *a, bn *b)
{
	int cmp = cpu_bn_cmp(a, b) > 0;

	/* need to do the same amount of operations so subtract 0 when below b */
	if (cmp)
		cpu_bn_sub(a, b, a);
}

static limb_t cpu_bn_llr(limb_t n0)
{
	limb_t s = 0ll, t = 1ll;
	limb_t alpha = -n0, beta = first_bit;
	int i;

#pragma unroll
	for (i = 0; i < full_shift; i++)
		if (((s & 1) == 0) && ((t & 1) == 0)) {
			s >>= 1;
			t >>= 1;
		} else {
			s = (s >> 1) + beta;
			t = (t - alpha) >> 1;
		}

	return s;
}

/**
 * Limb multiplication:
 *
 * m = a * b & full_mask;
 * q = a * b >>
 */
static void cpu_bn_llm(limb_t a, limb_t b, limb_t *m, limb_t *q)
{
	limb_t z0, z1, z2, z3, M, Q;

	z0 = (a & half_mask) * (b & half_mask);
	z1 = ((a >> half_shift) & half_mask) * (b & half_mask);
	z2 = (a & half_mask) * ((b >> half_shift) & half_mask);
	z3 = ((a >> half_shift) & half_mask) * ((b >> half_shift) & half_mask);

	M = z0 & half_mask;
	Q = (z0 >> half_shift) & half_mask;

	Q += (z1 & half_mask) + (z2 & half_mask);
	M |= (Q & half_mask) << half_shift;
	Q >>= half_shift;
	Q += z3 + ((z1 >> half_shift) & half_mask) + ((z2 >> half_shift) & half_mask);

	*m = M;
	*q = Q;
}

/**
 * Multiplying with a single digit
 */
static void cpu_bn_lm(bn *a, limb_t x, bn *b)
{
	limb_t m, q, carry = 0, tmp;
	int i;

#pragma unroll
	for (i = 0; i < LIMB_COUNT; i++) {
		cpu_bn_llm(a->limbs[i], x, &m, &q);
		carry += m & half_mask;
		tmp = carry & half_mask;
		carry >>= half_shift;
		carry += (m >> half_shift) & half_mask;
		tmp |= (carry & half_mask) << half_shift;
		carry >>= half_shift;
		b->limbs[i] = tmp;
		carry += q;
	}

	b->limbs_used = a->limbs_used;
	b->limbs_used += !!b->limbs[b->limbs_used];
}

/**
 * c = ab*((2^64)^{-||m||}) mod m
 */
static void cpu_bn_montgomery(bn *a, bn *b, bn *m, bn *c)
{
	limb_t rho = cpu_bn_llr(m->limbs[0]), q, rem, carry;
	int i, j;
	bn A, Y;

	memset(&A, 0, sizeof(bn));
	for (i = 0; i < m->limbs_used; i++) {
		/* a->limbs[i] * b->limbs[0] = 2^64 * q + rem */
		cpu_bn_llm(a->limbs[i], b->limbs[0], &rem, &q);

		/* q = (A.limbs[0] + rem) mod 2^64 */
		carry = (A.limbs[0] & half_mask) + (rem & half_mask);
		q = carry & half_mask;
		carry >>= half_shift;
		carry += ((A.limbs[0] >> half_shift) & half_mask) + ((rem >> half_shift) & half_mask);
		q |= (carry & half_mask) << half_shift;

		/* q * rho = 2^64*rem + q */
		cpu_bn_llm(q, rho, &q, &rem);

		/* A += a->limbs[i] * b + q*m */
		cpu_bn_lm(b, a->limbs[i], &Y);
		cpu_bn_add(&A, &Y, &A);
		cpu_bn_lm(m, q, &Y);
		cpu_bn_add(&A, &Y, &A);

		/* A >>= 64 */
		for (j = 0; j <= m->limbs_used; j++)
			A.limbs[j] = A.limbs[j+1];
		A.limbs_used--;
	}

	*c = A;
}

/**
 * ab mod m = c
 */
static void cpu_bn_mul(bn *a, bn *b, bn *m, bn *c)
{
	bn reductor;
	limb_t carry, old_carry;
	int i, j, k;

	cpu_bn_montgomery(a, b, m, c);

	/* reductor = (2^64)^(2*m->limbs_used) mod m */
	memset(&reductor, 0, sizeof(bn));
	reductor.limbs_used = m->limbs_used;
	reductor.limbs[1] = 1; /* reductor = 2^64 */
	for (i = 2 * m->limbs_used - 1; i > 0; i--)
		/* reductor = reductor * (2^64) mod m */
		for (j = 0; j < full_shift; j++) {
			old_carry = 0;
			for (k = 0; k <= reductor.limbs_used; k++) {
				carry = !!(reductor.limbs[k] & first_bit);
				reductor.limbs[k] <<= 1;
				reductor.limbs[k] |= old_carry;
				old_carry = carry;
			}
			reductor.limbs_used += !!reductor.limbs[reductor.limbs_used];
			cpu_bn_mod(&reductor, m);
		}

	cpu_bn_montgomery(c, &reductor, m, c);
}

/**
 * a^b mod N
 */
static void cpu_bn_exp(bn *a, bn *b, bn *m, bn *c)
{
	bn r, base;
	limb_t exp;
	int i;

	memset(&r, 0, sizeof(bn));
	r.limbs_used = m->limbs_used;
	r.limbs[0] = 1;
	base = *a;

	for (i = 0; i < b->limbs_used; i++) {
		exp = b->limbs[i];
		while (exp) {
			if (exp & 2)
				cpu_bn_mul(&r, &base, m, &r);
			exp >>= 1;
			cpu_bn_mul(&base, &base, m, &base);
		}
	}

	*c = r;
}

static void cpu_reduce(bn *a, int *d, bn *c, bn *N, int size)
{
	int i;
	bn sum, tmp;

	memset(&sum, 0, sizeof(bn));
	for (i = 0; i < size; i++) {
		memset(&tmp, 0, sizeof(bn));
		tmp.limbs_used = 1; tmp.limbs[0] = d[i];
		cpu_bn_mul(&a[i], &tmp, N, &tmp);
		cpu_bn_add(&sum, &tmp, &sum);
		cpu_bn_mod(&sum, N);
	}

	*c = sum;
}

static void bn_print(bn *x)
{
	int i;

	printf("[%ld]: [", x->limbs_used);
	for (i = 0; i < x->limbs_used; i++)
		printf("0x%016llx, ", x->limbs[i]);
	printf("0] \n");
}

static void bn_drand48(bn *x, int size)
{
	int i;

	x->limbs_used = size;
	for (i = 0; i < x->limbs_used; i++)
		x->limbs[i] = lrand48() | (lrand48() << 32);
}

int main(int argc, char *argv[])
{
	dim3 numThreads(THREADS, 1), numBlocks(BLOCKS, 1);
	int i, device = 0;
	size_t size;
	cudaEvent_t start, stop;
	float msecTotal;

	bn *gpuA, *gpuB, *gpuC;
	bn *a, *b, *c, cpuN;
	int *d;
	int *gpuD;

	size = sizeof(bn) * TOTAL_SIZE;

	/* init events and calloc vectors (calloc to ensure padding with 0) */
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	a = (bn *)calloc(size, 1);
	b = (bn *)calloc(size, 1);
	c = (bn *)calloc(sizeof(bn), 1);
	d = (int *)calloc(sizeof(int) * TOTAL_SIZE, 1);

	/* randomly fill up input vectors: use constant seed */
	srand48(42);
	for (i = 0; i < TOTAL_SIZE; i++) {
		bn_drand48(&a[i], NUMBER_SIZE);
		d[i] = lrand48() & 1;
	}

	/* setup N, don't forget to 0-it before */
	memset(&cpuN, 0, sizeof(bn));
	bn_drand48(&cpuN, NUMBER_SIZE);
	/* ensure that N is odd */
	cpuN.limbs[0] |= 1;

	/* set up vectors on device */
	cudaSetDevice(device);
	cudaMalloc((void**)&gpuA, size);
	cudaMalloc((void**)&gpuC, size);
	cudaMalloc((void**)&gpuB, size);
	cudaMalloc((void**)&gpuD, sizeof(int) * TOTAL_SIZE);

	/* copy vectors to device */
	cudaError_t err = cudaMemcpy(gpuA, a, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		printf("Err 1 %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(gpuD, d, sizeof(int) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		printf("Err 2 %s\n", cudaGetErrorString(err));
	cudaMemcpyToSymbol(N, (void*)&cpuN, sizeof(bn));

	/* GPU computation */
	cudaEventRecord(start, NULL);
	pairwiseMultiply<<<numBlocks, numThreads>>>(gpuA, gpuD, gpuC, TOTAL_SIZE);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("cannot invoke multiply kernel! %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	printf("gpu time: %.3f ms\n", msecTotal);
	exit(EXIT_SUCCESS);

	/* copy back result to CPU */
	cudaMemcpy(b, gpuC, sizeof(bn), cudaMemcpyDeviceToHost);

	/* CPU computation */
	cudaEventRecord(start, NULL);
	cpu_reduce(a, d, c, &cpuN, TOTAL_SIZE);
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	printf("cpu time: %.3f ms\n", msecTotal);

	return 0;
}
