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
#define THREADS 128
#endif
#ifndef BLOCKS
#define BLOCKS 16
#endif
#define TOTAL_THREADS (THREADS * BLOCKS)
#ifndef SIZE
#define SIZE 32
#endif

/* no size in here */
#define TOTAL_SIZE (TOTAL_THREADS)

/* size of the exponent in 64 bits, at least BASE, up to LIMB_COUNT/2 (not-checked) */
#ifndef EXP
#define EXP 16
#endif

/* size of the base in 64 bits (always 16 = 128 bytes) */
#define BASE 16

/* DBSize is D * BLOCKS * THREADS bytes */

/* number of digits in base 2^64 of the maximum representable number */
#ifndef LIMB_COUNT
#define LIMB_COUNT 40
#endif

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
__constant__ bn Base;
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

__global__ void work(bn *exp, bn *res)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	bn_exp(&Base, &exp[ix], &N, &res[ix]);
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

	bn *gpuExps, *gpuResults;
	bn *exps, *results, cpuN, cpuBase;

	size = sizeof(bn) * TOTAL_SIZE;

	/* init events and calloc vectors (calloc to ensure padding with 0) */
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	exps = (bn *)calloc(size, 1);
	results = (bn *)calloc(size, 1);

	/* randomly fill up input vectors: use constant seed */
	srand48(42);
	for (i = 0; i < TOTAL_SIZE; i++) {
		bn_drand48(&exps[i], EXP);
	}

	/* setup N, don't forget to 0-it before */
	memset(&cpuN, 0, sizeof(bn));
	bn_drand48(&cpuN, BASE);
	/* ensure that N is odd */
	cpuN.limbs[0] |= 1;

	/* the same for the base */
	memset(&cpuBase, 0, sizeof(bn));
	bn_drand48(&cpuBase, BASE);
	/* ensure that base is odd */
	cpuBase.limbs[0] |= 1;

	/* set up vectors on device */
	cudaSetDevice(device);
	cudaMalloc((void**)&gpuExps, size);
	cudaMalloc((void**)&gpuResults, size);

	/* copy vectors to device */
	cudaError_t err = cudaMemcpy(gpuExps, exps, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		printf("Err 1 %s\n", cudaGetErrorString(err));
	cudaMemcpyToSymbol(N, (void*)&cpuN, sizeof(bn));
	cudaMemcpyToSymbol(Base, (void*)&cpuBase, sizeof(bn));

	/* GPU computation */
	cudaEventRecord(start, NULL);
	work<<<numBlocks, numThreads>>>(gpuExps, gpuResults);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("cannot invoke work kernel! %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	printf("gpu time: %.3f ms\n", msecTotal);

	return 0;
}
