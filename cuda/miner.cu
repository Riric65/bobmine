// Kernel CUDA : sha256("{prefix}:{nonce}") avec 32+ bits à zéro.
//
// Spécialisé pour le cas réel du serveur bobphone :
//   prefix = 48 caractères hex, nonce = 6 caractères sur l'alphabet
//   [0x21..0x7e] sauf ':' (93 symboles) -> message de 55 octets = 1 seul bloc
//   SHA-256 (padding : 0x80, longueur 440 bits).
//
// Le mot "index" (base 93, poids fort d'abord) est le même que celui que le
// mineur Rust utilise : un indice trouvé ici est reconverti côté hôte avec
// Engine::nonce_at().
//
// Compilation bench :  nvcc -O3 -DBENCH cuda/miner.cu -o bench
// Compilation PTX   :  nvcc -O3 -ptx cuda/miner.cu -o miner.ptx

#include <stdint.h>

#define ALPHA_LEN 93
#define NONCE_LEN 6
#define PREFIX_WORDS 12 // 48 octets de prefix

extern "C" {
__device__ __constant__ uint32_t c_pre[PREFIX_WORDS]; // mots big-endian du prefix
__device__ __constant__ uint32_t c_K[64];
}

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return z ^ (x & (y ^ z));
}
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) | (z & (x | y));
}
__device__ __forceinline__ uint32_t S0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}
__device__ __forceinline__ uint32_t S1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}
__device__ __forceinline__ uint32_t s0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}
__device__ __forceinline__ uint32_t s1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Alphabet identique à build_alphabet() côté Rust : 0x21..0x7e sauf ':' (0x3a).
// 0x3a - 0x21 = 25, donc l'indice 25 est le premier sauté :
// indices 0..24 -> 0x21..0x39 ('!'..'9'), 25..92 -> 0x3b..0x7e (';'..'~')
__device__ __forceinline__ unsigned char alpha_char(unsigned char i) {
    return i < 25 ? (unsigned char)(0x21 + i) : (unsigned char)(0x22 + i);
}

// Le hash commence-t-il par `bits` bits à zéro ? `a` = H0, `b` = H1.
__device__ __forceinline__ int hits(uint32_t a, uint32_t b, unsigned int bits) {
    if (a != 0) return 0;
    if (bits <= 32) return 1;
    return (b >> (64 - bits)) == 0;
}

// Hash du message "prefix:nonce" (nonce = 6 caractères), un seul bloc SHA-256.
__device__ void hash_nonce(const unsigned char d[NONCE_LEN], uint32_t out[8]) {
    uint32_t W[64];
#pragma unroll
    for (int j = 0; j < PREFIX_WORDS; ++j) W[j] = c_pre[j];

    uint32_t n0 = alpha_char(d[0]);
    uint32_t n1 = alpha_char(d[1]);
    uint32_t n2 = alpha_char(d[2]);
    uint32_t n3 = alpha_char(d[3]);
    uint32_t n4 = alpha_char(d[4]);
    uint32_t n5 = alpha_char(d[5]);

    // octets 48..51 : ':' n0 n1 n2
    W[12] = 0x3a000000u | (n0 << 16) | (n1 << 8) | n2;
    // octets 52..55 : n3 n4 n5 0x80
    W[13] = (n3 << 24) | (n4 << 16) | (n5 << 8) | 0x80u;
    W[14] = 0;
    W[15] = 440; // 55 octets * 8

#pragma unroll
    for (int t = 16; t < 64; ++t)
        W[t] = s1(W[t - 2]) + W[t - 7] + s0(W[t - 15]) + W[t - 16];

    uint32_t a = 0x6a09e667u, b = 0xbb67ae85u, c = 0x3c6ef372u,
             d0 = 0xa54ff53au, e = 0x510e527fu, f = 0x9b05688cu,
             g = 0x1f83d9abu, h = 0x5be0cd19u;

#pragma unroll
    for (int t = 0; t < 64; ++t) {
        uint32_t t1 = h + S1(e) + Ch(e, f, g) + c_K[t] + W[t];
        uint32_t t2 = S0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d0 + t1;
        d0 = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    out[0] = a + 0x6a09e667u;
    out[1] = b + 0xbb67ae85u;
    out[2] = c + 0x3c6ef372u;
    out[3] = d0 + 0xa54ff53au;
    out[4] = e + 0x510e527fu;
    out[5] = f + 0x9b05688cu;
    out[6] = g + 0x1f83d9abu;
    out[7] = h + 0x5be0cd19u;
}

/// Debug : mots du bloc de message (avant le message schedule).
extern "C" __global__ void bob_msg_one(unsigned long long idx,
                                       uint32_t *__restrict__ out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned char d[NONCE_LEN];
    unsigned long long x = idx;
#pragma unroll
    for (int i = NONCE_LEN - 1; i >= 0; --i) {
        d[i] = (unsigned char)(x % ALPHA_LEN);
        x /= ALPHA_LEN;
    }
    uint32_t W[16];
#pragma unroll
    for (int j = 0; j < PREFIX_WORDS; ++j) W[j] = c_pre[j];
    uint32_t n0 = alpha_char(d[0]);
    uint32_t n1 = alpha_char(d[1]);
    uint32_t n2 = alpha_char(d[2]);
    uint32_t n3 = alpha_char(d[3]);
    uint32_t n4 = alpha_char(d[4]);
    uint32_t n5 = alpha_char(d[5]);
    W[12] = 0x3a000000u | (n0 << 16) | (n1 << 8) | n2;
    W[13] = (n3 << 24) | (n4 << 16) | (n5 << 8) | 0x80u;
    W[14] = 0;
    W[15] = 440;
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = W[i];
}

/// Debug : hash exact d'un indice donné (un seul thread).
extern "C" __global__ void bob_hash_one(unsigned long long idx,
                                        uint32_t *__restrict__ out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned char d[NONCE_LEN];
    unsigned long long x = idx;
#pragma unroll
    for (int i = NONCE_LEN - 1; i >= 0; --i) {
        d[i] = (unsigned char)(x % ALPHA_LEN);
        x /= ALPHA_LEN;
    }
    uint32_t o[8];
    hash_nonce(d, o);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = o[i];
}

extern "C" __global__ void bob_mine(unsigned long long start, unsigned long long count,
                                    unsigned long long *__restrict__ found,
                                    unsigned int bits) {
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long nthreads =
        (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long per = (count + nthreads - 1) / nthreads;
    unsigned long long begin = start + tid * per;
    unsigned long long limit = start + count;
    unsigned long long end = begin + per < limit ? begin + per : limit;
    if (begin >= end) return;

    // Index de départ -> chiffres en base 93 (poids fort d'abord).
    unsigned char d[NONCE_LEN];
    unsigned long long x = begin;
#pragma unroll
    for (int i = NONCE_LEN - 1; i >= 0; --i) {
        d[i] = (unsigned char)(x % ALPHA_LEN);
        x /= ALPHA_LEN;
    }

    for (unsigned long long i = begin; i < end; ++i) {
        // Sortie anticipée dès qu'une solution existe (vérifié toutes les 1024).
        if ((i & 0x3FFULL) == 0 && *found != ~0ULL) return;

        uint32_t out[8];
        hash_nonce(d, out);
        if (hits(out[0], out[1], bits)) atomicMin(found, i);

        // Odomètre +1 en base 93.
#pragma unroll
        for (int p = NONCE_LEN - 1; p >= 0; --p) {
            d[p] = (unsigned char)(d[p] + 1);
            if (d[p] < ALPHA_LEN) break;
            d[p] = 0;
            if (p == 0) return; // espace épuisé
        }
    }
}

// ---------------------------------------------------------------------------
// Bench standalone (nvcc -DBENCH)
// ---------------------------------------------------------------------------
#ifdef BENCH
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const uint32_t K32[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

#define CHK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1);} } while (0)

int main(int argc, char **argv) {
    const char *prefix = argc > 1 ? argv[1] : "5655991c5382421a44fad16a2d8ce0b415f5e8a4bbf38fa0";
    unsigned int bits = argc > 2 ? (unsigned int)atoi(argv[2]) : 32;
    unsigned long long total = argc > 3 ? strtoull(argv[3], 0, 10) : 4000000000ULL;
    if (strlen(prefix) != 48) { fprintf(stderr, "prefix doit faire 48 octets\n"); return 1; }

    // Le prefix fait 48 caractères = 48 octets ASCII = 12 mots SHA-256
    // (on ne décode PAS l'hex : le serveur hache la chaîne littérale).
    uint32_t pre[PREFIX_WORDS];
    for (int i = 0; i < PREFIX_WORDS; i++) {
        uint32_t v = 0;
        for (int j = 0; j < 4; j++) v = (v << 8) | (unsigned char)prefix[i * 4 + j];
        pre[i] = v;
    }

    CHK(cudaMemcpyToSymbol(c_pre, pre, sizeof(pre)));
    CHK(cudaMemcpyToSymbol(c_K, K32, sizeof(K32)));

    // Mode debug : "bench <prefix> --hash <index>" -> hash GPU d'un indice.
    if (argc > 2 && strcmp(argv[2], "--hash") == 0) {
        unsigned long long idx = strtoull(argv[3], 0, 10);
        uint32_t *d_out;
        CHK(cudaMalloc(&d_out, 16 * sizeof(uint32_t)));
        bob_hash_one<<<1, 1>>>(idx, d_out);
        CHK(cudaGetLastError());
        uint32_t out[16];
        CHK(cudaMemcpy(out, d_out, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost));
        printf("gpu_hash ");
        for (int i = 0; i < 8; i++) printf("%08x", out[i]);
        printf("\n");
        bob_msg_one<<<1, 1>>>(idx, d_out);
        CHK(cudaGetLastError());
        CHK(cudaMemcpy(out, d_out, sizeof(uint32_t) * 16, cudaMemcpyDeviceToHost));
        printf("gpu_msg  ");
        for (int i = 0; i < 16; i++) printf("%08x", out[i]);
        printf("\n");
        CHK(cudaFree(d_out));
        uint32_t cpre_read[12];
        CHK(cudaMemcpyFromSymbol(cpre_read, c_pre, sizeof(cpre_read)));
        printf("c_pre    ");
        for (int i = 0; i < 12; i++) printf("%08x", cpre_read[i]);
        printf("\n");
        uint32_t ck_read[4];
        CHK(cudaMemcpyFromSymbol(ck_read, c_K, sizeof(ck_read)));
        printf("c_K[0-3] %08x %08x %08x %08x (attendu 428a2f98 71374491 b5c0fbcf e9b5dba5)\n",
               ck_read[0], ck_read[1], ck_read[2], ck_read[3]);
        return 0;
    }

    unsigned long long *d_found;
    CHK(cudaMalloc(&d_found, sizeof(unsigned long long)));
    CHK(cudaMemset(d_found, 0xFF, sizeof(unsigned long long)));

    int dev = 0;
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s  SM=%d\n", prop.name, prop.multiProcessorCount);

    int block = 256;
    int grid = prop.multiProcessorCount * 8;
    unsigned long long threads = (unsigned long long)grid * block;

    // Découpe en lots pour rester réactif et pouvoir mesurer.
    unsigned long long batch = threads * 4096ULL;
    if (batch > total) batch = total;

    clock_t t0 = clock();
    struct timespec w0, w1;
    clock_gettime(CLOCK_MONOTONIC, &w0);

    unsigned long long done = 0, launches = 0;
    unsigned long long found = ~0ULL;
    while (done < total) {
        unsigned long long n = batch;
        if (done + n > total) n = total - done;
        bob_mine<<<grid, block>>>(done, n, d_found, bits);
        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(&found, d_found, sizeof(found), cudaMemcpyDeviceToHost));
        done += n;
        launches++;
        if (found != ~0ULL) break;
    }

    clock_gettime(CLOCK_MONOTONIC, &w1);
    double dt = (w1.tv_sec - w0.tv_sec) + (w1.tv_nsec - w0.tv_nsec) / 1e9;
    (void)t0;

    printf("testés   : %llu hashes en %.3f s\n", done, dt);
    printf("hashrate : %.0f H/s  (%llu lancements)\n", done / dt, launches);
    if (found != ~0ULL) {
        unsigned long long idx = found;
        unsigned long long x = idx;
        char nonce[NONCE_LEN + 1];
        for (int i = NONCE_LEN - 1; i >= 0; --i) {
            unsigned char digit = (unsigned char)(x % ALPHA_LEN);
            nonce[i] = (char)(digit < 27 ? (0x21 + digit) : (0x22 + digit));
            x /= ALPHA_LEN;
        }
        nonce[NONCE_LEN] = 0;
        printf("solution : index=%llu nonce=%s\n", idx, nonce);
        printf("message  : %s:%s\n", prefix, nonce);
    } else {
        printf("solution : aucune sur cet échantillon\n");
    }
    CHK(cudaFree(d_found));
    return 0;
}
#endif
