/*
 * dga_transformer_v3_safe.cu
 * Decoupled Geometric Attention Transformer (Safer / Revised)
 *
 * Идея сохранена:
 *  - 2D Q/K + Rich V
 *  - Parallel Top-K Causal Sparse Attention
 *  - SwiGLU FFN
 *  - 2D RoPE
 *  - masked CE only on assistant span
 *
 * Исправления:
 *  - safer kernel launches + checks
 *  - row-major GEMM wrappers clarified
 *  - parallel CE softmax
 *  - embedding kernels no longer assume blockDim=D
 *  - config validation
 *  - cleanup / destructors
 *
 * Компиляция (GTX 1060):
 *   nvcc -O3 -arch=sm_61 -std=c++17 -allow-unsupported-compiler ^
 *        -o dga_transformer dga_transformer_v3_safe.cu -lcublas ^
 *        -Xcompiler "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
 *
 * RTX 3070: sm_86
 * RTX 4070: sm_89
 */

#include <io.h>
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <windows.h>
#include <cstring>
#include <cassert>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t s = (x); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS %s:%d status=%d\n", __FILE__, __LINE__, (int)s); \
        exit(1); \
    } \
} while(0)

#define CUDA_LAUNCH(kernel_call) do { \
    kernel_call; \
    CUDA_CHECK(cudaGetLastError()); \
} while(0)

static cublasHandle_t cublas = nullptr;

// ============================================================
//  CONFIG
// ============================================================
struct Config {
    int vocab_size = 0;
    int d_model = 768;
    int n_heads = 32;
    int d_qk = 2;
    int d_v = 16;
    int n_layers = 12;
    int d_ff = 3072;
    int seq_len = 1024;
    int batch_size = 4;
    int micro_batch = 2;
    int top_k_attn = 12;
    float lr = 1e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps_adam = 1e-8f;

    int total_qk() const { return n_heads * d_qk; }
    int total_v()  const { return n_heads * d_v;  }

    void validate() const {
        if (vocab_size <= 0) {
            fprintf(stderr, "[Config] vocab_size must be > 0\n");
            exit(1);
        }
        if (d_model <= 0 || n_heads <= 0 || d_qk <= 0 || d_v <= 0 || n_layers <= 0 || d_ff <= 0 || seq_len <= 0) {
            fprintf(stderr, "[Config] invalid positive dims\n");
            exit(1);
        }
        if (d_qk != 2) {
            fprintf(stderr, "[Config] this implementation requires d_qk == 2 (got %d)\n", d_qk);
            exit(1);
        }
        if (top_k_attn <= 0) {
            fprintf(stderr, "[Config] top_k_attn must be > 0\n");
            exit(1);
        }
    }
};

// ============================================================
//  TOKENIZER
// ============================================================
struct Tokenizer {
    struct MergeRule { int a=0, b=0, new_id=0; };

    std::vector<std::string> id2piece;
    std::unordered_map<std::string,int> piece2id;
    std::vector<MergeRule> merges;
    int unk_id=0, pad_id=1, eos_id=2;
    static constexpr int special_count = 3;
    static constexpr const char* EOS_STR = "<eos>";

    static uint64_t pair_key(int a, int b) {
        return (uint64_t)(uint32_t)a << 32 | (uint32_t)b;
    }

    std::vector<std::string> split_special_aware(const std::string& s) const {
        std::vector<std::string> r;
        size_t i = 0;
        const size_t eos_len = std::strlen(EOS_STR);
        while (i < s.size()) {
            if (i + eos_len <= s.size() && s.compare(i, eos_len, EOS_STR) == 0) {
                r.push_back(EOS_STR);
                i += eos_len;
                continue;
            }
            unsigned char c = (unsigned char)s[i];
            int l = 1;
            if ((c & 0xF8) == 0xF0) l = 4;
            else if ((c & 0xF0) == 0xE0) l = 3;
            else if ((c & 0xE0) == 0xC0) l = 2;
            r.push_back(s.substr(i, l));
            i += l;
        }
        return r;
    }

    void rebuild_piece_map() {
        piece2id.clear();
        for (int i = 0; i < (int)id2piece.size(); ++i) {
            if (!piece2id.count(id2piece[i])) piece2id[id2piece[i]] = i;
        }
        unk_id = piece2id.count("<unk>") ? piece2id["<unk>"] : 0;
        pad_id = piece2id.count("<pad>") ? piece2id["<pad>"] : 1;
        eos_id = piece2id.count(EOS_STR) ? piece2id[EOS_STR] : 2;
    }

    void apply_merge(std::vector<int>& seq, const MergeRule& m) const {
        if (seq.size() < 2) return;
        std::vector<int> out;
        out.reserve(seq.size());
        for (size_t i = 0; i < seq.size();) {
            if (i + 1 < seq.size() && seq[i] == m.a && seq[i + 1] == m.b) {
                out.push_back(m.new_id);
                i += 2;
            } else {
                out.push_back(seq[i]);
                ++i;
            }
        }
        seq.swap(out);
    }

    void build(const std::string& text, int target_vocab = 2048) {
        id2piece = {"<unk>", "<pad>", EOS_STR};
        piece2id.clear();
        piece2id["<unk>"] = 0;
        piece2id["<pad>"] = 1;
        piece2id[EOS_STR] = 2;
        merges.clear();
        unk_id = 0; pad_id = 1; eos_id = 2;

        auto pieces = split_special_aware(text);
        std::vector<int> seq;
        seq.reserve(pieces.size());

        for (const auto& p : pieces) {
            if (p == EOS_STR) {
                seq.push_back(eos_id);
                continue;
            }
            auto it = piece2id.find(p);
            if (it == piece2id.end()) {
                int id = (int)id2piece.size();
                piece2id[p] = id;
                id2piece.push_back(p);
                seq.push_back(id);
            } else {
                seq.push_back(it->second);
            }
        }

        target_vocab = std::max(target_vocab, (int)id2piece.size());
        while ((int)id2piece.size() < target_vocab && seq.size() >= 2) {
            std::unordered_map<uint64_t, int> counts;
            counts.reserve(seq.size() * 2);

            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                if (seq[i] < special_count || seq[i + 1] < special_count) continue;
                counts[pair_key(seq[i], seq[i + 1])]++;
            }

            uint64_t best_key = 0;
            int best_count = 1;
            for (const auto& kv : counts) {
                if (kv.second > best_count) {
                    best_count = kv.second;
                    best_key = kv.first;
                }
            }
            if (best_count < 2) break;

            int a = (int)(best_key >> 32);
            int b = (int)(best_key & 0xffffffffu);
            int new_id = (int)id2piece.size();

            id2piece.push_back(id2piece[a] + id2piece[b]);
            merges.push_back({a, b, new_id});
            apply_merge(seq, merges.back());
        }

        rebuild_piece_map();
        printf("[Tokenizer:BPE+EOS] vocab=%d merges=%d\n", (int)id2piece.size(), (int)merges.size());
    }

    std::vector<int> encode(const std::string& t) const {
        std::vector<int> seq;
        seq.reserve(t.size());
        for (const auto& p : split_special_aware(t)) {
            if (p == EOS_STR) {
                seq.push_back(eos_id);
                continue;
            }
            auto it = piece2id.find(p);
            seq.push_back(it != piece2id.end() ? it->second : unk_id);
        }
        for (const auto& m : merges) apply_merge(seq, m);
        return seq;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string r;
        for (int id : ids) {
            if (id < 0 || id >= (int)id2piece.size()) continue;
            if (id == unk_id || id == pad_id || id == eos_id) continue;
            r += id2piece[id];
        }
        return r;
    }

    int vocab_size() const { return (int)id2piece.size(); }

    void save(std::ofstream& f) const {
        int vs = vocab_size();
        f.write((char*)&vs, 4);
        for (const auto& s : id2piece) {
            int l = (int)s.size();
            f.write((char*)&l, 4);
            f.write(s.data(), l);
        }
        int mc = (int)merges.size();
        f.write((char*)&mc, 4);
        for (const auto& m : merges) {
            f.write((char*)&m.a, 4);
            f.write((char*)&m.b, 4);
            f.write((char*)&m.new_id, 4);
        }
    }

    void load(std::ifstream& f) {
        int vs = 0;
        f.read((char*)&vs, 4);
        id2piece.clear();
        id2piece.reserve(vs);
        for (int i = 0; i < vs; ++i) {
            int l = 0;
            f.read((char*)&l, 4);
            std::string s(l, '\0');
            f.read(&s[0], l);
            id2piece.push_back(s);
        }

        int mc = 0;
        f.read((char*)&mc, 4);
        merges.resize(mc);
        for (auto& m : merges) {
            f.read((char*)&m.a, 4);
            f.read((char*)&m.b, 4);
            f.read((char*)&m.new_id, 4);
        }
        rebuild_piece_map();
    }
};

// ============================================================
//  BASIC CUDA KERNELS
// ============================================================

__global__ void embedding_fwd(
    float* x, const float* emb, const float* pos,
    const int* tok, int T, int D
) {
    int t = blockIdx.x;
    if (t >= T) return;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        x[t * D + i] = emb[tok[t] * D + i] + pos[t * D + i];
    }
}

__global__ void layer_norm_fwd(
    float* y, const float* x, const float* g, const float* b, int T, int D
) {
    int t = blockIdx.x;
    if (t >= T) return;

    const float* xr = x + t * D;
    float* yr = y + t * D;

    __shared__ float sm[256];
    float s = 0.0f;

    for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i];
    sm[threadIdx.x] = s;
    __syncthreads();

    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sm[threadIdx.x] += sm[threadIdx.x + w];
        __syncthreads();
    }
    float mean = sm[0] / D;

    s = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float d = xr[i] - mean;
        s += d * d;
    }
    sm[threadIdx.x] = s;
    __syncthreads();

    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sm[threadIdx.x] += sm[threadIdx.x + w];
        __syncthreads();
    }

    float inv = rsqrtf(sm[0] / D + 1e-5f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        yr[i] = (xr[i] - mean) * inv * g[i] + b[i];
    }
}

__global__ void swiglu_fwd(float* out, const float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        out[i] = g * sig * up[i];
    }
}

__global__ void add_bias(float* x, const float* b, int T, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * D) x[i] += b[i % D];
}

__global__ void add_inplace(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void scale_kernel(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void zero_buf(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

// ============================================================
//  2D RoPE
// ============================================================
__global__ void apply_rope_2d(float* qk, int T, int n_heads, int d_qk, float base_theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * n_heads;
    if (idx >= total) return;

    int t = idx / n_heads;
    int h = idx % n_heads;

    float freq = powf(base_theta, -2.0f * h / (float)n_heads);
    float angle = t * freq;
    float cos_a = cosf(angle), sin_a = sinf(angle);

    int offset = t * n_heads * d_qk + h * d_qk;
    float x0 = qk[offset];
    float x1 = qk[offset + 1];
    qk[offset]     = x0 * cos_a - x1 * sin_a;
    qk[offset + 1] = x0 * sin_a + x1 * cos_a;
}

__global__ void apply_rope_2d_bwd(float* dqk, int T, int n_heads, int d_qk, float base_theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * n_heads;
    if (idx >= total) return;

    int t = idx / n_heads;
    int h = idx % n_heads;

    float freq = powf(base_theta, -2.0f * h / (float)n_heads);
    float angle = t * freq;
    float cos_a = cosf(angle), sin_a = sinf(angle);

    int offset = t * n_heads * d_qk + h * d_qk;
    float dx0 = dqk[offset];
    float dx1 = dqk[offset + 1];
    dqk[offset]     =  dx0 * cos_a + dx1 * sin_a;
    dqk[offset + 1] = -dx0 * sin_a + dx1 * cos_a;
}

// ============================================================
//  TOP-K ATTENTION
// ============================================================
#define WARP_SIZE 32
#define MAX_TOPK 16

__global__ void topk_causal_attn_fwd_parallel(
    const float* __restrict__ Q,       // [T, n_heads * d_qk]
    const float* __restrict__ K,       // [T, n_heads * d_qk]
    const float* __restrict__ V,       // [T, n_heads * d_v]
    float* __restrict__ out,           // [T, n_heads * d_v]
    float* __restrict__ scores_out,    // [T * n_heads * K_topk]
    int*   __restrict__ indices_out,   // [T * n_heads * K_topk]
    int T, int n_heads, int d_qk, int d_v, int K_topk, float scale
) {
    int block_id = blockIdx.x;
    int t = block_id / n_heads;
    int h = block_id % n_heads;
    if (t >= T) return;

    int lane = threadIdx.x;
    int causal_len = t + 1;

    float q0 = Q[t * n_heads * d_qk + h * d_qk];
    float q1 = Q[t * n_heads * d_qk + h * d_qk + 1];

    float local_scores[MAX_TOPK];
    int   local_indices[MAX_TOPK];
    int local_count = 0;

    #pragma unroll
    for (int i = 0; i < MAX_TOPK; i++) {
        local_scores[i] = -1e30f;
        local_indices[i] = -1;
    }

    for (int j = lane; j < causal_len; j += WARP_SIZE) {
        float k0 = K[j * n_heads * d_qk + h * d_qk];
        float k1 = K[j * n_heads * d_qk + h * d_qk + 1];
        float sc = (q0 * k0 + q1 * k1) * scale;

        if (local_count < K_topk) {
            int pos = local_count;
            while (pos > 0 && sc > local_scores[pos - 1]) {
                local_scores[pos] = local_scores[pos - 1];
                local_indices[pos] = local_indices[pos - 1];
                pos--;
            }
            local_scores[pos] = sc;
            local_indices[pos] = j;
            local_count++;
        } else if (sc > local_scores[K_topk - 1]) {
            int pos = K_topk - 1;
            while (pos > 0 && sc > local_scores[pos - 1]) {
                local_scores[pos] = local_scores[pos - 1];
                local_indices[pos] = local_indices[pos - 1];
                pos--;
            }
            local_scores[pos] = sc;
            local_indices[pos] = j;
        }
    }

    __shared__ float sm_scores[WARP_SIZE * MAX_TOPK];
    __shared__ int   sm_indices[WARP_SIZE * MAX_TOPK];

    for (int i = 0; i < K_topk; i++) {
        sm_scores[lane * MAX_TOPK + i] = local_scores[i];
        sm_indices[lane * MAX_TOPK + i] = local_indices[i];
    }
    for (int i = K_topk; i < MAX_TOPK; i++) {
        sm_scores[lane * MAX_TOPK + i] = -1e30f;
        sm_indices[lane * MAX_TOPK + i] = -1;
    }
    __syncthreads();

    if (lane == 0) {
        float final_scores[MAX_TOPK];
        int   final_indices[MAX_TOPK];
        int ptrs[WARP_SIZE];

        for (int i = 0; i < MAX_TOPK; i++) {
            final_scores[i] = -1e30f;
            final_indices[i] = -1;
        }
        for (int w = 0; w < WARP_SIZE; w++) ptrs[w] = 0;

        int actual_k = (causal_len < K_topk) ? causal_len : K_topk;

        for (int step = 0; step < actual_k; step++) {
            float best = -1e30f;
            int best_w = -1;
            for (int w = 0; w < WARP_SIZE; w++) {
                if (ptrs[w] < K_topk) {
                    float val = sm_scores[w * MAX_TOPK + ptrs[w]];
                    if (val > best) {
                        best = val;
                        best_w = w;
                    }
                }
            }
            if (best_w < 0) break;
            final_scores[step] = best;
            final_indices[step] = sm_indices[best_w * MAX_TOPK + ptrs[best_w]];
            ptrs[best_w]++;
        }

        float mx = final_scores[0];
        float sum = 0.0f;
        for (int i = 0; i < actual_k; i++) {
            final_scores[i] = expf(final_scores[i] - mx);
            sum += final_scores[i];
        }
        if (sum > 0.0f) {
            for (int i = 0; i < actual_k; i++) final_scores[i] /= sum;
        }

        for (int i = 0; i < MAX_TOPK; i++) {
            sm_scores[i] = (i < actual_k) ? final_scores[i] : 0.0f;
            sm_indices[i] = (i < actual_k) ? final_indices[i] : -1;
        }

        int save_offset = (t * n_heads + h) * K_topk;
        for (int i = 0; i < K_topk; i++) {
            scores_out[save_offset + i] = (i < actual_k) ? final_scores[i] : 0.0f;
            indices_out[save_offset + i] = (i < actual_k) ? final_indices[i] : -1;
        }
    }
    __syncthreads();

    int out_offset = t * n_heads * d_v + h * d_v;
    int actual_k = (causal_len < K_topk) ? causal_len : K_topk;
    for (int d = lane; d < d_v; d += WARP_SIZE) {
        float acc = 0.0f;
        for (int i = 0; i < actual_k; i++) {
            float w = sm_scores[i];
            int j = sm_indices[i];
            if (j >= 0 && w > 0.0f) {
                acc += w * V[j * n_heads * d_v + h * d_v + d];
            }
        }
        out[out_offset + d] = acc;
    }
}

__global__ void topk_causal_attn_bwd_parallel(
    const float* __restrict__ dout,        // [T, n_heads * d_v]
    const float* __restrict__ scores,      // [T * n_heads * K_topk]
    const int*   __restrict__ indices,     // [T * n_heads * K_topk]
    const float* __restrict__ Q,           // [T, n_heads * d_qk]
    const float* __restrict__ K,           // [T, n_heads * d_qk]
    const float* __restrict__ V,           // [T, n_heads * d_v]
    float* __restrict__ dQ,                // [T, n_heads * d_qk]
    float* __restrict__ dK,                // [T, n_heads * d_qk]
    float* __restrict__ dV,                // [T, n_heads * d_v]
    int T, int n_heads, int d_qk, int d_v, int K_topk, float scale
) {
    int block_id = blockIdx.x;
    int t = block_id / n_heads;
    int h = block_id % n_heads;
    if (t >= T) return;

    int lane = threadIdx.x;
    int causal_len = t + 1;
    int actual_k = (causal_len < K_topk) ? causal_len : K_topk;
    int save_offset = (t * n_heads + h) * K_topk;
    int dout_offset = t * n_heads * d_v + h * d_v;

    __shared__ float sm_w[MAX_TOPK];
    __shared__ int   sm_j[MAX_TOPK];
    __shared__ float sm_ds[MAX_TOPK];
    __shared__ float sm_dpre[MAX_TOPK];

    if (lane < K_topk) {
        sm_w[lane] = scores[save_offset + lane];
        sm_j[lane] = indices[save_offset + lane];
    }
    __syncthreads();

    for (int i = 0; i < actual_k; i++) {
        int j = sm_j[i];
        float partial = 0.0f;
        if (j >= 0) {
            for (int d = lane; d < d_v; d += WARP_SIZE) {
                partial += dout[dout_offset + d] * V[j * n_heads * d_v + h * d_v + d];
            }
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        }
        if (lane == 0) sm_ds[i] = partial;
    }
    __syncthreads();

    if (lane == 0) {
        float dot_sd = 0.0f;
        for (int i = 0; i < actual_k; i++) dot_sd += sm_w[i] * sm_ds[i];
        for (int i = 0; i < actual_k; i++) sm_dpre[i] = sm_w[i] * (sm_ds[i] - dot_sd) * scale;
    }
    __syncthreads();

    for (int i = 0; i < actual_k; i++) {
        int j = sm_j[i];
        float w = sm_w[i];
        if (j >= 0 && w > 0.0f) {
            for (int d = lane; d < d_v; d += WARP_SIZE) {
                atomicAdd(&dV[j * n_heads * d_v + h * d_v + d], w * dout[dout_offset + d]);
            }
        }
    }

    if (lane == 0) {
        float dq0 = 0.0f, dq1 = 0.0f;
        for (int i = 0; i < actual_k; i++) {
            int j = sm_j[i];
            if (j < 0) continue;
            float k0 = K[j * n_heads * d_qk + h * d_qk];
            float k1 = K[j * n_heads * d_qk + h * d_qk + 1];
            dq0 += sm_dpre[i] * k0;
            dq1 += sm_dpre[i] * k1;
        }
        int q_offset = t * n_heads * d_qk + h * d_qk;
        dQ[q_offset]     += dq0;
        dQ[q_offset + 1] += dq1;
    }

    if (lane == 0) {
        float qt0 = Q[t * n_heads * d_qk + h * d_qk];
        float qt1 = Q[t * n_heads * d_qk + h * d_qk + 1];
        for (int i = 0; i < actual_k; i++) {
            int j = sm_j[i];
            if (j < 0) continue;
            int k_offset = j * n_heads * d_qk + h * d_qk;
            atomicAdd(&dK[k_offset],     sm_dpre[i] * qt0);
            atomicAdd(&dK[k_offset + 1], sm_dpre[i] * qt1);
        }
    }
}

// ============================================================
//  PARALLEL SOFTMAX CE + MASK
//  blockDim must be 256
// ============================================================
__global__ void softmax_ce_masked_fwd_parallel(
    float* logits, const int* tgt, const int* mask, float* loss, int T, int V
) {
    int t = blockIdx.x;
    if (t >= T) return;

    float* row = logits + t * V;

    __shared__ float sm1[256];
    __shared__ float sm2[256];

    float local_max = -1e30f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        local_max = fmaxf(local_max, row[j]);
    }
    sm1[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm1[threadIdx.x] = fmaxf(sm1[threadIdx.x], sm1[threadIdx.x + s]);
        __syncthreads();
    }
    float mx = sm1[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        row[j] = expf(row[j] - mx);
        local_sum += row[j];
    }
    sm2[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm2[threadIdx.x] += sm2[threadIdx.x + s];
        __syncthreads();
    }
    float sum = sm2[0];

    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        row[j] /= sum;
    }
    __syncthreads();

    if (mask[t]) {
        int y = tgt[t];
        if (threadIdx.x == 0) loss[t] = -logf(row[y] + 1e-9f);
        if (threadIdx.x == 0) row[y] -= 1.0f;
    } else {
        if (threadIdx.x == 0) loss[t] = 0.0f;
        for (int j = threadIdx.x; j < V; j += blockDim.x) row[j] = 0.0f;
    }
}

// ============================================================
//  BACKWARD KERNELS
// ============================================================
__global__ void layer_norm_bwd(
    float* dx, float* dg, float* db,
    const float* dy, const float* x_in, const float* g, int T, int D
) {
    int t = blockIdx.x;
    if (t >= T) return;

    const float* xr = x_in + t * D;
    const float* dyr = dy + t * D;
    float* dxr = dx + t * D;

    __shared__ float sm[256];
    float s = 0.0f;

    for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i];
    sm[threadIdx.x] = s;
    __syncthreads();

    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sm[threadIdx.x] += sm[threadIdx.x + w];
        __syncthreads();
    }
    float mean = sm[0] / D;

    s = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float d = xr[i] - mean;
        s += d * d;
    }
    sm[threadIdx.x] = s;
    __syncthreads();

    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sm[threadIdx.x] += sm[threadIdx.x + w];
        __syncthreads();
    }
    float inv = rsqrtf(sm[0] / D + 1e-5f);

    float s1 = 0.0f, s2 = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float xh = (xr[i] - mean) * inv;
        float dyg = dyr[i] * g[i];
        s1 += dyg;
        s2 += dyg * xh;
    }
    sm[threadIdx.x] = s1;
    __syncthreads();

    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sm[threadIdx.x] += sm[threadIdx.x + w];
        __syncthreads();
    }
    s1 = sm[0];
    __syncthreads();

    sm[threadIdx.x] = s2;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sm[threadIdx.x] += sm[threadIdx.x + w];
        __syncthreads();
    }
    s2 = sm[0];

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float xh = (xr[i] - mean) * inv;
        dxr[i] += inv * (dyr[i] * g[i] - s1 / D - xh * s2 / D);
        atomicAdd(&dg[i], dyr[i] * xh);
        atomicAdd(&db[i], dyr[i]);
    }
}

__global__ void swiglu_bwd(
    float* dgate, float* dup, const float* dout, const float* gate, const float* up, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = gate[i];
    float u = up[i];
    float sig = 1.0f / (1.0f + expf(-g));
    float do_val = dout[i];
    dup[i] = do_val * g * sig;
    dgate[i] = do_val * u * (sig + g * sig * (1.0f - sig));
}

__global__ void bias_bwd(float* db, const float* dy, int T, int D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float s = 0.0f;
    for (int t = 0; t < T; t++) s += dy[t * D + j];
    atomicAdd(&db[j], s);
}

__global__ void embedding_bwd(
    float* demb, float* dpos, const float* dx, const int* tok, int T, int D
) {
    int t = blockIdx.x;
    if (t >= T) return;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        atomicAdd(&demb[tok[t] * D + i], dx[t * D + i]);
        atomicAdd(&dpos[t * D + i], dx[t * D + i]);
    }
}

__global__ void adam_kernel(
    float* w, float* m, float* v, const float* g,
    int n, float lr, float b1, float b2, float eps, float bc1, float bc2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float gi = g[i];
    float mi = b1 * m[i] + (1.0f - b1) * gi;
    float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    w[i] -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
}

// ============================================================
//  GEMM WRAPPERS (row-major abstraction)
//
//  We treat A[m,k], B[k,n], C[m,n] as row-major tensors.
//  cuBLAS is column-major, so we compute:
//    C_row = A_row * B_row
//  by calling:
//    C_col^T = B_col * A_col
// ============================================================

inline void mm_rm(const float* A, const float* B, float* C, int m, int k, int n,
                  float alpha=1.f, float beta=0.f) {
    CUBLAS_CHECK(cublasSgemm(
        cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        B, n,
        A, k,
        &beta,
        C, n
    ));
}

inline void mm_rm_accum_dW(const float* X, const float* dY, float* dW,
                           int m, int k, int n,
                           float alpha=1.f, float beta=1.f) {
    // X:  [m,k]
    // dY: [m,n]
    // dW: [k,n] = X^T * dY
    CUBLAS_CHECK(cublasSgemm(
        cublas,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n, k, m,
        &alpha,
        dY, n,
        X, k,
        &beta,
        dW, n
    ));
}

inline void mm_rm_back_input(const float* dY, const float* W, float* dX,
                             int m, int n, int k,
                             float alpha=1.f, float beta=1.f) {
    // dY: [m,n]
    // W : [k,n]
    // dX: [m,k] = dY * W^T
    CUBLAS_CHECK(cublasSgemm(
        cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        k, m, n,
        &alpha,
        W, n,
        dY, n,
        &beta,
        dX, k
    ));
}

// ============================================================
//  PARAM
// ============================================================
struct Param {
    int n = 0;
    float *w = nullptr, *g = nullptr, *m = nullptr, *v = nullptr, *hw = nullptr;

    Param() = default;

    Param(const Param&) = delete;
    Param& operator=(const Param&) = delete;

    Param(Param&& other) noexcept {
        n  = other.n;   other.n = 0;
        w  = other.w;   other.w = nullptr;
        g  = other.g;   other.g = nullptr;
        m  = other.m;   other.m = nullptr;
        v  = other.v;   other.v = nullptr;
        hw = other.hw;  other.hw = nullptr;
    }

    Param& operator=(Param&& other) noexcept {
        if (this != &other) {
            free_all();
            n  = other.n;   other.n = 0;
            w  = other.w;   other.w = nullptr;
            g  = other.g;   other.g = nullptr;
            m  = other.m;   other.m = nullptr;
            v  = other.v;   other.v = nullptr;
            hw = other.hw;  other.hw = nullptr;
        }
        return *this;
    }

    void alloc(int sz) {
        free_all();
        n = sz;
        hw = new float[n];
        CUDA_CHECK(cudaMalloc(&w, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&v, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(g, 0, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(m, 0, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(v, 0, n * sizeof(float)));
    }

    void init_normal(float std_val, std::mt19937& rng) {
        std::normal_distribution<float> d(0.0f, std_val);
        for (int i = 0; i < n; i++) hw[i] = d(rng);
        CUDA_CHECK(cudaMemcpy(w, hw, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    void init_ones() {
        for (int i = 0; i < n; i++) hw[i] = 1.0f;
        CUDA_CHECK(cudaMemcpy(w, hw, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    void init_zeros() {
        for (int i = 0; i < n; i++) hw[i] = 0.0f;
        CUDA_CHECK(cudaMemcpy(w, hw, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    void zero_grad() {
        if (g && n > 0) CUDA_CHECK(cudaMemset(g, 0, n * sizeof(float)));
    }

    void to_cpu() {
        if (w && hw && n > 0)
            CUDA_CHECK(cudaMemcpy(hw, w, n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void to_gpu() {
        if (w && hw && n > 0)
            CUDA_CHECK(cudaMemcpy(w, hw, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    void adam(float lr, float b1, float b2, float eps, int t) {
        if (n <= 0) return;
        float bc1 = 1.0f - powf(b1, t);
        float bc2 = 1.0f - powf(b2, t);
        int bl = (n + 255) / 256;
        CUDA_LAUNCH((adam_kernel<<<bl,256>>>(w, m, v, g, n, lr, b1, b2, eps, bc1, bc2)));
    }

    void free_all() {
        if (w)  { cudaFree(w);  w = nullptr; }
        if (g)  { cudaFree(g);  g = nullptr; }
        if (m)  { cudaFree(m);  m = nullptr; }
        if (v)  { cudaFree(v);  v = nullptr; }
        if (hw) { delete[] hw;  hw = nullptr; }
        n = 0;
    }

    ~Param() {
        free_all();
    }
};

// ============================================================
//  MODEL
// ============================================================
struct Model {
    Config cfg;
    Param emb, pos_emb;

    struct Layer {
        Param ln1g, ln1b;
        Param Wq, Wk, Wv, Wo;
        Param ln2g, ln2b;
        Param W_gate, W_up, W_down, b_down;

        Layer() = default;
        Layer(Layer&&) = default;
        Layer& operator=(Layer&&) = default;
        Layer(const Layer&) = delete;
        Layer& operator=(const Layer&) = delete;
    };
    std::vector<Layer> layers;

    Param lnfg, lnfb, lmh;
    int adam_t = 0;
    std::mt19937 rng{42};
    std::vector<Param*> params_cache;

    struct Cache {
        float *x_in = nullptr, *xn1 = nullptr, *Q = nullptr, *K = nullptr, *Q_pre_rope = nullptr, *K_pre_rope = nullptr, *V = nullptr;
        float *attn_out = nullptr, *topk_scores = nullptr;
        int   *topk_idx = nullptr;
        float *x_mid = nullptr, *xn2 = nullptr, *gate_pre = nullptr, *up_pre = nullptr, *swiglu_out = nullptr;
    };
    std::vector<Cache> cache;

    float *d_x_final = nullptr, *d_x_pre_final = nullptr;

    float *d_x = nullptr, *d_buf = nullptr, *d_buf2 = nullptr;
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    float *d_at = nullptr, *d_pr = nullptr;
    float *d_gate = nullptr, *d_up = nullptr, *d_swi = nullptr, *d_h2 = nullptr;
    float *d_logits = nullptr, *d_loss = nullptr;
    float *d_topk_scores = nullptr;
    int   *d_topk_idx = nullptr;
    float *d_dx = nullptr, *d_tmp = nullptr, *d_tmp2 = nullptr;
    float *d_dQ = nullptr, *d_dK = nullptr, *d_dV = nullptr;
    float *d_dgate = nullptr, *d_dup = nullptr;
    int   *d_tok = nullptr, *d_tgt = nullptr, *d_mask = nullptr;

    static constexpr float ROPE_BASE = 10000.0f;

    void free_cache_layer(Cache& c) {
        if (c.x_in) cudaFree(c.x_in), c.x_in = nullptr;
        if (c.xn1) cudaFree(c.xn1), c.xn1 = nullptr;
        if (c.Q) cudaFree(c.Q), c.Q = nullptr;
        if (c.K) cudaFree(c.K), c.K = nullptr;
        if (c.Q_pre_rope) cudaFree(c.Q_pre_rope), c.Q_pre_rope = nullptr;
        if (c.K_pre_rope) cudaFree(c.K_pre_rope), c.K_pre_rope = nullptr;
        if (c.V) cudaFree(c.V), c.V = nullptr;
        if (c.attn_out) cudaFree(c.attn_out), c.attn_out = nullptr;
        if (c.topk_scores) cudaFree(c.topk_scores), c.topk_scores = nullptr;
        if (c.topk_idx) cudaFree(c.topk_idx), c.topk_idx = nullptr;
        if (c.x_mid) cudaFree(c.x_mid), c.x_mid = nullptr;
        if (c.xn2) cudaFree(c.xn2), c.xn2 = nullptr;
        if (c.gate_pre) cudaFree(c.gate_pre), c.gate_pre = nullptr;
        if (c.up_pre) cudaFree(c.up_pre), c.up_pre = nullptr;
        if (c.swiglu_out) cudaFree(c.swiglu_out), c.swiglu_out = nullptr;
    }

    void alloc_cache_layer(Cache& c, int T, int D, int tqk, int tv, int FF, int nh, int Kt) {
        CUDA_CHECK(cudaMalloc(&c.x_in,        T * D   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.xn1,         T * D   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.Q,           T * tqk * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.K,           T * tqk * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.Q_pre_rope,  T * tqk * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.K_pre_rope,  T * tqk * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.V,           T * tv  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.attn_out,    T * tv  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.topk_scores, T * nh * Kt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.topk_idx,    T * nh * Kt * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&c.x_mid,       T * D   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.xn2,         T * D   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.gate_pre,    T * FF  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.up_pre,      T * FF  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.swiglu_out,  T * FF  * sizeof(float)));
    }

    void free_device_buffers() {
        auto freep = [](void* p){ if (p) cudaFree(p); };

        freep(d_x_final); d_x_final = nullptr;
        freep(d_x_pre_final); d_x_pre_final = nullptr;

        freep(d_x); d_x = nullptr;
        freep(d_buf); d_buf = nullptr;
        freep(d_buf2); d_buf2 = nullptr;
        freep(d_Q); d_Q = nullptr;
        freep(d_K); d_K = nullptr;
        freep(d_V); d_V = nullptr;
        freep(d_at); d_at = nullptr;
        freep(d_pr); d_pr = nullptr;
        freep(d_gate); d_gate = nullptr;
        freep(d_up); d_up = nullptr;
        freep(d_swi); d_swi = nullptr;
        freep(d_h2); d_h2 = nullptr;
        freep(d_logits); d_logits = nullptr;
        freep(d_loss); d_loss = nullptr;
        freep(d_topk_scores); d_topk_scores = nullptr;
        freep(d_topk_idx); d_topk_idx = nullptr;
        freep(d_dx); d_dx = nullptr;
        freep(d_tmp); d_tmp = nullptr;
        freep(d_tmp2); d_tmp2 = nullptr;
        freep(d_dQ); d_dQ = nullptr;
        freep(d_dK); d_dK = nullptr;
        freep(d_dV); d_dV = nullptr;
        freep(d_dgate); d_dgate = nullptr;
        freep(d_dup); d_dup = nullptr;
        freep(d_tok); d_tok = nullptr;
        freep(d_tgt); d_tgt = nullptr;
        freep(d_mask); d_mask = nullptr;
    }

    void cleanup() {
        free_device_buffers();
        for (auto& c : cache) free_cache_layer(c);
        cache.clear();
        layers.clear();
        params_cache.clear();
    }

    ~Model() {
        cleanup();
    }

    void init(const Config& c) {
        cleanup();
        cfg = c;
        cfg.validate();

        if (cfg.top_k_attn > MAX_TOPK) {
            fprintf(stderr, "[Config] top_k_attn=%d exceeds MAX_TOPK=%d\n", cfg.top_k_attn, MAX_TOPK);
            exit(1);
        }

        int D = cfg.d_model;
        int V = cfg.vocab_size;
        int FF = cfg.d_ff;
        int S = cfg.seq_len;
        int L = cfg.n_layers;
        int tqk = cfg.total_qk();
        int tv = cfg.total_v();
        int nh = cfg.n_heads;
        int Kt = cfg.top_k_attn;

        float sc = 1.0f / sqrtf((float)D);

        emb.alloc(V * D);       emb.init_normal(sc, rng);
        pos_emb.alloc(S * D);   pos_emb.init_normal(sc, rng);

        layers.clear();
        layers.reserve(L);
        for (int i = 0; i < L; ++i) layers.emplace_back();

        for (auto& l : layers) {
            l.ln1g.alloc(D); l.ln1g.init_ones();
            l.ln1b.alloc(D); l.ln1b.init_zeros();

            l.Wq.alloc(D * tqk); l.Wq.init_normal(sc, rng);
            l.Wk.alloc(D * tqk); l.Wk.init_normal(sc, rng);
            l.Wv.alloc(D * tv);  l.Wv.init_normal(sc, rng);
            l.Wo.alloc(tv * D);  l.Wo.init_normal(sc, rng);

            l.ln2g.alloc(D); l.ln2g.init_ones();
            l.ln2b.alloc(D); l.ln2b.init_zeros();

            l.W_gate.alloc(D * FF); l.W_gate.init_normal(sc, rng);
            l.W_up.alloc(D * FF);   l.W_up.init_normal(sc, rng);
            l.W_down.alloc(FF * D); l.W_down.init_normal(sc, rng);
            l.b_down.alloc(D);      l.b_down.init_zeros();
        }

        lnfg.alloc(D); lnfg.init_ones();
        lnfb.alloc(D); lnfb.init_zeros();
        lmh.alloc(D * V); lmh.init_normal(sc, rng);

        cache.resize(L);
        for (auto& ca : cache) alloc_cache_layer(ca, S, D, tqk, tv, FF, nh, Kt);

        CUDA_CHECK(cudaMalloc(&d_x_final,      S * D   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x_pre_final,  S * D   * sizeof(float)));

        int max_buf = std::max({S * D, S * tqk, S * tv, S * FF, S * V});

        CUDA_CHECK(cudaMalloc(&d_x,            S * D   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_buf,          max_buf * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_buf2,         max_buf * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q,            S * tqk * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K,            S * tqk * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V,            S * tv  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_at,           S * tv  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pr,           S * D   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gate,         S * FF  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_up,           S * FF  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_swi,          S * FF  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_h2,           S * D   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_logits,       S * V   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_loss,         S       * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_topk_scores,  S * nh * Kt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_topk_idx,     S * nh * Kt * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dx,           S * D   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tmp,          max_buf * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tmp2,         max_buf * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dQ,           S * tqk * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dK,           S * tqk * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dV,           S * tv  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dgate,        S * FF  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dup,          S * FF  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tok,          S       * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_tgt,          S       * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_mask,         S       * sizeof(int)));

        rebuild_param_cache();

        long long tot = 0;
        for (auto* p : all_params()) tot += p->n;

        printf("[Model DGA v3] Params: %lld (~%.1fM)\n", tot, tot / 1e6f);
        printf("[Model DGA v3] d=%d heads=%d d_qk=%d d_v=%d ff=%d top_k=%d\n",
               D, nh, cfg.d_qk, cfg.d_v, FF, Kt);

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("[CUDA] %s | %.0f GB VRAM\n", prop.name, prop.totalGlobalMem / 1e9f);
    }

    void rebuild_param_cache() {
        params_cache = {&emb, &pos_emb};
        for (auto& l : layers) {
            for (auto* p : {&l.ln1g,&l.ln1b,&l.Wq,&l.Wk,&l.Wv,&l.Wo,
                            &l.ln2g,&l.ln2b,&l.W_gate,&l.W_up,&l.W_down,&l.b_down}) {
                params_cache.push_back(p);
            }
        }
        params_cache.push_back(&lnfg);
        params_cache.push_back(&lnfb);
        params_cache.push_back(&lmh);
    }

    std::vector<Param*>& all_params() { return params_cache; }
    const std::vector<Param*>& all_params() const { return params_cache; }

    void zero_grad() {
        for (auto* p : all_params()) p->zero_grad();
    }

    void clip_gradients(float max_norm = 1.0f) {
        float total_sq = 0.0f;
        for (auto* p : all_params()) {
            if (p->n <= 0) continue;
            float nrm = 0.0f;
            CUBLAS_CHECK(cublasSnrm2(cublas, p->n, p->g, 1, &nrm));
            total_sq += nrm * nrm;
        }
        float total_norm = sqrtf(total_sq);
        if (total_norm > max_norm) {
            float s = max_norm / (total_norm + 1e-6f);
            for (auto* p : all_params()) {
                int bl = (p->n + 255) / 256;
                CUDA_LAUNCH((scale_kernel<<<bl,256>>>(p->g, s, p->n)));
            }
        }
    }

    void adam_all() {
        adam_t++;
        for (auto* p : all_params()) p->adam(cfg.lr, cfg.beta1, cfg.beta2, cfg.eps_adam, adam_t);
    }

    // ============================================================
    //  FORWARD
    // ============================================================
    void forward(const std::vector<int>& tokens_cpu, bool save_cache = false) {
        int T = (int)tokens_cpu.size();
        int D = cfg.d_model;
        int V = cfg.vocab_size;
        int FF = cfg.d_ff;
        int tqk = cfg.total_qk();
        int tv = cfg.total_v();
        int nh = cfg.n_heads;
        int dqk = cfg.d_qk;
        int dv = cfg.d_v;
        int Kt = cfg.top_k_attn;

        (void)V;
        float sa = 1.0f / sqrtf((float)dqk);

        CUDA_CHECK(cudaMemcpy(d_tok, tokens_cpu.data(), T * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_LAUNCH((embedding_fwd<<<T,256>>>(d_x, emb.w, pos_emb.w, d_tok, T, D)));

        for (int li = 0; li < cfg.n_layers; li++) {
            auto& l = layers[li];
            auto& ca = cache[li];

            if (save_cache) CUDA_CHECK(cudaMemcpy(ca.x_in, d_x, T * D * sizeof(float), cudaMemcpyDeviceToDevice));

            CUDA_LAUNCH((layer_norm_fwd<<<T,256>>>(d_buf, d_x, l.ln1g.w, l.ln1b.w, T, D)));
            if (save_cache) CUDA_CHECK(cudaMemcpy(ca.xn1, d_buf, T * D * sizeof(float), cudaMemcpyDeviceToDevice));

            mm_rm(d_buf, l.Wq.w, d_Q, T, D, tqk);
            mm_rm(d_buf, l.Wk.w, d_K, T, D, tqk);
            mm_rm(d_buf, l.Wv.w, d_V, T, D, tv);

            if (save_cache) {
                CUDA_CHECK(cudaMemcpy(ca.Q_pre_rope, d_Q, T * tqk * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(ca.K_pre_rope, d_K, T * tqk * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(ca.V, d_V, T * tv * sizeof(float), cudaMemcpyDeviceToDevice));
            }

            CUDA_LAUNCH((apply_rope_2d<<<(T * nh + 255) / 256, 256>>>(d_Q, T, nh, dqk, ROPE_BASE)));
            CUDA_LAUNCH((apply_rope_2d<<<(T * nh + 255) / 256, 256>>>(d_K, T, nh, dqk, ROPE_BASE)));

            if (save_cache) {
                CUDA_CHECK(cudaMemcpy(ca.Q, d_Q, T * tqk * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(ca.K, d_K, T * tqk * sizeof(float), cudaMemcpyDeviceToDevice));
            }

            int attn_blocks = T * nh;
            CUDA_LAUNCH((topk_causal_attn_fwd_parallel<<<attn_blocks, WARP_SIZE>>>(
                d_Q, d_K, d_V, d_at,
                d_topk_scores, d_topk_idx,
                T, nh, dqk, dv, Kt, sa
            )));

            if (save_cache) {
                CUDA_CHECK(cudaMemcpy(ca.attn_out, d_at, T * tv * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(ca.topk_scores, d_topk_scores, T * nh * Kt * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(ca.topk_idx, d_topk_idx, T * nh * Kt * sizeof(int), cudaMemcpyDeviceToDevice));
            }

            mm_rm(d_at, l.Wo.w, d_pr, T, tv, D);
            CUDA_LAUNCH((add_inplace<<<(T * D + 255) / 256, 256>>>(d_x, d_pr, T * D)));
            if (save_cache) CUDA_CHECK(cudaMemcpy(ca.x_mid, d_x, T * D * sizeof(float), cudaMemcpyDeviceToDevice));

            CUDA_LAUNCH((layer_norm_fwd<<<T,256>>>(d_buf2, d_x, l.ln2g.w, l.ln2b.w, T, D)));
            if (save_cache) CUDA_CHECK(cudaMemcpy(ca.xn2, d_buf2, T * D * sizeof(float), cudaMemcpyDeviceToDevice));

            mm_rm(d_buf2, l.W_gate.w, d_gate, T, D, FF);
            mm_rm(d_buf2, l.W_up.w,   d_up,   T, D, FF);

            if (save_cache) {
                CUDA_CHECK(cudaMemcpy(ca.gate_pre, d_gate, T * FF * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(ca.up_pre,   d_up,   T * FF * sizeof(float), cudaMemcpyDeviceToDevice));
            }

            CUDA_LAUNCH((swiglu_fwd<<<(T * FF + 255) / 256, 256>>>(d_swi, d_gate, d_up, T * FF)));
            if (save_cache) CUDA_CHECK(cudaMemcpy(ca.swiglu_out, d_swi, T * FF * sizeof(float), cudaMemcpyDeviceToDevice));

            mm_rm(d_swi, l.W_down.w, d_h2, T, FF, D);
            CUDA_LAUNCH((add_bias<<<(T * D + 255) / 256, 256>>>(d_h2, l.b_down.w, T, D)));
            CUDA_LAUNCH((add_inplace<<<(T * D + 255) / 256, 256>>>(d_x, d_h2, T * D)));
        }

        if (save_cache) CUDA_CHECK(cudaMemcpy(d_x_pre_final, d_x, T * D * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_LAUNCH((layer_norm_fwd<<<T,256>>>(d_buf, d_x, lnfg.w, lnfb.w, T, D)));
        if (save_cache) CUDA_CHECK(cudaMemcpy(d_x_final, d_buf, T * D * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_x, d_buf, T * D * sizeof(float), cudaMemcpyDeviceToDevice));

        mm_rm(d_x, lmh.w, d_logits, T, D, V);
    }

    // ============================================================
    //  BACKWARD
    // ============================================================
    void backward(const std::vector<int>& inp, float loss_scale) {
        int T = (int)inp.size();
        int D = cfg.d_model;
        int V = cfg.vocab_size;
        int FF = cfg.d_ff;
        int tqk = cfg.total_qk();
        int tv = cfg.total_v();
        int nh = cfg.n_heads;
        int dqk = cfg.d_qk;
        int dv = cfg.d_v;
        int Kt = cfg.top_k_attn;
        float sa = 1.0f / sqrtf((float)dqk);

        CUDA_LAUNCH((scale_kernel<<<(T * V + 255) / 256, 256>>>(d_logits, loss_scale, T * V)));

        // lmh.g += x_final^T * dlogits
        mm_rm_accum_dW(d_x_final, d_logits, lmh.g, T, D, V, 1.0f, 1.0f);

        CUDA_CHECK(cudaMemset(d_dx, 0, T * D * sizeof(float)));
        // d_dx += dlogits * lmh^T
        mm_rm_back_input(d_logits, lmh.w, d_dx, T, V, D, 1.0f, 1.0f);

        CUDA_CHECK(cudaMemset(d_tmp, 0, T * D * sizeof(float)));
        CUDA_LAUNCH((layer_norm_bwd<<<T,256>>>(d_tmp, lnfg.g, lnfb.g, d_dx, d_x_pre_final, lnfg.w, T, D)));
        CUDA_CHECK(cudaMemcpy(d_dx, d_tmp, T * D * sizeof(float), cudaMemcpyDeviceToDevice));

        for (int li = cfg.n_layers - 1; li >= 0; li--) {
            auto& l = layers[li];
            auto& ca = cache[li];

            // === FFN backward ===
            CUDA_LAUNCH((bias_bwd<<<(D + 255) / 256, 256>>>(l.b_down.g, d_dx, T, D)));
            mm_rm_accum_dW(ca.swiglu_out, d_dx, l.W_down.g, T, FF, D, 1.0f, 1.0f);

            CUDA_CHECK(cudaMemset(d_tmp, 0, T * FF * sizeof(float)));
            mm_rm_back_input(d_dx, l.W_down.w, d_tmp, T, D, FF, 1.0f, 1.0f);

            CUDA_CHECK(cudaMemset(d_dgate, 0, T * FF * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_dup,   0, T * FF * sizeof(float)));
            CUDA_LAUNCH((swiglu_bwd<<<(T * FF + 255) / 256, 256>>>(d_dgate, d_dup, d_tmp, ca.gate_pre, ca.up_pre, T * FF)));

            mm_rm_accum_dW(ca.xn2, d_dgate, l.W_gate.g, T, D, FF, 1.0f, 1.0f);
            mm_rm_accum_dW(ca.xn2, d_dup,   l.W_up.g,   T, D, FF, 1.0f, 1.0f);

            CUDA_CHECK(cudaMemset(d_tmp2, 0, T * D * sizeof(float)));
            mm_rm_back_input(d_dgate, l.W_gate.w, d_tmp2, T, FF, D, 1.0f, 1.0f);
            mm_rm_back_input(d_dup,   l.W_up.w,   d_tmp2, T, FF, D, 1.0f, 1.0f);

            CUDA_CHECK(cudaMemset(d_tmp, 0, T * D * sizeof(float)));
            CUDA_LAUNCH((layer_norm_bwd<<<T,256>>>(d_tmp, l.ln2g.g, l.ln2b.g, d_tmp2, ca.x_mid, l.ln2g.w, T, D)));

            CUDA_LAUNCH((add_inplace<<<(T * D + 255) / 256, 256>>>(d_tmp, d_dx, T * D)));
            CUDA_CHECK(cudaMemcpy(d_dx, d_tmp, T * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // === Attention backward ===
            mm_rm_accum_dW(ca.attn_out, d_dx, l.Wo.g, T, tv, D, 1.0f, 1.0f);

            CUDA_CHECK(cudaMemset(d_tmp2, 0, T * tv * sizeof(float)));
            mm_rm_back_input(d_dx, l.Wo.w, d_tmp2, T, D, tv, 1.0f, 1.0f);

            CUDA_CHECK(cudaMemset(d_dQ, 0, T * tqk * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_dK, 0, T * tqk * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_dV, 0, T * tv  * sizeof(float)));

            int attn_blocks = T * nh;
            CUDA_LAUNCH((topk_causal_attn_bwd_parallel<<<attn_blocks, WARP_SIZE>>>(
                d_tmp2, ca.topk_scores, ca.topk_idx,
                ca.Q, ca.K, ca.V,
                d_dQ, d_dK, d_dV,
                T, nh, dqk, dv, Kt, sa
            )));

            CUDA_LAUNCH((apply_rope_2d_bwd<<<(T * nh + 255) / 256, 256>>>(d_dQ, T, nh, dqk, ROPE_BASE)));
            CUDA_LAUNCH((apply_rope_2d_bwd<<<(T * nh + 255) / 256, 256>>>(d_dK, T, nh, dqk, ROPE_BASE)));

            mm_rm_accum_dW(ca.xn1, d_dQ, l.Wq.g, T, D, tqk, 1.0f, 1.0f);
            mm_rm_accum_dW(ca.xn1, d_dK, l.Wk.g, T, D, tqk, 1.0f, 1.0f);
            mm_rm_accum_dW(ca.xn1, d_dV, l.Wv.g, T, D, tv,  1.0f, 1.0f);

            CUDA_CHECK(cudaMemset(d_tmp2, 0, T * D * sizeof(float)));
            mm_rm_back_input(d_dQ, l.Wq.w, d_tmp2, T, tqk, D, 1.0f, 1.0f);
            mm_rm_back_input(d_dK, l.Wk.w, d_tmp2, T, tqk, D, 1.0f, 1.0f);
            mm_rm_back_input(d_dV, l.Wv.w, d_tmp2, T, tv,  D, 1.0f, 1.0f);

            CUDA_CHECK(cudaMemset(d_tmp, 0, T * D * sizeof(float)));
            CUDA_LAUNCH((layer_norm_bwd<<<T,256>>>(d_tmp, l.ln1g.g, l.ln1b.g, d_tmp2, ca.x_in, l.ln1g.w, T, D)));

            CUDA_LAUNCH((add_inplace<<<(T * D + 255) / 256, 256>>>(d_tmp, d_dx, T * D)));
            CUDA_CHECK(cudaMemcpy(d_dx, d_tmp, T * D * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        CUDA_CHECK(cudaMemcpy(d_tok, inp.data(), T * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_LAUNCH((embedding_bwd<<<T,256>>>(emb.g, pos_emb.g, d_dx, d_tok, T, D)));
    }

    // ============================================================
    //  TRAIN STEP
    // ============================================================
    float accumulate_gradients_masked(const std::vector<int>& tokens, int prompt_len, float batch_scale=1.0f) {
        int T = (int)tokens.size() - 1;
        if (T <= 0) return 0.0f;

        std::vector<int> inp(tokens.begin(), tokens.begin() + T);
        std::vector<int> tgt(tokens.begin() + 1, tokens.end());

        std::vector<int> mask(T, 0);
        int active_from = std::max(0, prompt_len - 1);
        for (int t = active_from; t < T; ++t) mask[t] = 1;

        int active = 0;
        for (int v : mask) active += v;
        if (active <= 0) return 0.0f;

        forward(inp, true);

        CUDA_CHECK(cudaMemcpy(d_tgt, tgt.data(), T * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), T * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_loss, 0, T * sizeof(float)));

        CUDA_LAUNCH((softmax_ce_masked_fwd_parallel<<<T,256>>>(d_logits, d_tgt, d_mask, d_loss, T, cfg.vocab_size)));

        std::vector<float> hl(T);
        CUDA_CHECK(cudaMemcpy(hl.data(), d_loss, T * sizeof(float), cudaMemcpyDeviceToHost));
        float loss = 0.0f;
        for (float v : hl) loss += v;
        loss /= active;

        backward(inp, (1.0f / active) * batch_scale);
        return loss;
    }

    void optimizer_step(float max_norm = 1.0f) {
        clip_gradients(max_norm);
        adam_all();
    }

    // ============================================================
    //  GENERATION
    // ============================================================
    std::vector<int> generate(
        std::vector<int> ctx, int maxn, float temp=0.7f, float top_p=0.85f,
        float repeat_penalty=1.10f,
        const std::vector<std::vector<int>>& stop_sequences = {},
        int eos_id = -1, int min_gen_tokens = 0
    ) {
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        int V = cfg.vocab_size;
        std::vector<float> hl(V);
        int generated = 0;

        for (int s = 0; s < maxn; s++) {
            std::vector<int> inp = ctx;
            if ((int)inp.size() > cfg.seq_len) {
                inp = std::vector<int>(inp.end() - cfg.seq_len, inp.end());
            }

            forward(inp, false);

            int T = (int)inp.size();
            CUDA_CHECK(cudaMemcpy(hl.data(), d_logits + (T - 1) * V, V * sizeof(float), cudaMemcpyDeviceToHost));

            int recent_n = std::min(48, (int)ctx.size());
            for (int i = (int)ctx.size() - recent_n; i < (int)ctx.size(); ++i) {
                int id = ctx[i];
                if (id < 0 || id >= V) continue;
                if (hl[id] > 0) hl[id] /= repeat_penalty;
                else hl[id] *= repeat_penalty;
            }

            if (eos_id >= 0 && eos_id < V && generated < min_gen_tokens) {
                hl[eos_id] = -1e30f;
            }

            float mx = *std::max_element(hl.begin(), hl.end());
            float sum = 0.0f;
            for (auto& v : hl) {
                v = expf((v - mx) / fmaxf(temp, 1e-4f));
                sum += v;
            }
            for (auto& v : hl) v /= sum;

            std::vector<std::pair<float,int>> pr(V);
            for (int i = 0; i < V; i++) pr[i] = {hl[i], i};
            std::sort(pr.begin(), pr.end(), std::greater<>());

            float cs = 0.0f;
            int cut = V;
            for (int i = 0; i < V; i++) {
                cs += pr[i].first;
                if (cs >= top_p) { cut = i + 1; break; }
            }

            float ss = 0.0f;
            for (int i = 0; i < cut; i++) ss += pr[i].first;

            float r = uni(rng) * ss;
            float acc = 0.0f;
            int ch = pr[0].second;
            for (int i = 0; i < cut; i++) {
                acc += pr[i].first;
                if (acc >= r) { ch = pr[i].second; break; }
            }

            ctx.push_back(ch);
            generated++;

            bool should_stop = false;
            for (const auto& seq : stop_sequences) {
                if (seq.empty() || seq.size() > ctx.size()) continue;
                bool match = true;
                size_t start = ctx.size() - seq.size();
                for (size_t i = 0; i < seq.size(); ++i) {
                    if (ctx[start + i] != seq[i]) { match = false; break; }
                }
                if (match) { should_stop = true; break; }
            }
            if (should_stop) break;
        }

        return ctx;
    }

    // ============================================================
    //  SAVE / LOAD
    // ============================================================
    void save(const std::string& path, const Tokenizer& tok) {
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            fprintf(stderr, "[Save] cannot open %s\n", path.c_str());
            return;
        }
        f.write((char*)&cfg, sizeof(Config));
        f.write((char*)&adam_t, sizeof(int));
        const_cast<Tokenizer&>(tok).save(f);

        for (auto* p : all_params()) {
            p->to_cpu();
            f.write((char*)p->hw, p->n * sizeof(float));

            std::vector<float> tmp(p->n);
            CUDA_CHECK(cudaMemcpy(tmp.data(), p->m, p->n * sizeof(float), cudaMemcpyDeviceToHost));
            f.write((char*)tmp.data(), p->n * sizeof(float));

            CUDA_CHECK(cudaMemcpy(tmp.data(), p->v, p->n * sizeof(float), cudaMemcpyDeviceToHost));
            f.write((char*)tmp.data(), p->n * sizeof(float));
        }
        printf("[Model DGA v3] Saved to %s\n", path.c_str());
    }

    bool load(const std::string& path, Tokenizer& tok) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        Config loaded_cfg;
        int loaded_adam_t = 0;

        f.read((char*)&loaded_cfg, sizeof(Config));
        f.read((char*)&loaded_adam_t, sizeof(int));
        tok.load(f);

        init(loaded_cfg);
        adam_t = loaded_adam_t;

        for (auto* p : all_params()) {
            f.read((char*)p->hw, p->n * sizeof(float));
            p->to_gpu();

            std::vector<float> tmp(p->n);
            f.read((char*)tmp.data(), p->n * sizeof(float));
            CUDA_CHECK(cudaMemcpy(p->m, tmp.data(), p->n * sizeof(float), cudaMemcpyHostToDevice));

            f.read((char*)tmp.data(), p->n * sizeof(float));
            CUDA_CHECK(cudaMemcpy(p->v, tmp.data(), p->n * sizeof(float), cudaMemcpyHostToDevice));
        }

        printf("[Model DGA v3] Loaded: vocab=%d d=%d heads=%d d_qk=%d d_v=%d\n",
               cfg.vocab_size, cfg.d_model, cfg.n_heads, cfg.d_qk, cfg.d_v);
        return true;
    }
};

// ============================================================
//  DATA / TRAINING HELPERS
// ============================================================
std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f), {}};
}

static inline bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static inline std::string trim_copy(std::string s) {
    auto is_ws = [](unsigned char c){ return c==' ' || c=='\t' || c=='\r' || c=='\n'; };
    while (!s.empty() && is_ws((unsigned char)s.back())) s.pop_back();
    size_t i = 0;
    while (i < s.size() && is_ws((unsigned char)s[i])) ++i;
    return s.substr(i);
}

struct DialoguePair { std::string user, bot; };
struct EncodedSample { std::vector<int> ids; int prompt_len = 0; };

std::vector<DialoguePair> extract_dialogue_pairs(const std::string& text) {
    std::istringstream iss(text);
    std::string line, pending_user;
    std::vector<DialoguePair> out;

    while (std::getline(iss, line)) {
        auto s = trim_copy(line);
        if (starts_with(s, "Человек:")) {
            pending_user = trim_copy(s.substr(std::string("Человек:").size()));
        } else if (!pending_user.empty() && starts_with(s, "Бот:")) {
            std::string bot = trim_copy(s.substr(std::string("Бот:").size()));
            if (!pending_user.empty() && !bot.empty()) out.push_back({pending_user, bot});
            pending_user.clear();
        } else if (!s.empty()) {
            pending_user.clear();
        }
    }
    return out;
}

std::string join_dialogue_pairs(const std::vector<DialoguePair>& pairs) {
    std::string joined;
    for (size_t i = 0; i < pairs.size(); ++i) {
        joined += "Человек: " + pairs[i].user + "\nБот: " + pairs[i].bot;
        if (i + 1 < pairs.size()) joined += "\n\n";
    }
    return joined;
}

std::string cut_at_stop_markers(std::string s) {
    std::vector<std::string> stops = {"\nЧеловек:", "\nБот:", "\n\n", "<eos>"};
    size_t cut = std::string::npos;
    for (const auto& stop : stops) {
        size_t pos = s.find(stop);
        if (pos != std::string::npos) cut = std::min(cut, pos);
    }
    if (cut != std::string::npos) s = s.substr(0, cut);
    return trim_copy(s);
}

std::string checkpoint_path_for_epoch(const std::string& mdl, int ep) {
    std::string path = mdl;
    size_t slash = path.find_last_of("/\\");
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
        return path + "_ep" + std::to_string(ep);
    return path.substr(0, dot) + "_ep" + std::to_string(ep) + path.substr(dot);
}

void run_eval_prompts(Model& model, const Tokenizer& tok) {
    std::vector<std::string> prompts = {
        "привет",
        "как перестать откладывать дела",
        "объясни рекурсию простыми словами",
        "помоги придумать идею трека",
        "мне тревожно",
        "как улучшить текст песни"
    };

    auto stop1 = tok.encode("\nЧеловек:");
    auto stop2 = tok.encode("\nБот:");
    auto stop3 = tok.encode("\n\n");
    std::vector<int> stop_eos = {tok.eos_id};
    std::vector<std::vector<int>> stop_seqs = {stop1, stop2, stop3, stop_eos};

    printf("  [Eval]\n");
    for (const auto& q : prompts) {
        auto ctx = tok.encode("Человек: " + q + "\nБот: ");
        auto out = model.generate(ctx, 48, 0.55f, 0.78f, 1.24f, stop_seqs, tok.eos_id, 6);
        std::vector<int> tail(out.begin() + ctx.size(), out.end());
        printf("    Q: %s\n    A: %s\n", q.c_str(), cut_at_stop_markers(tok.decode(tail)).c_str());
    }
    printf("\n");
}

// ============================================================
//  TRAIN
// ============================================================
void train(const std::string& data, const std::string& mdl, int epochs) {
    std::string raw = read_file(data);
    if (raw.empty()) {
        fprintf(stderr, "File not found or empty: %s\n", data.c_str());
        return;
    }
    printf("[Data] Raw: %zu bytes\n", raw.size());

    auto pairs = extract_dialogue_pairs(raw);
    if (pairs.empty()) {
        fprintf(stderr, "[Data] No dialogues found\n");
        return;
    }

    std::string text = join_dialogue_pairs(pairs) + "\n" + Tokenizer::EOS_STR;
    printf("[Data] Dialogues: %zu bytes, %zu pairs\n", text.size(), pairs.size());

    Tokenizer tok;
    tok.build(text, 2048);

    auto corpus_ids = tok.encode(text);
    printf("[Data] BPE tokens: %zu\n", corpus_ids.size());

    Config cfg;
    cfg.vocab_size = tok.vocab_size();

    Model model;
    model.init(cfg);

    const int S = cfg.seq_len;
    std::vector<EncodedSample> samples;
    samples.reserve(pairs.size());

    int skipped_long = 0;
    for (const auto& p : pairs) {
        std::string prompt = "Человек: " + p.user + "\nБот: ";
        std::string full = prompt + p.bot + Tokenizer::EOS_STR;

        auto prompt_ids = tok.encode(prompt);
        auto full_ids = tok.encode(full);

        if ((int)prompt_ids.size() < 1 || (int)full_ids.size() < 2) continue;
        if ((int)prompt_ids.size() >= S) { skipped_long++; continue; }

        if ((int)full_ids.size() > S + 1) full_ids.resize(S + 1);
        if ((int)full_ids.size() <= (int)prompt_ids.size()) continue;

        if (full_ids.back() != tok.eos_id && (int)full_ids.size() < S + 1) full_ids.push_back(tok.eos_id);

        samples.push_back({std::move(full_ids), (int)prompt_ids.size()});
    }

    if (samples.empty()) {
        fprintf(stderr, "[Data] No valid samples\n");
        return;
    }

    printf("[Data] Training samples: %zu (skipped long: %d)\n", samples.size(), skipped_long);

    std::mt19937 rng(42);
    std::vector<int> order(samples.size());
    std::iota(order.begin(), order.end(), 0);
    int steps = (int)samples.size();

    auto stop1 = tok.encode("\nЧеловек:");
    auto stop2 = tok.encode("\nБот:");
    auto stop3 = tok.encode("\n\n");
    std::vector<int> stop_eos = {tok.eos_id};
    std::vector<std::vector<int>> stop_seqs = {stop1, stop2, stop3, stop_eos};

    for (int ep = 0; ep < epochs; ep++) {
        std::shuffle(order.begin(), order.end(), rng);
        float tot = 0.0f;
        int accum = 0;
        model.zero_grad();

        auto t0 = std::chrono::steady_clock::now();

        for (int st = 0; st < steps; st++) {
            const auto& sm = samples[order[st]];
            tot += model.accumulate_gradients_masked(sm.ids, sm.prompt_len,
                                                     1.0f / std::max(1, cfg.micro_batch));
            accum++;

            if (accum >= cfg.micro_batch || st + 1 == steps) {
                model.optimizer_step(1.0f);
                model.zero_grad();
                accum = 0;
            }

            if ((st + 1) % 200 == 0 || st + 1 == steps) {
                CUDA_CHECK(cudaDeviceSynchronize());
                auto t1 = std::chrono::steady_clock::now();
                int ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                printf("  Epoch %d/%d step %d/%d loss=%.4f time=%dms\n",
                       ep + 1, epochs, st + 1, steps, tot / (st + 1), ms);
                t0 = t1;
            }
        }

        model.save(mdl, tok);
        std::string ckpt = checkpoint_path_for_epoch(mdl, ep + 1);
        model.save(ckpt, tok);
        printf("[Model DGA v3] Epoch checkpoint: %s\n", ckpt.c_str());

        auto si = tok.encode("Человек: привет\nБот: ");
        auto gi = model.generate(si, 40, 0.55f, 0.78f, 1.24f, stop_seqs, tok.eos_id, 6);
        std::vector<int> tail(gi.begin() + si.size(), gi.end());
        printf("\n  [Gen] Q: привет\n  A: %s\n\n", cut_at_stop_markers(tok.decode(tail)).c_str());

        run_eval_prompts(model, tok);
    }
}

// ============================================================
//  CHAT
// ============================================================
void chat(const std::string& mdl) {
    Tokenizer tok;
    Model model;

    if (!model.load(mdl, tok)) {
        fprintf(stderr, "No model: %s\n", mdl.c_str());
        return;
    }

    auto stop1 = tok.encode("\nЧеловек:");
    auto stop2 = tok.encode("\nБот:");
    auto stop3 = tok.encode("\n\n");
    std::vector<int> stop_eos = {tok.eos_id};
    std::vector<std::vector<int>> stop_seqs = {stop1, stop2, stop3, stop_eos};

    printf("\n+----------------------------------------------+\n");
    printf("| DGA Transformer v3 — Safer Revised           |\n");
    printf("| 2D Q/K + Rich V + Warp Top-K + SwiGLU + RoPE |\n");
    printf("| /exit /reset /temp X /maxlen X               |\n");
    printf("+----------------------------------------------+\n\n");

    std::string ctx;
    float temp = 0.58f;
    int maxl = 56;

    while (true) {
        printf("You: ");
        std::string line;
        if (!std::getline(std::cin, line)) break;
        if (line == "/exit") break;
        if (line == "/reset") { ctx.clear(); printf("[reset]\n"); continue; }
        if (line.rfind("/temp ", 0) == 0) { temp = std::stof(line.substr(6)); continue; }
        if (line.rfind("/maxlen ", 0) == 0) { maxl = std::stoi(line.substr(8)); continue; }

        if (!ctx.empty()) ctx += "\n";
        ctx += "Человек: " + line + "\nБот: ";

        auto ci = tok.encode(ctx);
        if ((int)ci.size() > model.cfg.seq_len - 24) {
            ci = std::vector<int>(ci.end() - (model.cfg.seq_len - 24), ci.end());
        }

        auto gi = model.generate(ci, maxl, temp, 0.78f, 1.24f, stop_seqs, tok.eos_id, 6);
        std::vector<int> ni(gi.begin() + ci.size(), gi.end());
        std::string rep = cut_at_stop_markers(tok.decode(ni));

        printf("Bot: %s\n\n", rep.c_str());
        ctx += rep;
    }
}

// ============================================================
//  MAIN
// ============================================================
int main(int argc, char* argv[]) {
    SetConsoleOutputCP(65001);
    system("chcp 65001 > nul");

    if (argc < 2) {
        printf("  %s train data.txt model.bin [epochs=10]\n", argv[0]);
        printf("  %s chat  model.bin\n", argv[0]);
        return 1;
    }

    CUBLAS_CHECK(cublasCreate(&cublas));

    std::string mode = argv[1];

    if (mode == "train") {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s train data.txt model.bin [epochs=10]\n", argv[0]);
            cublasDestroy(cublas);
            return 1;
        }
        train(argv[2], argv[3], (argc >= 5) ? std::stoi(argv[4]) : 10);
    } else if (mode == "chat") {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s chat model.bin\n", argv[0]);
            cublasDestroy(cublas);
            return 1;
        }
        chat(argv[2]);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode.c_str());
    }

    cublasDestroy(cublas);
    return 0;
}
