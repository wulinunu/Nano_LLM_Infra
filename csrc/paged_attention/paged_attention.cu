#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void paged_attention_decode_kernel(
    const scalar_t* __restrict__ q,             // [num_seqs, num_heads, head_dim]
    const scalar_t* __restrict__ k_cache,       // [num_blocks, num_heads, block_size, head_dim]
    const scalar_t* __restrict__ v_cache,       // [num_blocks, num_heads, block_size, head_dim]
    const int* __restrict__ block_tables,       // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,       // [num_seqs]
    scalar_t* __restrict__ out,                 // [num_seqs, num_heads, head_dim]
    const int num_seqs,
    const int num_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq) {
    
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (seq_idx >= num_seqs || head_idx >= num_heads) return;

    int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    float q_val = 0.0f;
    if (tid < head_dim) {
        int q_offset = seq_idx * (num_heads * head_dim) + head_idx * head_dim + tid;
        q_val = static_cast<float>(q[q_offset]);
    }

    float m_i = -1e20f;
    float l_i = 0.0f;
    float acc = 0.0f;
    float scale = 1.0f / sqrtf((float)head_dim);

    __shared__ float s_dot;
    __shared__ float s_reduce[32]; //一个block最多只能包含1024个线程

    int num_logical_blocks = (context_len + block_size - 1) / block_size;

    for (int logical_block = 0; logical_block < num_logical_blocks; ++logical_block) {
        int phys_block = block_tables[seq_idx * max_blocks_per_seq + logical_block];
        
        int start_t = logical_block * block_size;
        int tokens_in_block = min(block_size, context_len - start_t); //最后一个没装满的装了多少
        
        for (int i = 0; i < tokens_in_block; ++i) {
            float k_val = 0.0f;
            if (tid < head_dim) {
                int k_offset = phys_block * (num_heads * block_size * head_dim) +
                               head_idx * (block_size * head_dim) +
                               i * head_dim +
                               tid;
                k_val = static_cast<float>(k_cache[k_offset]);
            }
            
            float qk = q_val * k_val * scale;
            
            // Block reduce
            int lane = tid % 32; // 我是小队(Warp)里的第几号？(0-31)
            int wid = tid / 32;  // 我属于第几个小队(Warp)？
            float val = qk;
            #pragma unroll // 展开循环，减少指令
            for (int offset = 16; offset > 0; offset /= 2)
                val += __shfl_down_sync(0xffffffff, val, offset); // 越界的会变成undefined
            // __shfl_down_sync(mask, val, offset)
            // mask: 哪些线程有效
            // val: 当前线程寄存器值
            // offset: 向下读取几个 lane
            
            if (lane == 0) s_reduce[wid] = val;
            __syncthreads(); //等所有人写完
            
            val = (tid < (blockDim.x + 31) / 32) ? s_reduce[lane] : 0.0f;
            //SIMT 的铁律：同生共死 在一个 Warp (32个线程) 里，只要有 1 个线程在跑这句加法，其他 31 个线程也必须在这一刻执行相同的指令。哪怕它加的是 0，硬件底层也是同步发射指令的。你让它加 0 也是加一次，你想办法用 if 让它不加，反而会破坏 Warp 的同步性，导致严重的性能拖慢（Warp Divergence）。
            if (wid == 0) {
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2)
                    val += __shfl_down_sync(0xffffffff, val, offset);
                if (tid == 0) s_dot = val;
            }
            __syncthreads();
            
            float dot = s_dot;
            
            float m_ij = max(m_i, dot);
            float p = expf(dot - m_ij);
            float alpha = expf(m_i - m_ij); //onlinesoftmax
            
            l_i = l_i * alpha + p;
            
            float v_val = 0.0f;
            if (tid < head_dim) {
                int v_offset = phys_block * (num_heads * block_size * head_dim) +
                               head_idx * (block_size * head_dim) +
                               i * head_dim +
                               tid;
                v_val = static_cast<float>(v_cache[v_offset]);
            }
            
            acc = acc * alpha + p * v_val;
            
            m_i = m_ij;
        }
    }

    if (tid < head_dim) {
        int out_offset = seq_idx * (num_heads * head_dim) + head_idx * head_dim + tid;
        out[out_offset] = static_cast<scalar_t>(acc / l_i);
    }
}

torch::Tensor paged_attention_cuda(
    torch::Tensor q,             // [num_seqs, num_heads, head_dim]
    torch::Tensor k_cache,       // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor v_cache,       // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor block_tables,  // [num_seqs, max_blocks_per_seq]
    torch::Tensor context_lens,  // [num_seqs]
    int block_size) {
    
    auto num_seqs = q.size(0);
    auto num_heads = q.size(1);
    auto head_dim = q.size(2);
    auto num_blocks = k_cache.size(0);
    auto max_blocks_per_seq = block_tables.size(1);

    auto out = torch::empty_like(q);

    dim3 grid(num_heads, num_seqs);
    // Round block size up to nearest 32 (warp size)
    int threads = ((head_dim + 31) / 32) * 32;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "paged_attention_decode_kernel", [&] {
        paged_attention_decode_kernel<scalar_t><<<grid, threads>>>(
            q.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            out.data_ptr<scalar_t>(),
            num_seqs,
            num_heads,
            head_dim,
            block_size,
            max_blocks_per_seq
        );
    });

    return out;
}

// 绑定到 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_cuda", &paged_attention_cuda, "Paged Attention Decode (CUDA)");
}
