import torch
import triton
import triton.language as tl
@triton.jit
def _paged_attention_decode_kernel(
    Q, K_cache, V_cache, Block_Tables, Context_Lens, Out,
    stride_q_bs, stride_q_h, stride_q_d,
    stride_kc_b, stride_kc_h, stride_kc_s, stride_kc_d,
    stride_vc_b, stride_vc_h, stride_vc_s, stride_vc_d,
    stride_bt_b, stride_bt_s,
    stride_out_bs, stride_out_h, stride_out_d,
    sm_scale, # 预先在CPU算好的缩放因子
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, # 通常是 2^n >= HEAD_DIM
    ):
    """
    极简 Triton 版 Paged Attention Kernel (仅用于 Decode 阶段)。
    """
    # 线程格: 每个头对应一个 program
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    # 1. 获取当前请求的上下文长度  
    context_len = tl.load(Context_Lens + batch_idx)  

    # 2. 定位到当前 Batch 和 Head 的 Q  
    q_ptr = Q + batch_idx * stride_q_bs + head_idx * stride_q_h  
    # 获取 Q 的向量: [HEAD_DIM]  
    offs_d = tl.arange(0, BLOCK_DMODEL)  
    mask_d = offs_d < HEAD_DIM  
    q = tl.load(q_ptr + offs_d * stride_q_d, mask=mask_d, other=0.0)  

    # 用来累加分母和分子  
    m_i = -float('inf')  # 当前最大得分  
    l_i = 0.0            # 当前 softmax 概率的分母  
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)  

    # 3. 遍历所有的 Logical Block  
    num_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE  
    
    for logical_block_idx in range(num_blocks):  
        # 通过 Block Table 获取 Physical Block ID  
        bt_ptr = Block_Tables + batch_idx * stride_bt_b + logical_block_idx * stride_bt_s  
        physical_block_idx = tl.load(bt_ptr)  

        # 当前块里实际有多少个 token (如果是最后一个块可能不满)  
        start_token_idx = logical_block_idx * BLOCK_SIZE  
        tokens_in_this_block = tl.minimum(BLOCK_SIZE, context_len - start_token_idx)  

        # 加载 K 缓存 [BLOCK_SIZE, HEAD_DIM]  
        offs_s = tl.arange(0, BLOCK_SIZE)  
        k_ptrs = K_cache + physical_block_idx * stride_kc_b + head_idx * stride_kc_h + (offs_s[:, None] * stride_kc_s + offs_d[None, :] * stride_kc_d)  
        
        mask_k = (offs_s[:, None] < tokens_in_this_block) & (offs_d[None, :] < HEAD_DIM)  
        k = tl.load(k_ptrs, mask=mask_k, other=0.0)  

        # 计算 Q * K^T  
        # q [BLOCK_DMODEL] => [1, BLOCK_DMODEL], k [BLOCK_SIZE, BLOCK_DMODEL]  
        # dot 结果 shape: [BLOCK_SIZE]  
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale  

        # 屏蔽无效的 token  
        qk = tl.where(offs_s < tokens_in_this_block, qk, -float('inf'))  

        # Online Softmax 逻辑  
        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))  
        p = tl.exp(qk - m_ij)  
        
        # 修正过去的指数  
        alpha = tl.exp(m_i - m_ij)  
        l_i = l_i * alpha + tl.sum(p, axis=0)  

        # 加载 V 缓存  
        v_ptrs = V_cache + physical_block_idx * stride_vc_b + head_idx * stride_vc_h + (offs_s[:, None] * stride_vc_s + offs_d[None, :] * stride_vc_d)  
        v = tl.load(v_ptrs, mask=mask_k, other=0.0)  

        # p: [BLOCK_SIZE], v: [BLOCK_SIZE, BLOCK_DMODEL]  
        # acc = acc * alpha + sum(p * v)  
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)  

        m_i = m_ij  

    # 4. 归一化并写入结果  
    acc = acc / l_i  
    out_ptr = Out + batch_idx * stride_out_bs + head_idx * stride_out_h + offs_d * stride_out_d  
    tl.store(out_ptr, acc, mask=mask_d)  

def paged_attention_triton(q, k_cache, v_cache, block_tables, context_lens, block_size):
    """
    对外的 Python 接口
    """
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    out = torch.empty_like(q)  

    # 确定 Triton 块大小 (需要是2的幂)  
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)  

    # 提前在 CPU 算好缩放因子  
    import math  
    sm_scale = 1.0 / math.sqrt(head_dim)  

    grid = (batch_size, num_heads)  
    
    _paged_attention_decode_kernel[grid](  
        q, k_cache, v_cache, block_tables, context_lens, out,  
        q.stride(0), q.stride(1), q.stride(2),  
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),  
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),  
        block_tables.stride(0), block_tables.stride(1),  
        out.stride(0), out.stride(1), out.stride(2),  
        sm_scale, # 传入预先计算好的 sm_scale  
        BLOCK_SIZE=block_size,  
        HEAD_DIM=head_dim,  
        BLOCK_DMODEL=BLOCK_DMODEL,  
    )  

    return out  

