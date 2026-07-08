import torch
import math

def paged_attention_ref(
    q: torch.Tensor,                # [1, num_heads, head_dim]
    k_cache: torch.Tensor,          # [num_blocks, num_heads, block_size, head_dim]
    v_cache: torch.Tensor,          # [num_blocks, num_heads, block_size, head_dim]
    block_tables: torch.Tensor,     # [1, max_num_blocks_per_seq]
    context_lens: torch.Tensor,     # [1]
    block_size: int,
    ):
    """
    纯 PyTorch 版本的 PagedAttention 参考实现。
    由于是参考实现，我们使用高级索引（index_select）将不连续的块拼凑成连续张量，再调用标准 Attention。
    """
    # 假设 batch_size = 1 (decode 阶段，单 token)
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    seq_len = context_lens[0].item()

    # 1. 找到该请求占用的物理块 ID  
    num_blocks_needed = (seq_len + block_size - 1) // block_size  
    physical_block_ids = block_tables[0, :num_blocks_needed] # [num_blocks_needed]  
    
    # 2. 从 K/V Cache 池中抓取对应物理块的数据  
    # gathered_k: [num_blocks_needed, num_heads, block_size, head_dim]  
    gathered_k = k_cache.index_select(0, physical_block_ids)  
    gathered_v = v_cache.index_select(0, physical_block_ids)  
    
    # 3. 拼凑连续：先转成 [num_heads, num_blocks_needed * block_size, head_dim]  
    gathered_k = gathered_k.transpose(0, 1).reshape(num_heads, -1, head_dim)  
    gathered_v = gathered_v.transpose(0, 1).reshape(num_heads, -1, head_dim)  
    
    # 4. 裁剪掉后面未使用的虚假 token 空间  
    # 真实有用的 K, V 形状: [num_heads, seq_len, head_dim]  
    k_real = gathered_k[:, :seq_len, :]  
    v_real = gathered_v[:, :seq_len, :]  
    
    # 5. 标准的 Scaled Dot-Product Attention  
    q_reshaped = q[0] # [num_heads, head_dim]  
    q_reshaped = q_reshaped.unsqueeze(1) # [num_heads, 1, head_dim]  
    
    # [num_heads, 1, seq_len]  
    scores = torch.matmul(q_reshaped, k_real.transpose(-2, -1)) / math.sqrt(head_dim)  
    probs = torch.softmax(scores, dim=-1)  
    
    # [num_heads, 1, head_dim]  
    out = torch.matmul(probs, v_real)  
    
    # 恢复形状 [1, num_heads, head_dim]  
    return out.squeeze(1).unsqueeze(0)
