class TritonCodegen:
    """
    基于 Kernel IR 动态拼装 Python 源码，以及上层的 Dispatch 调度函数。
    现在的 Codegen 不再是死板的字符串替换，而是一个【代码组装机】。
    """
    def generate(self, kernel_irs):
        header_code = """
import torch
import triton
import triton.language as tl

# ==========================================
# 自动生成的 Triton Kernels (基于微指令组装)
# ==========================================
"""
        code = [header_code]
        
        # 1. 组装 Kernel (Printer)
        for kernel in kernel_irs:
            code.append(self._print_kernel(kernel))
                
        # 2. 生成 Dispatcher
        dispatcher_code = """
# ==========================================
# 运行时 Dispatcher (含静态内存复用)
# ==========================================
def dispatch_and_run(x, residual, weight, linear_weight, linear_bias=None):
    N = x.shape[-1]
    B = x.shape[0]
    BLOCK_SIZE = 128
    grid = lambda meta: (B, )

    # In-place 内存复用逻辑保持不变
    out_add = residual  # <--- In-place 复用
    out_norm = out_add  # <--- In-place 复用
"""
        code.append(dispatcher_code)
        
        # 动态发射 Kernel
        for kernel in kernel_irs:
            # 根据 name 推导 Kernel 名称，例如 add -> add_kernel
            k_name = kernel.name if "fused" in kernel.name else f"{kernel.name}_kernel"
            
            # 传参映射 (Demo 中简单写死映射关系，真实编译器有一个 Variable Tracker)
            if 'add' in kernel.name and 'fused' not in kernel.name:
                args = "x, residual, out_add, N, BLOCK_SIZE=BLOCK_SIZE"
            elif 'fused_rmsnorm' in kernel.name:
                args = "out_add, weight, out_norm, N, BLOCK_SIZE=BLOCK_SIZE"
            else:
                args = ""
                
            code.append(f"    {k_name}[grid]({args})")
            code.append("")
        
        # Linear 回退逻辑
        fallback_code = """
    # 3. Linear / MatMul (回退到 Torch 实现)
    out_final = torch.matmul(out_norm, linear_weight.t())
    if linear_bias is not None:
        out_final += linear_bias

    return out_final
"""
        code.append(fallback_code)
        
        return "\n".join(code)

    def _print_kernel(self, kernel):
        """
        核心 Printer：把 Loads, Computes, Stores 三段指令无缝拼凑成合法的 Triton 函数。
        """
        lines = []
        lines.append("@triton.jit")
        
        # 函数签名
        k_name = kernel.name if "fused" in kernel.name else f"{kernel.name}_kernel"
        sig = ", ".join(kernel.signature + ["BLOCK_SIZE: tl.constexpr"])
        lines.append(f"def {k_name}({sig}):")
        
        # 基础寻址 boilerplate
        boilerplate = """    
    row_idx = tl.program_id(0)
    row_start_ptr = row_idx * n_elements
    offsets = row_start_ptr + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_elements
"""
        lines.append(boilerplate)
        
        # 组装 1: 访存读入
        for load_ins in kernel.loads:
            lines.append(f"    {load_ins}")
        lines.append("")
        
        # 组装 2: 高速运算
        for comp_ins in kernel.computes:
            lines.append(f"    {comp_ins}")
        lines.append("")
        
        # 组装 3: 访存写回
        for store_ins in kernel.stores:
            lines.append(f"    {store_ins}")
        lines.append("")
        
        return "\n".join(lines)
