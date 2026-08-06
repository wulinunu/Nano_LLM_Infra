from .ir import GraphIR

class KernelIR:
    """
    真正的底层 Kernel 中间表示。
    工业界（如 TorchInductor）会将操作严格划分为微指令：
    1. 访存 (Loads)
    2. 计算 (Computes)
    3. 写回 (Stores)
    """
    def __init__(self, name, grid, block_size):
        self.name = name
        self.grid = grid
        self.block_size = block_size
        self.signature = []  # Kernel 形参列表
        self.loads = []      # HBM -> SRAM 的代码指令
        self.computes = []   # SRAM 内部运算指令
        self.stores = []     # SRAM -> HBM 的代码指令

    def __repr__(self):
        return f"KernelIR(name='{self.name}', loads={len(self.loads)}, computes={len(self.computes)}, stores={len(self.stores)})"


class LoweringPass:
    """
    将高层的算子节点映射为微指令级 (Instruction-level) 的 Kernel IR。
    """
    def apply(self, graph: GraphIR):
        kernels = {}
        for node in graph.nodes.values():
            # 过滤掉非计算节点
            if node.op_type in ['placeholder', 'get_attr', 'output'] or 'linear' in node.target:
                continue 
                
            # 确定并发规模
            feature_dim = node.shape[-1] if node.shape else 128
            block_size = 128
            grid = f"({feature_dim} + {block_size} - 1) // {block_size}"
            
            kernel = KernelIR(node.name, grid, block_size)
            
            # --- 细粒度指令映射 (Micro-op Instruction Selection) ---
            if 'add' in node.target and 'fused' not in node.target:
                kernel.signature = ["x_ptr", "y_ptr", "out_ptr", "n_elements"]
                kernel.loads = [
                    "x = tl.load(x_ptr + offsets, mask=mask)",
                    "y = tl.load(y_ptr + offsets, mask=mask)"
                ]
                kernel.computes = ["tmp0 = x + y"]
                kernel.stores = ["tl.store(out_ptr + offsets, tmp0, mask=mask)"]
                
            elif 'fused_rmsnorm' in node.target:
                kernel.signature = ["x_ptr", "weight_ptr", "out_ptr", "n_elements"]
                kernel.loads = [
                    "x = tl.load(x_ptr + offsets, mask=mask)",
                    "weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask)"
                ]
                kernel.computes = [
                    "x_f32 = x.to(tl.float32)",
                    "variance = tl.sum(x_f32 * x_f32, axis=0) / n_elements",
                    "rsqrt = tl.math.rsqrt(variance + 1e-6)",
                    "tmp0 = x * rsqrt * weight"
                ]
                kernel.stores = ["tl.store(out_ptr + offsets, tmp0, mask=mask)"]
                
            kernels[node.name] = kernel
            
        return list(kernels.values())
