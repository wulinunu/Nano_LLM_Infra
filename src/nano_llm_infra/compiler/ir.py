import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

class IRNode:
    """
    我们自定义的计算图节点表示，屏蔽了底层 torch.fx 的复杂性。
    """
    def __init__(self, name, op_type, target, inputs, shape=None, dtype=None):
        self.name = name          # 节点名称，比如 'add'
        self.op_type = op_type    # 节点类型: 'placeholder' (输入), 'call_function', 'call_module', 'output' (输出)
        self.target = target      # 调用的具体函数或模块名
        self.inputs = inputs      # 依赖的输入节点名称列表 (List[str]) - 【入边】
        self.shape = shape        # 输出 Tensor 的形状
        self.dtype = dtype        # 输出 Tensor 的数据类型
        self.users = []           # 依赖当前节点的下游节点名称列表 (List[str]) - 【出边】

    def __repr__(self):
        return f"IRNode(name='{self.name}', op='{self.op_type}', target='{self.target}', shape={self.shape}, inputs={self.inputs})"


class GraphIR:
    """
    统一的图表示，包含图中所有的节点。
    """
    def __init__(self):
        self.nodes = {}  # Dict[str, IRNode] 存储节点
        self.inputs = [] # 输入节点名称列表
        self.outputs = [] # 输出节点名称列表

    def add_node(self, node: IRNode):
        self.nodes[node.name] = node
        if node.op_type == 'placeholder':
            self.inputs.append(node.name)
        elif node.op_type == 'output':
            self.outputs.append(node.name)
            
        # 维护双向图拓扑：把当前节点添加到它所有 inputs 节点的 users 列表里
        for inp in node.inputs:
            if inp in self.nodes:
                self.nodes[inp].users.append(node.name)

    def print_graph(self):
        print("=== Custom Graph IR ===")
        for name, node in self.nodes.items():
            print(node)
        print("=======================\n")


class GraphCapturer:
    """
    前端捕获器：使用 torch.fx 抓取 PyTorch 模型并转换为我们的 GraphIR
    """
    def __init__(self, model):
        self.model = model

    def capture(self, *dummy_inputs) -> GraphIR:
        # 1. Symbolic Trace 捕获原始图
        tracer = torch.fx.Tracer()
        fx_graph = tracer.trace(self.model)
        gm = torch.fx.GraphModule(self.model, fx_graph)

        # 第二步：Shape 传播（为了拿到推导后的 Tensor Shape 和 Dtype）
        # dummy_inputs 必须和 fx 期望的名字对齐或者用 tuple 传入
        ShapeProp(gm).propagate(*dummy_inputs)

        # 3. 转换为我们自定义的 GraphIR，斩断与pytorch原生库的强耦合
        graph_ir = GraphIR()
        for node in gm.graph.nodes:
            # 提取依赖的输入节点名
            inputs = [n.name for n in node.args if isinstance(n, torch.fx.Node)]
            
            # 提取 Tensor 元数据 (Shape/Dtype)
            shape, dtype = None, None
            if 'tensor_meta' in node.meta:
                meta = node.meta['tensor_meta']
                # 处理有多个输出的情况，为了简单这里只取第一个
                if not hasattr(meta, 'shape') and isinstance(meta, tuple):
                    meta = meta[0]
                if hasattr(meta, 'shape'): 
                    shape = tuple(meta.shape)
                if hasattr(meta, 'dtype'): 
                    dtype = meta.dtype

            # 构建自定义节点
            ir_node = IRNode(
                name=node.name,
                op_type=node.op,
                target=str(node.target),
                inputs=inputs,
                shape=shape,
                dtype=dtype
            )
            graph_ir.add_node(ir_node)
            
        return graph_ir
