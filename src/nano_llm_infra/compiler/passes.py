from .ir import GraphIR, IRNode

class PassManager:
    """Pass管理器：按顺序在图上应用一系列变换"""
    def __init__(self):
        self.passes = []
        
    def add_pass(self, p):
        self.passes.append(p)
        
    def run(self, graph: GraphIR) -> GraphIR:
        for p in self.passes:
            graph = p.apply(graph)
        return graph

class FusionPass:
    """
    算子融合 (Pattern Matching & Rewriting)
    使用【真正的图拓扑数据流】匹配：pow -> mean -> add -> rsqrt -> mul -> mul
    """
    def apply(self, graph: GraphIR) -> GraphIR:
        new_graph = GraphIR()
        skip_names = set()
        
        # 遍历所有节点，作为模式匹配的起点
        for node_name, node in graph.nodes.items():
            if node_name in skip_names:
                continue
                
            # 【起点】：如果遇到 pow，我们开始顺着数据流（users）往下摸瓜
            if 'pow' in str(node.target):
                # 尝试抓取这条数据流链上的 6 个节点
                try:
                    n1 = node
                    
                    # 沿着出边 (users) 找 mean
                    if len(n1.users) != 1: raise ValueError
                    n2 = graph.nodes[n1.users[0]]
                    if 'mean' not in str(n2.target): raise ValueError
                    
                    # 沿着出边找 add
                    if len(n2.users) != 1: raise ValueError
                    n3 = graph.nodes[n2.users[0]]
                    if 'add' not in str(n3.target): raise ValueError
                    
                    # 沿着出边找 rsqrt
                    if len(n3.users) != 1: raise ValueError
                    n4 = graph.nodes[n3.users[0]]
                    if 'rsqrt' not in str(n4.target): raise ValueError
                    
                    # 沿着出边找第一次 mul
                    if len(n4.users) != 1: raise ValueError
                    n5 = graph.nodes[n4.users[0]]
                    if 'mul' not in str(n5.target): raise ValueError
                    
                    # 沿着出边找第二次 mul (乘 weight)
                    if len(n5.users) != 1: raise ValueError
                    n6 = graph.nodes[n5.users[0]]
                    if 'mul' not in str(n6.target): raise ValueError
                    
                    # ======== 【命中拓扑匹配！】 ========
                    # 走到这里，说明无论这 6 个节点在执行顺序里隔了多远，
                    # 它们在数据流上是严丝合缝串在一起的！
                    fused_name = f"fused_rmsnorm_{node_name}"
                    
                    # 寻找融合算子的输入：
                    # RMSNorm 的第一个输入必须是最初那个 pow(x) 的输入 x
                    x_input = n1.inputs[0]
                    # RMSNorm 的第二个输入是最后那步乘法的另一个参数 (weight)
                    weight_input = [inp for inp in n6.inputs if inp != n5.name][0]
                    
                    fused_node = IRNode(
                        name=fused_name,
                        op_type="call_fused",
                        target="fused_rmsnorm",
                        inputs=[x_input, weight_input],
                        shape=n6.shape,
                        dtype=n6.dtype
                    )
                    new_graph.add_node(fused_node)
                    
                    # 将全图中原本吃 n6 结果的人，改为吃 fused_name
                    for later_node in graph.nodes.values():
                        later_node.inputs = [fused_name if inp == n6.name else inp for inp in later_node.inputs]
                        
                    # 记录这 6 个节点已经被“吃”掉了，不再加入新图
                    skip_names.update([n.name for n in [n1, n2, n3, n4, n5, n6]])
                    continue
                    
                except ValueError:
                    # 如果中间发现链条断了，或者某个人不止一个下游，说明不是标准的 RMSNorm 模式，放弃匹配
                    pass
                    
            # 如果没被融合（被跳过），就原样加入新图
            new_graph.add_node(node)
            
        return new_graph

class MemoryPlanningPass:
    """
    静态内存规划: 生命期分析 (Liveness Analysis)
    通过分析每个 tensor 最后一次被使用的位置，决定计算结束后是否可以立即复用 Buffer。
    """
    def apply(self, graph: GraphIR) -> GraphIR:
        # 1. 记录每个张量最后被作为输入的节点名称
        last_uses = {}
        for node in graph.nodes.values():
            for inp in node.inputs:
                last_uses[inp] = node.name
                
        # 2. 为每个节点附加 liveness 属性
        for node in graph.nodes.values():
            node.last_use = last_uses.get(node.name, node.name) # 如果用完没人接手，寿命到此为止
            
        return graph
