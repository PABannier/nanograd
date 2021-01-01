from graphviz import Digraph


class CompGraphVisualizer:
    r"""
        CompGraphVisualizer is the building block object for quick 
        visualization of a computational graph. Nanograd dynamically 
        builds a computational graph and keeps track of the operations 
        made while building the computational graph.

        Performing DFS over the computational graph, CompGraphVisualizer
        is able to build a trace from a root node. Usually the root node
        is simply the final result of an operation. For a MLP, it usually
        is the loss function.

        ..note: CompGraphVisualizer is made to visualize a computational 
        graph. To have a more holistic view of a neural network architecture,
        prefer the use of NetworkVisualizer. 
    """
    def __init__(self):
        self.nodes, self.edges = set(), set()
    
    def visualize(self, root, rankdir="LR"):
        r"""
            Builds the computational graph and displays it.

            Args:
                root (Tensor): node to start building the trace.
                rankdir (str): graph organization
        """
        self._build_trace(root)
        graph = self._build_graph(rankdir=rankdir)
        return graph
    
    def _build_trace(self, node):
        r"""Performs a recursive DFS over the computational graph."""
        if node not in self.nodes:
            self.nodes.add(node)
            for child in node.children:
                self.edges.add((child, node))
                self._build_trace(child)
    
    def _build_graph(self, rankdir:str='LR'):
        r"""
            Plots the graph

            Args:
                rankdir (str): TB (top to bottom graph) | LR (left to right)

            ..note: A node is uniquely identified with the id() function which guarantees
                    a unique number for every Python object.
        """
        assert rankdir in ['LR', 'TB'], f"Unexpected rankdir argument (TB, LR available). Got {rankdir}."
        graph = Digraph(format='png', graph_attr={'rankdir': rankdir})
        
        for n in self.nodes:
            graph.node(name=str(id(n)), label = f"{n.name} | {n.shape}", shape='record')
            if n.op:
                graph.node(name=str(id(n)) + n.op, label=n.op)
                graph.edge(str(id(n)) + n.op, str(id(n)))
        
        for n1, n2 in self.edges:
            if n2.op:
                graph.edge(str(id(n1)), str(id(n2)) + n2.op)
            else:
                graph.edge(str(id(n1)), str(id(n2)))
        
        return graph
