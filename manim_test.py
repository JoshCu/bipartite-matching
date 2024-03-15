from manim import *
from copy import copy
from utils import load_file, matrix_to_edge_pairs


class BipartiteGraphAnimation(Scene):
    def construct(self):
        adj_matrix = load_file("size3.graph")
        u_nodes = range(len(adj_matrix))
        v_nodes = range(len(adj_matrix[0]))
        u_nodes = [str(node) for node in u_nodes]
        v_nodes = [f"u_{node}" for node in v_nodes]
        nodes = u_nodes + v_nodes
        partitions = [u_nodes, v_nodes]
        edges = matrix_to_edge_pairs(adj_matrix)
        edges = [(u_nodes[i], v_nodes[j]) for i, j in edges]
        print(nodes)
        print(edges)
        print(partitions)

        graph = Graph(nodes, edges, layout="partite", partitions=partitions)
        # Animate the creation of the graph

        self.play(Create(graph))
        self.wait(1)
        # graph2 = copy(graph)
        self.play(graph.animate.to_edge(LEFT))
        # self.wait(1)
        # self.play(Create(graph2))

        # Hold the final state for a moment before closing
        self.wait(1)
