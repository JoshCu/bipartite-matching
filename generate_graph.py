import networkx as nx
import argparse


def generate_bipartite_graph(size):
    # Generate a bipartite graph using networkx
    G = nx.algorithms.bipartite.generators.random_graph(size, size, 0.3)
    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a bipartite graph")
    parser.add_argument("size", type=int, help="Size of the bipartite graph")
    args = parser.parse_args()

    graph_size = args.size
    bipartite_graph = generate_bipartite_graph(graph_size)

    print(bipartite_graph)
    a_nodes = {n for n, d in bipartite_graph.nodes(data=True) if d["bipartite"] == 0}
    adjacency_matrix = nx.algorithms.bipartite.matrix.biadjacency_matrix(
        bipartite_graph, row_order=sorted(a_nodes)
    ).toarray()
    file_name = f"size{graph_size}.graph"
    with open(file_name, "w") as file:
        for row in adjacency_matrix:
            file.write(" ".join(map(str, row)) + "\n")
    print(f"Graph written to {file_name}")
