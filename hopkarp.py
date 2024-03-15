from queue import Queue
from math import inf as INFINITY
from utils import load_file, matrix_to_adj_list

DUMMY_NODE = -1


class BipGraph(object):
    """
    Class to represent a bipartite graph
    dim_m: number of left nodes
    dim_n: number of right nodes
    adjacency_hash: dict containing an adjacency list of the bipartite graph
    u_to_v_map: dict containing the matching of left nodes to right nodes
    v_to_u_map: dict containing the matching of right nodes to left nodes
    distance_to_dummy: dict containing the distance of each left node to the dummy node
    """

    def __init__(self, adj_matrix):
        """
        :param adj_matrix: The adjacency matrix of the bipartite graph
        """

        self.dim_m = len(adj_matrix)
        self.dim_n = len(adj_matrix[0])
        self.adjacency_hash = matrix_to_adj_list(adj_matrix)

        self.bfs_edges = []

        self.u_to_v_map = {}
        self.v_to_u_map = {}
        self.distance_to_dummy = {}

        # initalize all pairs to connect to the dummy node
        for u in range(self.dim_m):
            self.u_to_v_map[u] = DUMMY_NODE
            self.distance_to_dummy[u] = DUMMY_NODE
        for v in range(self.dim_n):
            self.v_to_u_map[v] = DUMMY_NODE

    def bfs(self):
        """
        Returns true an augmenting path is found, else returns false
        """
        self.bfs_edges = []
        Q = Queue()
        for u in range(self.dim_m):
            # if the node is paired to the dummy (aka unmatched)
            if self.u_to_v_map[u] == DUMMY_NODE:
                self.distance_to_dummy[u] = 0
                Q.put(u)
            else:
                self.distance_to_dummy[u] = INFINITY

        self.distance_to_dummy[DUMMY_NODE] = INFINITY
        # variable to keep track of the tree created
        while not Q.empty():
            u = Q.get()
            # see if we can find a shorter path to the dummy node
            if self.distance_to_dummy[u] < self.distance_to_dummy[DUMMY_NODE]:
                # for all right nodes adjacent to left node u
                for v in self.adjacency_hash[u]:
                    # if the right node is unmatched
                    self.bfs_edges.append((u, v))
                    if self.distance_to_dummy[self.v_to_u_map[v]] == INFINITY:
                        # we found an augmenting path, update the distance and add the right node to the queue
                        self.distance_to_dummy[self.v_to_u_map[v]] = self.distance_to_dummy[u] + 1
                        Q.put(self.v_to_u_map[v])
        augmenting_path_found = self.distance_to_dummy[DUMMY_NODE] != INFINITY
        return augmenting_path_found

    def dfs(self, u):
        """
        Returns true if there is an augmenting path beginning with free vertex u
        """
        if u == DUMMY_NODE:
            # if we reach the dummy node, we found an augmenting path
            return True
        # get all nodes adjacent to and see if we can find an augmenting path
        for v in self.adjacency_hash[u]:
            if self.distance_to_dummy[self.v_to_u_map[v]] == self.distance_to_dummy[u] + 1:
                # recursively search deeper for an augmenting path
                if self.dfs(self.v_to_u_map[v]):
                    print(f"dfs: {u}, {v}")
                    self.v_to_u_map[v] = u
                    self.u_to_v_map[u] = v
                    return True
        # this is reached if there is no augmenting path beginning with u
        # so we update the distance to the dummy node to infinity
        self.distance_to_dummy[u] = INFINITY
        return False

    def find_matches_hopcroft_karp(self):
        matches = 0
        # while there is an augmenting path to be found
        while self.bfs():
            print(f"bfs_edges: {self.bfs_edges}")
            matched_nodes = [(u, v) for u, v in self.u_to_v_map.items() if v != DUMMY_NODE]
            print(f"matched nodes: {matched_nodes}")
            # first intersection of the edges of the bfs tree and the matching nodes will be removed
            edge_to_augment = [edge for edge in self.bfs_edges if edge in matched_nodes]
            if len(edge_to_augment) > 0:
                edge_to_augment = edge_to_augment[0]
            print(f"edge remove: {edge_to_augment}")
            unmatched_nodes = [u for u, v in self.u_to_v_map.items() if v == DUMMY_NODE]
            print(f"unmatched nodes: {unmatched_nodes}")
            for u in range(self.dim_m):
                # if the node is free and we found an augmenting path
                if self.u_to_v_map[u] == DUMMY_NODE and self.dfs(u):
                    matches += 1
        matched_nodes = [(u, v) for u, v in self.u_to_v_map.items() if v != DUMMY_NODE]
        print(f"matched nodes: {matched_nodes}")
        return matches


if __name__ == "__main__":
    adj_matrix = load_file("size3.graph")
    g = BipGraph(adj_matrix)
    print("Size of maximum matching is %d" % g.find_matches_hopcroft_karp())
