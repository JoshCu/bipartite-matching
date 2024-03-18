# Description: Implementation of the Hopcroft-Karp algorithm for finding the maximum matching in a bipartite graph
# It makes use of an additional dummy node connected to all unmatched nodes in the left set
# This all makes it easier to find augmenting paths


from queue import Queue
from math import inf as INFINITY
from utils import load_file, matrix_to_adj_list, parse_args

DUMMY_NODE = -1

# left nodes are u
# right nodes are v


class BipGraph(object):
    """
    Class to represent a bipartite graph
    dim_u: number of u nodes
    dim_v: number of v nodes
    adj_lookup: dict containing an adjacency list of the bipartite graph
    u_to_v_map: dict containing the matching of u nodes to v nodes
    v_to_u_map: dict containing the matching of v nodes to u nodes
    distance_to_dummy: dict containing the distance of each u node to the dummy node
    """

    def __init__(self, adj_matrix):
        """
        :param adj_matrix: The adjacency matrix of the bipartite graph
        """

        self.dim_u = len(adj_matrix)
        self.dim_v = len(adj_matrix[0])
        self.adj_lookup = matrix_to_adj_list(adj_matrix)

        self.bfs_edges = []

        self.u_to_v_map = {}
        self.v_to_u_map = {}
        self.distance_to_dummy = {}

        # initalize all pairs to connect to the dummy node
        for u in range(self.dim_u):
            self.u_to_v_map[u] = DUMMY_NODE
            self.distance_to_dummy[u] = DUMMY_NODE
        for v in range(self.dim_v):
            self.v_to_u_map[v] = DUMMY_NODE

    def bfs(self):
        """
        Returns true an augmenting path is found, else returns false
        """
        self.bfs_edges = []
        Q = Queue()
        for u in range(self.dim_u):
            # if the node is paired to the dummy (aka unmatched)
            if self.u_to_v_map[u] == DUMMY_NODE:
                # set the distance to the dummy node to 0 and add it to the queue
                self.distance_to_dummy[u] = 0
                Q.put(u)
            else:
                self.distance_to_dummy[u] = INFINITY

        self.distance_to_dummy[DUMMY_NODE] = INFINITY
        # variable to keep track of the tree created
        while not Q.empty():
            u = Q.get()
            # if u is less than the distance to the dummy node
            if self.distance_to_dummy[u] < self.distance_to_dummy[DUMMY_NODE]:
                # for each v connected to u
                for v in self.adj_lookup[u]:
                    # add the edge to the tree
                    self.bfs_edges.append((u, v))
                    # if the distance to the dummy node is infinity
                    if self.distance_to_dummy[self.v_to_u_map[v]] == INFINITY:
                        # we found an augmenting path, update the distance and add the right node to the queue
                        self.distance_to_dummy[self.v_to_u_map[v]] = self.distance_to_dummy[u] + 1
                        Q.put(self.v_to_u_map[v])
        # if the distance to the dummy node is not infinity then a path to the dummy node was found
        augmenting_path_found = self.distance_to_dummy[DUMMY_NODE] != INFINITY
        return augmenting_path_found

    def dfs(self, u):
        """
        Returns true if there is an augmenting path beginning with free vertex u
        """
        if u == DUMMY_NODE:
            # if we reach the dummy node, we found an augmenting path
            return True
        # get all nodes adjacent to u and see if we can find an augmenting path
        for v in self.adj_lookup[u]:
            if self.distance_to_dummy[self.v_to_u_map[v]] == self.distance_to_dummy[u] + 1:
                # recursively search deeper for an augmenting path
                if self.dfs(self.v_to_u_map[v]):
                    print(f"add match: {repr((u,v))}")
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
        # perform a bfs on all the unmatched nodes
        while self.bfs():
            print(f"bfs_edges: {repr(self.bfs_edges)}")
            matched_nodes = [(u, v) for u, v in self.u_to_v_map.items() if v != DUMMY_NODE]
            print(f"matched nodes: {repr(matched_nodes)}")
            # first intersection of the edges of the bfs tree and the matching nodes will be removed
            edge_to_augment = [edge for edge in self.bfs_edges if edge in matched_nodes]
            if len(edge_to_augment) > 0:
                edge_to_augment = edge_to_augment[0]
            print(f"edge remove: {repr(edge_to_augment)}")
            unmatched_nodes = [u for u, v in self.u_to_v_map.items() if v == DUMMY_NODE]
            print(f"unmatched nodes: {repr(unmatched_nodes)}")
            for u in range(self.dim_u):
                # if the node is free and we found an augmenting path
                if self.u_to_v_map[u] == DUMMY_NODE and self.dfs(u):
                    matches += 1
        matched_nodes = [(u, v) for u, v in self.u_to_v_map.items() if v != DUMMY_NODE]
        print(f"matched nodes: {repr(matched_nodes)}")
        return matches


if __name__ == "__main__":
    adj_matrix = load_file(parse_args())
    g = BipGraph(adj_matrix)
    print("Size of maximum matching is %d" % g.find_matches_hopcroft_karp())
