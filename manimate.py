from manim import *
from utils import load_file, matrix_to_edge_pairs

import networkx as nx

graph_name = "size4.graph"
anim_step_file = graph_name + ".anim"

NODE_RADIUS = 0.3
MATCHED_COLOR = GREEN
SCR_HEIGHT = 8
SCR_WIDTH = 14


def get_node_name(node, is_left):
    if is_left:
        return f"u_{node}"
    else:
        return f"v_{node}"


def get_node_config(nodes):
    node_config = {}
    for node in nodes:
        if node.startswith("u"):
            node_config[node] = {"fill_color": GREEN, "radius": NODE_RADIUS}
        else:
            node_config[node] = {"fill_color": BLUE, "radius": NODE_RADIUS}
    return node_config


def get_edge_config(edges):
    edge_config = {}
    for edge in edges:
        edge_config[edge] = {"stroke_width": 6}
    return edge_config


def get_node_partitions(graph, root_node):
    """
    Gets the partitions of the nodes, using distance from root as the partition
    """
    distances = nx.shortest_path_length(graph, root_node)
    # convert d[node] = distance to [[nodes_dist1][nodes_dist2]...]
    partitions = [[root_node]]
    for node, dist in distances.items():
        if dist == 0:
            continue
        while len(partitions) <= dist:
            partitions.append([])
        partitions[dist].append(node)
    return partitions


def create_graph_from_adj_matrix(adj_matrix):
    # get lists of u_0 to u_n and v_0 to v_n
    u_nodes = [get_node_name(node, True) for node in range(len(adj_matrix))]
    v_nodes = [get_node_name(node, False) for node in range(len(adj_matrix[0]))]
    nodes = u_nodes + v_nodes
    # get the adjacency list and rename the nodes from indices to u_0 to u_n and v_0 to v_n
    edges = matrix_to_edge_pairs(adj_matrix)
    edges = [(u_nodes[i], v_nodes[j]) for i, j in edges]
    # set the partitions used for the layout
    partitions = [u_nodes, v_nodes]

    return Graph(
        nodes,
        edges,
        labels=True,
        layout="partite",
        partitions=partitions,
        vertex_config=get_node_config(nodes),
        edge_config=get_edge_config(edges),
    )


def create_bfs_graphs(unnamed_edges, free_nodes, matched_edges):
    free_nodes = [get_node_name(node, True) for node in free_nodes]
    edges = [(get_node_name(u, True), get_node_name(v, False)) for u, v in unnamed_edges]
    # reverse the matched edges for making a digraph
    # this ensures alternating match unmatched paths
    matched_edges = [(get_node_name(v, False), get_node_name(u, True)) for u, v in matched_edges]
    graphs = VGroup()
    edges += matched_edges

    reference_graph = nx.DiGraph(edges)
    for root in free_nodes:
        # get the subgraph of the reference graph with the root as the center
        subgraph = nx.ego_graph(reference_graph, root, radius=len(reference_graph.nodes))
        # reorder the nodes so the root is first
        nodes = list(subgraph.nodes)
        nodes.remove(root)
        nodes.insert(0, root)

        edges = list(subgraph.edges)
        # partitions for animation layout
        partitions = get_node_partitions(subgraph, root)
        # get the edge config and set the matched edges to green
        edge_config = get_edge_config(edges)
        for e in edges:
            if e in matched_edges:
                edge_config[e]["stroke_color"] = MATCHED_COLOR

        graph = Graph(
            nodes,
            edges,
            labels=True,
            layout="partite",
            partitions=partitions,
            vertex_config=get_node_config(nodes),
            root_vertex=root,
            edge_config=edge_config,
        )
        graphs.add(graph)

    return graphs


def get_step_inputs(anim_steps):
    bfs_edges = eval(anim_steps.pop(0).split(":")[1])
    matched_nodes = eval(anim_steps.pop(0).split(":")[1])
    edge_remove = eval(anim_steps.pop(0).split(":")[1])
    free_nodes = eval(anim_steps.pop(0).split(":")[1])

    # remove matched edges from bfs_edges
    bfs_edges = [edge for edge in bfs_edges if edge not in matched_nodes]

    return bfs_edges, matched_nodes, edge_remove, free_nodes


class BipartiteGraphAnimation(Scene):
    def construct(self):
        # default manim grid is 14*8
        spacing = 1

        G = create_graph_from_adj_matrix(load_file(graph_name))
        with open(anim_step_file, "r") as f:
            anim_steps = f.readlines()

        # scale graph to fit available height
        height_scale = (SCR_HEIGHT - spacing) / G.get_height()
        G.scale(height_scale)
        self.play(Create(G))
        self.wait()
        self.play(G.animate.to_edge(LEFT, buff=spacing))
        self.wait()

        # set all edges to grey
        self.play(*[e.animate.set_color(GRAY) for _, e in G.edges.items()])

        # calculate the width of the graph and the available width for the bfs graphs
        whole_graph_width = G.get_width() + spacing * 2
        available_width = SCR_WIDTH - whole_graph_width

        self.next_section("BFS Graphs")
        while True:
            bfs_edges, matched_nodes, edge_remove, free_nodes = get_step_inputs(anim_steps)

            free_n_highlight = {}
            indicate_free_nodes = []

            for u in free_nodes:
                u = get_node_name(u, True)
                # draw a circle around the node
                circle = Circle(color=YELLOW, stroke_width=5)
                circle.surround(G.vertices[u])
                indicate_free_nodes.append(Indicate(G.vertices[u]))
                # save the circle for later removal
                free_n_highlight[u] = circle

            self.play(
                *[Create(circle) for circle in free_n_highlight.values()], *indicate_free_nodes
            )

            self.next_section("Adding BFS Trees")
            self.wait(1)
            bfs_graphs = create_bfs_graphs(bfs_edges, free_nodes, matched_nodes)
            # arrange and scale to fit on screen
            bfs_graphs.arrange_in_grid(buff=0.3)
            # get the scale factor to fit the available width
            width_scale = (available_width - (2 * spacing)) / bfs_graphs.get_width()
            # get the scale factor to fit the available height
            height_scale = (SCR_HEIGHT - (2 * spacing)) / bfs_graphs.get_height()
            # use the smaller scale factor
            scale = min(width_scale, height_scale)
            bfs_graphs.scale(scale)
            bfs_graphs.next_to(G, RIGHT, buff=spacing)

            # unscale the stroke width
            for gr in bfs_graphs:
                for _, e in gr.edges.items():
                    scaled_stroke = e.get_stroke_width() * scale
                    e.set_stroke_width(scaled_stroke)

            # thankfully VGroup preserves order
            for i, gr in enumerate(bfs_graphs):
                root_key = get_node_name(free_nodes[i], True)
                root = gr.vertices[root_key]
                circle = free_n_highlight.get(root_key, None)
                if circle:
                    self.play(
                        Create(gr),
                        circle.animate.surround(root),
                    )
                    self.play(Indicate(root), FadeTransform(circle, root))
            self.wait(1)

            # set the edge to be removed to red then back to grey
            if type(edge_remove) == tuple:
                u = get_node_name(edge_remove[0], True)
                v = get_node_name(edge_remove[1], False)
                edges_to_remove = [G.edges[(u, v)]]

                for gr in bfs_graphs:
                    if (u, v) in gr.edges:
                        edges_to_remove.append(gr.edges[(u, v)])
                    if (v, u) in gr.edges:
                        edges_to_remove.append(gr.edges[(v, u)])

                self.play(*[e.animate.set_color(RED) for e in edges_to_remove], run_time=0.5)
                self.wait(1)
                self.play(*[e.animate.set_color(GRAY) for e in edges_to_remove], run_time=0.5)

            self.next_section("Matching Edges")

            # pop lines beginning with "add match" until we reach the next section
            matches = []
            while anim_steps[0].startswith("add match"):
                # space separated list of tuples
                matches.append(eval(anim_steps.pop(0).split(":")[1]))

            for match in matches:
                u = get_node_name(match[0], True)
                v = get_node_name(match[1], False)
                edge = (u, v)
                edges_to_match = [G.edges[edge]]
                for gr in bfs_graphs:
                    if edge in gr.edges:
                        edges_to_match.append(gr.edges[edge])
                    if (v, u) in gr.edges:
                        edges_to_match.append(gr.edges[(v, u)])
                self.play(*[e.animate.set_color(GREEN) for e in edges_to_match], run_time=0.5)
                self.play(*[Indicate(e) for e in edges_to_match], run_time=0.5)
                all_u = [g.vertices[u] for g in bfs_graphs if u in g.vertices.keys()]
                all_v = [g.vertices[v] for g in bfs_graphs if v in g.vertices.keys()]
                # remove any edge containing u or v
                all_edges = []
                for g in bfs_graphs:
                    for e in g.edges:
                        if u in e or v in e:
                            all_edges.append(g.edges[e])
                objects_to_remove = all_edges + all_u + all_v
                self.play(
                    *[Circumscribe(n, Circle) for n in all_v + all_u],
                    Circumscribe(G.vertices[u], Circle),
                    Circumscribe(G.vertices[v], Circle),
                    run_time=3,
                )

                # remove the vertices from the actual graphs
                for g in bfs_graphs:
                    if u in g.vertices.keys():
                        g.remove_vertices(u)
                    if v in g.vertices.keys():
                        g.remove_vertices(v)
                    if len(g.vertices) == 1:
                        objects_to_remove.append(g)
                        bfs_graphs.remove(g)
                if len(objects_to_remove) > 0:
                    self.play(*[FadeOut(o) for o in objects_to_remove])
                    # remove the objects from the scene
                    self.remove(*objects_to_remove)
                self.wait(1)

            if not anim_steps[0].startswith("bfs_edges"):
                if bfs_graphs:
                    self.play(*[FadeOut(g) for g in bfs_graphs])
                    self.remove(*bfs_graphs)
                break

        # add the bfs graph

        # Hold the final state for a moment before closing
        self.wait(1)

        # move the graph back to the center
        self.play(G.animate.move_to(ORIGIN))
        self.wait(1)

        # remove edges set to GRAY
        self.play(*[FadeOut(e) for _, e in G.edges.items() if e.get_color() == GRAY])
