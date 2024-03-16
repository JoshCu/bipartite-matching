from manim import *
from copy import copy
from utils import load_file, matrix_to_edge_pairs
from collections import defaultdict

graph_name = "size3.graph"
anim_step_file = graph_name + ".anim"

EDGE_CONFIG = {"stroke_width": 6}
NODE_RADIUS = 0.2


def get_node_name(node, is_left):
    if is_left:
        return f"u_{node}"
    else:
        return f"v_{node}"


def create_graph_from_adj_matrix(adj_matrix):
    u_nodes = range(len(adj_matrix))
    v_nodes = range(len(adj_matrix[0]))
    u_nodes = [get_node_name(node, True) for node in u_nodes]
    v_nodes = [get_node_name(node, False) for node in v_nodes]
    nodes = u_nodes + v_nodes
    partitions = [u_nodes, v_nodes]
    # colour u and v nodes differently
    node_config = {}
    for node in u_nodes:
        node_config[node] = {"fill_color": GREEN, "radius": NODE_RADIUS}
    for node in v_nodes:
        node_config[node] = {"fill_color": BLUE, "radius": NODE_RADIUS}
    edges = matrix_to_edge_pairs(adj_matrix)
    edges = [(u_nodes[i], v_nodes[j]) for i, j in edges]
    graph = Graph(
        nodes,
        edges,
        layout="partite",
        partitions=partitions,
        vertex_config=node_config,
        # layout_scale=1,
        edge_config=EDGE_CONFIG,
    )
    return graph


def create_graph_from_edges(unnamed_edges):
    all_edges = defaultdict(set)

    for e in unnamed_edges:
        u_e = get_node_name(e[0], True)
        v_e = get_node_name(e[1], False)
        all_edges[u_e].add(v_e)

    graphs = VGroup()
    for u_node, v_nodes in all_edges.items():
        v_nodes = sorted(list(v_nodes))

        # two lists below determine drawing order
        node_edges = [(u_node, v) for v in v_nodes]
        unique_nodes = [u_node] + v_nodes

        partitions = [[u_node], v_nodes]

        node_config = {}
        for node in unique_nodes:
            node_config[node] = {"fill_color": GREEN, "radius": NODE_RADIUS}
        node_config[u_node] = {"fill_color": BLUE, "radius": NODE_RADIUS}

        graph = Graph(
            unique_nodes,
            node_edges,
            layout="tree",
            partitions=partitions,
            vertex_config=node_config,
            root_vertex=u_node,
            edge_config=EDGE_CONFIG,
        )
        graphs.add(graph)
    return graphs


class BipartiteGraphAnimation(Scene):
    def construct(self):
        # default manim grid is 14*8
        wait_time = 0.1
        spacing = 1

        adj_matrix = load_file(graph_name)
        G = create_graph_from_adj_matrix(adj_matrix)

        with open(anim_step_file, "r") as f:
            anim_steps = f.readlines()

        # scale graph to fit available height
        self.play(Create(G), run_time=wait_time)
        self.wait(wait_time)
        self.play(G.animate.to_edge(LEFT, buff=spacing))
        self.wait(wait_time)

        # set all edges to grey
        self.play(*[e.animate.set_color(GRAY) for _, e in G.edges.items()])
        whole_graph_width = G.get_width() + spacing * 2
        available_width = 14 - whole_graph_width

        self.next_section("BFS Graphs")
        while True:
            bfs_edges = eval(anim_steps.pop(0).split(":")[1])
            matched_nodes = eval(anim_steps.pop(0).split(":")[1])
            edge_remove = eval(anim_steps.pop(0).split(":")[1])
            free_nodes = eval(anim_steps.pop(0).split(":")[1])

            for e in edge_remove:
                # set the edge to be removed to red then back to grey
                u = get_node_name(e[0], True)
                v = get_node_name(e[1], False)
                self.play(G.edges[(u, v)].animate.set_color(RED), run_time=0.5)
                self.wait(1)
                self.play(G.edges[(u, v)].animate.set_color(GRAY), run_time=0.5)

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
            bfs_graphs = create_graph_from_edges(bfs_edges)
            # arrange and scale to fit on screen
            bfs_graphs.arrange_in_grid(buff=0.3)
            # get the scale factor to fit the available width
            width_scale = (available_width - (2 * spacing)) / bfs_graphs.get_width()
            # get the scale factor to fit the available height
            height_scale = (8 - (2 * spacing)) / bfs_graphs.get_height()
            # use the smaller scale factor
            scale = min(width_scale, height_scale)
            bfs_graphs.scale(scale)
            bfs_graphs.next_to(G, RIGHT, buff=spacing)

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
                        # circle.animate.move_to(root.get_center()),
                        circle.animate.surround(root),
                    )
                    self.play(Indicate(root), FadeTransform(circle, root))
            self.wait(1)

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
                self.play(G.edges[edge].animate.set_color(GREEN))
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
                    *[Indicate(n) for n in all_v + all_u],
                    Indicate(G.vertices[u]),
                    Indicate(G.vertices[v]),
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

                self.play(*[FadeOut(o) for o in objects_to_remove])
                # remove the objects from the scene
                self.remove(*objects_to_remove)
                self.wait(1)

            break

        # add the bfs graph

        # Hold the final state for a moment before closing
        self.wait(1)
