"""Module visualized a graph."""
import plotly.graph_objs as go
import networkx
import tensorjo as tj
import numpy as np


class visualizer():
    """Keeps visualization state."""

    def __init__(self,
                 graph: "tj.graph",
                 layout: "networkx.drawing" = networkx.drawing.spectral_layout,
                 filename="graph"):
        """Initialize with nodes to visualize."""
        self.graph = graph
        self.layout = layout
        self.filename = filename

    def draw(self, master: "node.node"):
        """Draw the graph.

        If master is none -- draw full graph.
        """
        nodes = trace(master)

        graph = build_graph(nodes)

        edge_x, edge_y, node_x, node_y = get_geometric_data(
            graph, self.layout(graph, scale=5))

        node_x = node_x
        node_y = node_y

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name="Edges")

        color = []
        text = []
        sizes = []
        for n in graph.nodes:
            o = n.output()

            t = None
            if isinstance(n, tj.primitive):
                color.append("black")
                t = "primitive"

            if isinstance(n, tj.functor):
                color.append("purple")
                t = "functor"

            if isinstance(n, tj.monoid):
                color.append("red")
                t = "monoid"

            if n == master:
                color[-1] = "green"

            text.append("name: %s --- output: %s --- type: %s" % (n.name, o,
                                                                  t))

            sizes.append(o)

        sizes = np.array(sizes)
        sizes = (sizes - sizes.mean()) / sizes.std()

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            marker={"color": color,
                    "size": 6 * sizes + 10},
            hovertext=text,
            name="Nodes")
        """Add info to nodes."""

        fig = go.Figure()
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)

        fig.update_layout(
            title=go.layout.Title(
                text="Visualization of Calculation Graph", xref="paper", x=0),
        )

        fig.show(filename=self.filename)


def get_geometric_data(graph: "networkx.graph", layout: dict):
    """Get node positions and edges."""
    edge_x = []
    edge_y = []

    node_x = []
    node_y = []

    layout = {
        n: [x + np.random.rand(), y + np.random.rand()]
        for n, (x, y) in layout.items()
    }

    # print(len(graph.edges))
    # print(len(graph.nodes))
    # raise Exception()

    for f, t in graph.edges:
        fx, fy = layout[f]
        tx, ty = layout[t]

        edge_x.append(fx)
        edge_x.append(tx)
        edge_x.append(None)

        edge_y.append(fy)
        edge_y.append(ty)
        edge_y.append(None)

    for n in graph.nodes:
        nx, ny = layout[n]

        node_x.append(nx)
        node_y.append(ny)

    return edge_x, edge_y, node_x, node_y


def build_graph(nodes: ["node.node"]) -> "networkx.Graph":
    """Build a DAG of the nodes."""
    dag = networkx.Graph()
    """add nodes to graph."""
    for n in nodes:
        dag.add_node(n)
    """add edges to graph."""
    for n in nodes:
        if isinstance(n, tj.monoid):
            dag.add_edge(n.m1, n)
            dag.add_edge(n.m2, n)

        if isinstance(n, tj.functor):
            dag.add_edge(n.m1, n)

    return dag


def trace(master: "node.node") -> ["node.node"]:
    """Trace the graph from master."""
    nodes = set()

    def dfs(node: "node.node"):
        """DFS the graph to find all inputs."""
        nodes.add(node)

        if isinstance(node, tj.primitive):
            return

        if node.m1 in nodes:
            return

        nodes.add(node)

        if isinstance(node, tj.monoid):
            dfs(node.m1)
            dfs(node.m2)
        elif isinstance(node, tj.functor):
            dfs(node.m1)
        elif isinstance(node, tj.primitive):
            return

    dfs(master)

    return list(nodes)
