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

        fig = go.Figure(
            layout=go.Layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))
        """Make edge trace."""
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name="Edges")

        fig.add_trace(edge_trace)
        """Make node traces."""

        nodes = {
            n: {
                "node_x": [],
                "node_y": [],
                "color": [],
                "text": [],
                "size": []
            }
            for n in ['primitive', 'functor', 'monoid', "output"]
        }

        for i, n in enumerate(graph.nodes):
            o = n.output()

            t = None
            c = None
            if isinstance(n, tj.primitive):
                c = "black"
                t = "primitive"

            if isinstance(n, tj.functor):
                c = "purple"
                t = "functor"

            if isinstance(n, tj.monoid):
                c = "red"
                t = "monoid"

            text = "name: %s --- output: %s --- type: %s" % (n.name, o, t)

            if n == master:
                nodes["output"]["node_x"].append(node_x[i])
                nodes["output"]["node_y"].append(node_y[i])
                nodes["output"]["color"].append("green")
                nodes["output"]["text"].append(text)
                nodes["output"]["size"].append(o)
            else:
                nodes[t]["node_x"].append(node_x[i])
                nodes[t]["node_y"].append(node_y[i])
                nodes[t]['color'].append(c)
                nodes[t]['text'].append(text)
                nodes[t]['size'].append(o)

        # Normalize sizes
        sizes = []
        for k in nodes.keys():
            sizes.extend(nodes[k]["size"])

        sizes = np.array(sizes)

        for k in nodes.keys():
            nodes[k]["size"] = np.array(nodes[k]["size"])
            nodes[k]["size"] = (nodes[k]["size"] - sizes.mean())\
                / (sizes.std() + 1e-8)

            nodes[k]["size"] = np.nan_to_num(nodes[k]["size"])

        # Add traces
        for k in nodes.keys():
            node_trace = go.Scatter(
                x=nodes[k]["node_x"],
                y=nodes[k]["node_y"],
                mode='markers',
                hoverinfo='text',
                marker={
                    "color": nodes[k]["color"],
                    "size": 6 * nodes[k]["size"] + 10
                },
                hovertext=nodes[k]["text"],
                name=k)

            fig.add_trace(node_trace)

        fig.update_layout(
            title=go.layout.Title(
                text="Visualization of Calculation Graph", xref="paper", x=0),
            xaxis={
                'showgrid': False,  # thin lines in the background
                'zeroline': False,  # thick line at x=0
                'visible': False,  # numbers below
            },
            yaxis={
                'showgrid': False,  # thin lines in the background
                'zeroline': False,  # thick line at x=0
                'visible': False,  # numbers below
            })

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
