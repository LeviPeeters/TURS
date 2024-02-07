from pycallgraph2 import PyCallGraph, Config, GlobbingFilter, output

def call_graph_filtered(
        function_, 
        output_png="call_graph.png",
        custom_include=None
    ):

    """A call graph generator filtered"""
    config = Config()
    config.trace_filter = GlobbingFilter(include=custom_include)
    graphviz = output.GraphvizOutput(output_file=output_png)

    with PyCallGraph(output=graphviz, config=config):
            function_()