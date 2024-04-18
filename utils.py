from pycallgraph2 import PyCallGraph, Config, GlobbingFilter, output
import logging
import time
import pandas as pd

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

class ElapsedTimeFormatter(logging.Formatter):
    """
    A custom formatter for logging that includes the elapsed time between log entries.
    """

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        elapsed = time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))
        self.start_time = time.time()
        return f"{elapsed} - {record.getMessage()}"

def time_report(csv_filename):
    # Read the csv file into a pandas dataframe
    df = pd.read_csv(csv_filename)

    df = df.groupby("Function").agg(Total_time=('Time', 'sum'), Total_visits=('Time', 'count')).sort_values("Total_time", ascending=False).reset_index()
    df["Average_time_per_visit"] = df["Total_time"] / df["Total_visits"]

    return df
