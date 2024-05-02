from pycallgraph2 import PyCallGraph, Config, GlobbingFilter, output
import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
import pickle

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

def time_report(log_folder_path):
    # Read the csv file into a pandas dataframe
    df = pd.read_csv(f"{log_folder_path}/log_time.csv")

    df = df.groupby("Function").agg(Total_time=('Time', 'sum'), Total_visits=('Time', 'count')).sort_values("Total_time", ascending=False).reset_index()
    df["Average_time_per_visit"] = df["Total_time"] / df["Total_visits"]

    return df

def time_report_boxplot(log_folder_path):
    df = pd.read_csv(f"{log_folder_path}/log_time.csv")
    plt.figure(figsize=(10, 5))
    for i, name in enumerate(df["Function"].unique()):
        to_plot = df.loc[df["Function"] == name, "Time"].values
        # Normalize the time distribution
        to_plot = (to_plot - to_plot.min()) / (to_plot.max() - to_plot.min())
        plt.boxplot(to_plot, labels=[name], positions=[i], vert=False)
    
    plt.tight_layout()
    plt.title("Normalized time distribution per function")
    plt.savefig(f"{log_folder_path}/time_report_boxplot.png")

    return df

def get_unpicklable(instance, exception=None, string='', first_only=True):
    """
    Recursively go through all attributes of instance and return a list of whatever
    can't be pickled.

    Set first_only to only print the first problematic element in a list, tuple or
    dict (otherwise there could be lots of duplication).
    """
    problems = []
    if isinstance(instance, tuple) or isinstance(instance, list):
        for k, v in enumerate(instance):
            try:
                pickle.dumps(v)
            except BaseException as e:
                problems.extend(get_unpicklable(v, e, string + f'[{k}]'))
                if first_only:
                    break
    elif isinstance(instance, dict):
        for k in instance:
            try:
                pickle.dumps(k)
            except BaseException as e:
                problems.extend(get_unpicklable(
                    k, e, string + f'[key type={type(k).__name__}]'
                ))
                if first_only:
                    break
        for v in instance.values():
            try:
                pickle.dumps(v)
            except BaseException as e:
                problems.extend(get_unpicklable(
                    v, e, string + f'[val type={type(v).__name__}]'
                ))
                if first_only:
                    break
    else:
        for k, v in instance.__dict__.items():
            try:
                pickle.dumps(v)
            except BaseException as e:
                print(f"Error in {k} with value {v}")
                problems.extend(get_unpicklable(v, e, string + '.' + k))

    # if we get here, it means pickling instance caused an exception (string is not
    # empty), yet no member was a problem (problems is empty), thus instance itself
    # is the problem.
    if string != '' and not problems:
        problems.append(
            string + f" (Type '{type(instance).__name__}' caused: {exception})"
        )

    return problems