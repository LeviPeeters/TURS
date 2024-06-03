import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
import pickle

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

# def time_report(log_folder_path):
#     # Read the csv file into a pandas dataframe
#     df = pd.read_csv(f"{log_folder_path}/log_time.csv")

#     df = df.groupby("Function").agg(Total_time=('Time', 'sum'), Total_visits=('Time', 'count')).sort_values("Total_time", ascending=False).reset_index()
#     df["Average_time_per_visit"] = df["Total_time"] / df["Total_visits"]

#     return df

def time_report_per_literal(log_folder_path):
    df = pd.read_csv(f"{log_folder_path}/log_time.csv")
    plt.figure(figsize=(10, 5))
    
    plt.boxplot(df["time"], vert=False)

    plt.title("Distribution of time spent per literal")
    plt.savefig(f"{log_folder_path}/time_report_boxplot.png")

    return df

def time_report_per_candidate(chunksize, log_folder_path):
    df = pd.read_csv(f"{log_folder_path}/log_time_per_candidate.csv", header=1)
    plt.figure(figsize=(10, 10))
    df = df.groupby("n_literals").agg(time=('time', 'mean')).reset_index()
    
    plt.plot(df["n_literals"], df["time"], marker='.', linestyle=" ", color='b')

    if chunksize == 97:
        plt.title(f"Average time per literal per candidate, chunksize = len(literals) / 16")
    else:
        plt.title(f"Average time per literal per candidate, chunksize = {chunksize}")
    plt.xlabel("Number of literals for candidate")
    plt.ylabel("Time per literal (s)")
    plt.savefig(f"{log_folder_path}/time_vs_nliterals.png")

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