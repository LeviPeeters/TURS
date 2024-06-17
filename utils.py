import logging
import time
import pandas as pd
import matplotlib.pyplot as plt

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

class RegularFormatter(logging.Formatter):
    """
    A custom formatter for logging that includes the elapsed time between log entries.
    """

    def __init__(self):
        pass

    def format(self, record):
        timestamp = time.strftime('%H:%M:%S', time.gmtime(time.time()))
        return f"{timestamp} - {record.getMessage()}"

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
        if len(to_plot) > 1:
            # Normalize the time distribution
            to_plot = (to_plot - to_plot.min()) / (to_plot.max() - to_plot.min())
            plt.boxplot(to_plot, labels=[name], positions=[i], vert=False)
    
    plt.tight_layout()
    plt.title("Normalized time distribution per function")
    plt.savefig(f"{log_folder_path}/time_report_boxplot.png")

    return df