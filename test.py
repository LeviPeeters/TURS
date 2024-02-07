import numpy as np
import pandas as pd
import utils 

def huts():
    print("Huts")
    return 1

def huts2():
    x = np.array([1, 2, 3])
    y = pd.DataFrame(x)
    # Call a bunch of descriptive functions
    y = y.to_numpy()
    y = y.flatten()
    y = y.mean()
    y = y.sum()
    y = y.max()
    y = y.min()
    y = y.std()
    y = y.var()
    y = y.cumsum()
    
    huts()

if __name__=="__main__":
    utils.call_graph_filtered(huts2, "call_graph_test.png")
    huts2()
    print("Done")