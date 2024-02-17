import numpy as np
import pandas as pd
import os
import utils 
from multiprocessing import shared_memory
from test2 import DataInfo
from datetime import datetime

class ClassB:
    def __init__(self, shared_array):
        self.shared_array = shared_array

    def get_data_view(self, rows):
        return self.shared_array[rows]
    
    def close(self):
        self.shared_array.close()

print(os.getppid())
print(f"{datetime.now()} - begin")
breakpoint()

# Initialize ClassA with some data
datainfo = DataInfo()
datainfo.initialize((100000, 1000))

print(f"{datetime.now()} - data initialized")
breakpoint()

# Initialize ClassB with the shared memory
b = ClassB(datainfo.shared_data.buf)

print(f"{datetime.now()} - ClassB initialized")

# Now you can use ClassB with different permutations of rows
rows1 = [0, 1, 2, 3]
rows2 = [100, 101, 102, 103]
# view1 = b.get_data_view(rows1)
view2 = b.get_data_view(rows2)

# Ensure to cleanup shared memory when done
b.close()
datainfo.close()