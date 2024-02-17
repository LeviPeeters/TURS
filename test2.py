# Contains a class representing DataInfo, with a large array 
import numpy as np
from multiprocessing import shared_memory

class DataInfo:
    def __init__(self):
        pass

    @classmethod
    def initialize(self, shape):
        data = np.random.rand(shape[0], shape[1])  # Initialize with your actual data
        
        # Create a shared memory block from ClassA's data
        self.shared_data = shared_memory.SharedMemory(create=True, size=data.nbytes)
        # Copy ClassA's data into the shared memory
        shared_data_array = np.ndarray(data.shape, dtype=data.dtype, buffer=self.shared_data.buf)
        shared_data_array[:] = data[:]
    
    def close(self):
        self.shared_data.close()
        self.shared_data.unlink()
