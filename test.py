import multiprocessing as mp
import time

def setup(huts):
    global huge_test_array
    huge_test_array = huts

class Test:
    def __init__(self):
        self.unit = 4
        huge_test_array = [x for x in range(10000000)]
        # self.huge_test_array = [x for x in range(50000000)]
        pool = mp.Pool(2, initializer=setup, initargs=(huge_test_array,))
        unit = Unit()
        pool.map(unit.func, [(x) for x in [1, 2, 3]])

    def func(self, x):
        # print(self.huge_test_array[15*x]*x)
        print(huge_test_array[15*x]*x)
        # print(x)

class Unit:
    def func(self, x):
        print(huge_test_array[15*x]*x)

if __name__ == '__main__':
    s = time.time()
    Test()
    print(time.time()-s)