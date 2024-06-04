#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Idea and code was taken from stackoverflow().
This sample illustrates how to
+ how to pass method of instance method
  to multiprocessing(idea and code was introduced
  at http://goo.gl/tRHN1D by torek).
+ how to use fuctools.partial() with multiprocessing.pool.map()
"""
import os
import multiprocessing as mp
import numpy as np
import time

def experiment(task, i):
    task.work(i)

class Ruleset():
    def __init__(self, data):
        self.data = data
    
    def get_data(self):
        return self.data

    def run(self):
        tasks = []
        for i in range(10):
            worker = Rule(self)
            tasks.append(worker)

        res = pool.starmap_async(experiment, [(task,i) for i, task in enumerate(tasks)])
        res.wait()
        pool.close()


class Rule(Ruleset):
    def __init__(self, data):
        super().__init__(data)

    def work(self, i):
        print("hoerenhel")
        print(super().data)

if __name__ == '__main__':
    s = time.time()
    pool = mp.Pool()
    np.random.seed(0)
    data = np.random.rand(10**7)
    ruleset = Ruleset(data)
    ruleset.run()
    

    print('Time:', time.time() - s)