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
import functools


def call_it2(instance, name, arg):
    "indirect caller for instance methods and multiprocessing"
    return getattr(instance, name)(arg)


class Klass(object):
    """
    """

    def __init__(self, nobj, workers=mp.cpu_count()):
        # print ("Constructor (in pid=%d)..." % os.getpid() )
        self.count = nobj

        pool = mp.Pool(processes=workers)

        # with functools.partial(), I can use pool.map()
        func_call_it = functools.partial(call_it2, self, 'process_obj')
        results = pool.map(func_call_it, list(i for i in range(nobj)))

        pool.close()
        pool.join()

        print (results)

    def __del__(self):
        self.count -= 1
        # print ("... Destructor (in pid=%d) count=%d" % (os.getpid(), self.count) )

    def process_obj(self, index):
        # print ("object %d") % index
        return "results"


def main():
    Klass(nobj=8, workers=3)
    print ('ok')


if __name__ == '__main__':
    main()