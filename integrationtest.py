from cluster import Cluster, Kmeans
from time import time as TIME

"""Tests for data normalisation and scaling methods."""

def testminmaxscaler(k):
    kmeans = Kmeans("euclidian distance", 10, k)
    test = kmeans.loadplanktonsize()
    testscaled = kmeans.minmaxscaler(test)
    for sample in testscaled:
        for value in sample:
            assert not (value > 1)
            assert not (value < 0)

"""Tests for evaluation methods"""    

def testsumsqrdist(k):
    kmeans = Kmeans("euclidian distance", 10, k)
    newarr = kmeans.loadspecimensize()
    results = kmeans.run(newarr)
    print(kmeans.sumsqrdist(newarr, results, kmeans.centroids))

"""Tests for kmeans class methods"""

def testinitcentro(k):
    test = [[1.434, 5.217, 3.146], [7.777, 12.241, 2.672], [1.004, 2.535, 6.244], [1.111, 9.098, 4.601], [1.111, 2.222, 3.333], [5.606, 1.007, 2.500]]
    kmeans = Kmeans("euclidian distance", 10, k)
    initcent = []
    initcent = kmeans.initcentro(test)
    assert len(initcent) == k

def testassign(k):
    test = [[1.434, 5.217, 3.146], [7.777, 12.241, 2.672], [1.004, 2.535, 6.244], [1.111, 9.098, 4.601], [1.111, 2.222, 3.333], [5.606, 1.007, 2.500], [5.111, 2.234, 7.808], [6.080, 2.534, 7.187], [3.343, 6.091, 8.353]]
    results = [0] * len(test)
    kmeans = Kmeans("euclidian distance", 10, k)
    kmeans.centroids = kmeans.initcentro(test)
    results = kmeans.assigncluster(test, results)
    for i in range(len(results)):
        assert 0 < results[i] < k+1
    
def testnewcentroids():
    curcent = [[1,4]]
    test = [[1, 2],[2, 3],[3,2],[3,4],[1,4]]
    kmeans = Kmeans("euclidian distance", 10, 3)
    results = [0] * len(test)
    kmeans.centroids = curcent
    testcent = kmeans.newcentro(test, results)
    assert testcent == [[2.0, 3.0]]

def testkmeans(k):
    print("\nTesting kmeans on fake dataset.")
    test = [[1.434, 5.217, 3.146, 0], [7.777, 12.241, 2.672, 0], [1.004, 2.535, 6.244, 0], [1.111, 9.098, 4.601, 0], 
            [1.111, 2.222, 3.333, 0], [5.606, 1.007, 2.500, 0], [5.111, 2.234, 7.808, 0], [6.080, 2.534, 7.187, 0], 
            [3.343, 6.091, 8.353, 0], [5.201, 12.111, 19.501, 0], [9.800, 1.321, 3.098, 0], [4.056, 7.809, 1.954, 0], 
            [10.400, 5.055, 7.677, 0], [6.579, 3.121, 8.808, 0], [1.366, 6.098, 0.099, 0], [5.816, 1.007, 9.500, 0]]
    kmeans = Kmeans("euclidian distance", 10, k)
    start = TIME()
    results = kmeans.run(test)
    print("Time to run algorithm:", TIME() - start,"seconds")
    kmeans.summary(test, results)

"""Normalisation and scaling test"""
testminmaxscaler(3)

"""Similarity metric tests"""
testsumsqrdist(5)

"""Kmeans class method tests"""
testinitcentro(3)
testassign(4)
testnewcentroids()
testkmeans(3)