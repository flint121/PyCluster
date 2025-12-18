from cluster import Cluster, Kmeans
from time import time as TIME

"""Tests for initialising objects"""

def testinit():
    kmeans = Kmeans("euclidian distance", 10, 3)
    
"""Tests for similarity and distance metrics"""

def testeuclid():
    v1 = [1, 2, 3, 4]
    v2 = v1
    v3 = [3, 2, 1, 0]
    kmeans = Kmeans("euclidian distance", 10, 3)
    assert kmeans.euclidiandistance(v1, v2) == 0
    assert '%.5f'%(kmeans.euclidiandistance(v1, v3)) == "4.89898"

def testmanhattan():
    v1 = [1, 2, 3, 4]
    v2 = v1
    v3 = [3, 2, 1, 0]
    kmeans = Kmeans("euclidian distance", 10, 3)
    assert kmeans.manhattandistance(v1, v2) == 0
    assert kmeans.manhattandistance(v1, v3) == 8
    
def testvectoradd():
    v1 = [1.5, 3.05, 3.23, 5.98]
    v2 = [9.1, 4.7, 1.5, 12.3]
    kmeans = Kmeans("euclidian distance", 10, 3)
    v3 = kmeans.vectadd(v1, v2)
    assert v3 == [10.6, 7.75, 4.73, 18.28]

"""Tests for getter, setter and help methods"""

def testgetfeatures():
    kmeans = Kmeans(10)
    newarr = kmeans.loadspecimensize()
    print(kmeans.features)
    print(kmeans.getfeatures())

"""Tests for biological database parsing"""
    
def testcsvparse():
    kmeans = Kmeans("euclidian distance", 10, 3)
    newlist = kmeans.listfromcsv("specimensizes.csv", [1])
    assert len(newlist) == 300
    
"""Tests for data normalisation and scaling methods."""

def testminmax():
    test = [[1.434, 5.217, 3.146], [7.777, 12.241, 2.672], [1.004, 2.535, 6.244], [1.111, 9.098, 4.601], [1.111, 2.222, 3.333], [5.606, 1.007, 2.500], [5.111, 2.234, 7.808], [6.080, 2.534, 7.187], [3.343, 6.091, 8.353]]
    kmeans = Kmeans("euclidian distance", 10, 3)
    minmax = kmeans.getminmax(test)
    assert minmax == [[1.004,7.777],[1.007,12.241],[2.500,8.353]]
    
"""Initialisation tests"""    
testinit()

"""Similarity metric tests"""
testeuclid()
testmanhattan()
testvectoradd()

"""Ease of use tests"""
testgetfeatures()

"""Biological database parsing tests"""
testcsvparse()

"""Normalisation and scaling test"""
testminmax()