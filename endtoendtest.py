from cluster import Cluster, Kmeans
from time import time as TIME

def testkmeansspecimen(k):
    print("\nTesting kmeans on specimen sizes dataset.")
    kmeans = Kmeans("euclidian distance", 10, k)
    test = kmeans.loadspecimensize()
    start = TIME()
    results = kmeans.run(test)
    print("Time to run algorithm:", TIME() - start,"seconds")
    kmeans.summary(test, results)

def testkmeansplankton(k):
    print("\nTesting kmeans on plankton sizes dataset.")
    kmeans = Kmeans("euclidian distance", 10, k)
    test = kmeans.loadplanktonsize()
    start = TIME()
    results = kmeans.run(test)
    print("Time to run algorithm:", TIME() - start,"seconds")
    #print(kmeans.sumsqrdist(test, results, kmeans.centroids))
    #print(kmeans.centroids)
    kmeans.summary(test, results)
    with open('results.txt', 'w') as f:
            for n in range(len(results)):
                f.write(str(results[n]))
                f.write("\n")
                
def testkmeansplscaled(k):
    print("\nTesting kmeans on plankton sizes dataset, scaled with minmaxscaler.")
    kmeans = Kmeans("euclidian distance", 10, k)
    test = kmeans.loadplanktonsize()
    start = TIME()
    test = kmeans.minmaxscaler(test)
    results = kmeans.run(test)
    print("Time to run algorithm:", TIME() - start,"seconds")
    kmeans.summary(test, results)
    with open('scaledresults.txt', 'w') as f:
            for n in range(len(results)):
                f.write(str(results[n]))
                f.write("\n")


"""Kmeans class method tests"""
testkmeansspecimen(5)
testkmeansplankton(10)
testkmeansplscaled(10)