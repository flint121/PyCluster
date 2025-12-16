from collections.abc import Callable
from typing import Any, Literal
from numpy import random as RANDOM
from math import inf as INF
from math import sqrt as SQRT
from pandas import Index, read_csv as READCSV
from pandas import isnull as ISNULL


Mode = Literal["euclidian distance", "manhattan distance"]

class Cluster:

    def __init__(self) -> None:
        self.modes: dict[Mode, Callable[[Any, Any], float]] = {}
        self.mode = ""
        self.name = ""
        self.features = []
        self.featcount = 0
        # Params is a list containing sublists of length 3. Each sublist contains: the name of the parameter (e.g. "K value"), the data type of the input (can be of types int, float, string, or mode) and an associated message so that the user knows what to input, in that order.
        self.params: list[tuple[str, int | float | str | Mode, str]] = []
        # Scalers is a dictionary containing the available options for data scaling and normalisation.
        self.scalers = {"MinMax scaler": self.minmaxscaler}
        print("Call the 'gethelp()' function to get started.")

    def gethelp(self) -> None:
        """Help method for getting started"""
        print(
            "Getting started: Please use the modelist() method to see a list of similarity metrics available for this algorithm, then use the setmode() method to select a metric. \nYou may then run the clustering algorithm on a list containing your data or generate your own from a CSV file using the listfromcsv() method."
        )

    """Methods for comparing and performing mathematical operations on samples, which are treated as vectors."""

    def vectadd(self, p1, p2):
        """Combines the components of two samples using the vector addition technique."""
        for n in range(len(p2)):
            p1[n] = p1[n] + p2[n]
        return p1

    def euclidiandistance(self, p1, p2) -> float:
        """Returns the euclidian distance between two samples in the form of vectors."""
        sqreuc = 0
        for n in range(len(p1)):
            sqreuc = sqreuc + (p1[n] - p2[n]) ** 2
        return SQRT(sqreuc)

    def manhattandistance(self, p1, p2):
        """Returns the manhattan distance between two samples."""
        dist = 0
        for n in range(len(p1)):
            dist = dist + abs((p1[n] - p2[n]))
        return dist

    def sumsqrdist(self, data, results, centroids) -> list[int]:
        """Returns the sum of squared distance of samples in a cluster, intended for use in centroid based clustering algorithms"""
        sums = [0 for _ in range(len(centroids))]
        for n in range(len(data)):
            c = results[n] - 1
            sums[c] = sums[c] + self.euclidiandistance(data[n], centroids[c])
        return sums

    """Methods for handling data, reading CSV files and converting them into a compatible format."""

    def removecol(self, data, errindex):
        """Called when an erronious sample is found in dataset. Removes all items in the offending column and returns modified dataset"""
        for n in range(len(data)):
            del data[n][errindex]
        return data

    def getnumcols(self, filename) -> Index[str]:
        """Returns a list of the columns in a CSV file that have numeric data types. The user will then be asked which features they wish to use for clustering from this list"""
        numericcols = READCSV(filename, nrows=1).select_dtypes("number").columns
        return numericcols

    def checkfornan(self, filename):
        """After user has selected which features to use, this function will check the data in those columns in the CSV file for Nan values and inform the user of which features have been excluded due to containing nan values"""
        reader = READCSV(filename)
        usecolumns = reader.columns[reader.isna().any()].tolist()
        return usecolumns

    def listfromcsv(self, filename, usecolumns):
        """Reads a CSV file and creates a list containing the data in an indexable format."""
        data = []
        self.features = READCSV(filename, nrows=1, usecols=usecolumns).columns.to_list()
        self.featcount = len(self.features)
        reader = READCSV(filename, usecols=usecolumns)
        data = reader[self.features].values.tolist()
        return data

    def loadspecimensize(self):
        """Calls arrfromcsv with specific parameters for loading the specimensizes dataset."""
        data = self.listfromcsv("specimensizes.csv", [1])
        return data

    def loadplanktonsize(self):
        """Calls arrfromcsv with specific parameters for loading the planktonsizes dataset."""
        data = self.listfromcsv(
            "planktonsizes.csv", [2, 3, 5, 8, 9, 10, 12, 13, 14, 15, 18]
        )
        return data

    """Normalisation and scaling methods"""

    def getminmax(self, data) -> list[list[float]]:
        """Returns a list containing the minimum and maximum value of each feature in the dataset."""
        minmax = [
            [INF, -INF] for _ in range(len(data[0]))
        ]  # Initialise the list of min and max values for each feature.

        for i in range(len(minmax)):
            column = [x[i] for x in data]
            minmax[i][0] = min(column)
            minmax[i][1] = max(column)

        return minmax

    def minmaxscaler(self, data):
        minmaxdata = data.copy()
        minmax = self.getminmax(minmaxdata)
        for n in range(len(minmaxdata)):
            for i in range(len(minmaxdata[0])):
                toscale = (minmaxdata[n][i] - minmax[i][0]) / (
                    minmax[i][1] - minmax[i][0]
                )
                minmaxdata[n][
                    i
                ] = toscale  # * (minmax[i][1] - minmax[i][0]) + minmax[i][0]
        return minmaxdata

    """Getter and setter methods."""

    def getfeatures(self) -> None:
        for i in range(len(self.features)):
            print("Feature", i, ":", self.features[i])

    def getparams(self):
        return self.params

    def modelist(self) -> None:
        print(
            "Modes of operation: these are the similarity metrics available when using the",
            self.name,
            "algorithm. \nPlease use the key phrase provided as an argument for the method setmode() to choose a mode of operation. \nNote: some algorithms use a specific distanc metric, while some will have multiple modes of operation available",
        )
        for i in range(len(self.modes)):
            print("Mode:", self.modes.values(i), "key phrase:", self.modes.keys(i))

    def setmode(self, mode: str | Mode) -> None:
        if mode in self.modes:
            self.mode = mode


class Kmeans(Cluster):
    def __init__(self, mode: Mode ="euclidian distance", maxiter=10, k=3) -> None:
        super(Kmeans, self).__init__()
        self.name = "K means"
        self.modes = {
            "euclidian distance": self.euclidiandistance,
            "manhattan distance": self.manhattandistance,
        }
        if mode not in self.modes:
            self.mode = list(self.modes)[0]
        else:
            self.mode = mode
        self.params = [
            ("Similarity metric", "mode", "None"),
            (
                "K value",
                "int",
                "Enter a value of k to be used in clustering: \n Integer values greater than zero accepted \n Default value: 3",
            ),
            (
                "Maximum iterations",
                "int",
                "Enter a value for the maximum iterations to run before stopping automatically:\n Integer values greater than zero accepted \n Default value: 10",
            ),
        ]
        self.loops = 0
        self.maxiter = maxiter
        self.centroids = []
        self.k = k
        if k < 2:
            k = 2
            print(
                "Values of k < 2 cannot be used for clustering with K means algorithm. K has been set to its minimum value, 2"
            )

    """Four main methods for kmeans algorithm."""

    def run(self, data):
        """This function calls 3 other subroutines: init_centro, assign_clust and new_centro to perform k-means clustering on the given data array."""
        print(
            "Running kmeans with k =", self.k, "\nNumber of iterations =", self.maxiter
        )
        self.loops = 0
        prevcent = []
        self.centroids = []
        results = [-1 for _ in range(len(data))]

        self.centroids = self.initcentro(data)
        # Checks if the centroids in the previous 2 iterations are the same and if the maximum iteration value has been reached, otherwise continues kmeans algorithm.
        while self.centroids != prevcent and self.loops < self.maxiter:
            results = self.assigncluster(data, results)
            prevcent = self.centroids.copy()
            self.centroids = self.newcentro(data, results)
            self.loops += 1
        return results

    def initcentro(self, data):
        """Returns a list containing k initial centroids chosen randomly."""
        initcent = []
        while len(initcent) < self.k:
            centrind = RANDOM.randint(0, high=len(data))
            toappend = data[centrind].copy()
            if toappend not in initcent:
                initcent.append(toappend)
        return initcent

    def assigncluster(self, data, results):
        """Assigns each item in the dataset to a cluster, decided by the minimum distance from each point to any of the centroids."""
        for n in range(len(data)):
            mindist = INF
            for i in range(len(self.centroids)):
                curdist = self.modes[self.mode](
                    data[n].copy(), self.centroids[i].copy()
                )
                if curdist < mindist:
                    mindist = curdist
                    results[n] = i + 1
        return results

    def newcentro(self, data, results) -> list[list[int]]:
        """Finds the mean of each cluster and assigns new centroids based on results."""
        totals = [0 for _ in range(len(self.centroids))]
        newcent = [
            [0 for x in range(len(self.centroids[0]))]
            for y in range(len(self.centroids))
        ]
        for n in range(len(data)):
            """adds the vector at index n of the data array to the value at index c-1 of the current centroids array. this will give a total to be divided by the number of items to give the mean of that cluster."""
            c = (results[n]) - 1
            newcent[c] = self.vectadd(newcent[c].copy(), data[n].copy())
            totals[c] = totals[c] + 1

        for n in range(len(newcent)):
            for i in range(len(self.centroids[n])):
                try:
                    newcent[n][i] = newcent[n][i] / totals[n]
                except ZeroDivisionError:
                    newcent[n][i] = newcent[n][i]
        return newcent

    """Getter and setter methods."""

    def setk(self, k) -> None:
        """Allows the user to change the value of k before or after running the algorithm without creating a new object."""
        self.k = k

    def setmaxiter(self, maxiter) -> None:
        """Allows the user to change the value of maxiter to allow the kmeans algorithm to run for more than the default 10 iterations."""
        self.maxiter = maxiter

    def setall(self, paramlist) -> None:
        """All clustering algorithm classes will contain a setall method, which aids in dynamic instantiation by taking a list of parameters and calling all the setter modes for the class"""
        self.setmode(paramlist[0])
        self.setk(paramlist[1])
        self.setmaxiter(paramlist[2])

    """Evaluation methods"""

    def summary(self, data, results) -> None:
        """returns a brief summary of the clustering performed on the dataset, giving sum of squared distance and number of data items in each cluster"""
        totals = [0 for _ in range(len(self.centroids))]
        for n in range(len(data)):
            c = results[n] - 1
            totals[c] = totals[c] + 1
        sums = self.sumsqrdist(data, results, self.centroids)
        for n in range(len(self.centroids)):
            print("Cluster", n + 1, "contains", totals[n], "items.")
            print("Sum of squared distance for cluster", n + 1, ":", sums[n])
        print("Algorithm ran for", self.loops, "iterations.")

