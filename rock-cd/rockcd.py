import time, sys
import pandas as pd
import networkx as nx
from networkx.algorithms.community import modularity

from itertools import combinations
from sklearn.metrics import jaccard_score
import numpy as np

QUALITY_FUNCTIONS = ["modularity"]


class RockCD(object):
    def __init__(
        self, g: nx.Graph, attrs: dict, sim=0.05, quality="modularity", min_com=2
    ):
        """Initializes RockCD object

        :param g: a NetworkX Graph
        :param attrs: a dict having nodes as keys and dicts of the form {name : value} as value
        :param sim: similarity threshold, defaults to 0.05
        :type sim: float, optional
        :param quality: quality function to optimize, defaults to 'modularity'
        :type quality: str, optional
        :param min_com: minimun number of communities, defaults to 2
        :type min_com: int, optional
        :raises NetworkXNotImplemented: Only implemented for Graphs and DiGraphs
        """
        self.graph = g.copy()
        self.attrs = attrs
        self.sim = sim
        self.quality = quality
        self.MIN_COM = min_com

        self.matrix = None  # ROCK matrix

        self.__partition = dict()  # {c_id : [n0,n1,n2...]}
        self.__communities = []  # list of lists of nodes
        self.best_partition = dict()  # final partition
        self.communities = []  # final list of communities
        self.evaluation = None  # best quality function score

        if len(self.graph) == 0:
            raise nx.NetworkXPointlessConcept("The graph is null")
        if self.graph.is_directed():
            self.graph = self.graph.to_undirected()
        if not isinstance(self.graph, nx.Graph):
            class_name = self.graph.__class__.__name__.lower()
            raise NotImplementedError(f"Not implemented for {class_name}s")

        if self.sim < 0 or self.sim > 1:
            raise ValueError("sim parameter must be between 0 and 1")

        if self.quality not in QUALITY_FUNCTIONS:
            raise ValueError(
                f"quality parameter must be one of the following: {QUALITY_FUNCTIONS}"
            )

        self.setup()

        self.execute()

    def setup(self):
        adjacency_matrix = nx.to_pandas_adjacency(self.graph)

        # creates list of attributes for Jaccard

        df = pd.DataFrame.from_dict(self.attrs, orient='index')
        attributes = pd.get_dummies(df).values
        # computes attributes similarity matrix
        similarity_matrix = RockCD.__jaccard_attributes_similarity_matrix(adjacency_matrix, attributes)

        # computes initial ROCD matrix
        self.matrix = self.__get_ROCK_matrix(adjacency_matrix, similarity_matrix)

    def execute(self):
        """
        Executes ROCK-CD algorithm
        """

        while len(self.matrix) > self.MIN_COM:

            # computes fitness measure
            goodness = RockCD.__fitness_measure(self.matrix, self.sim)
            # updates matrix
            self.matrix = RockCD.__merge_and_update(goodness, self.matrix)
            # formats output
            self.__format_output()
            # optimizes quality function
            qual = self.__optimize_quality_function()

        self.communities = list(self.best_partition.values())

    def __jaccard_attributes_similarity_matrix(adjacency_matrix, data: list):
        sim_matrix = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range(len(data)):
                if i == j:
                    sim_matrix[i][j] = 1
                elif adjacency_matrix.iloc[i,j] == 1:
                    sim_matrix[i][j] = jaccard_score(data[i], data[j])
                else:
                    sim_matrix[i][j] = 0   

        return sim_matrix

    def __get_ROCK_matrix(self, adjacency_matrix, sim_matrix):
        _sim_matrix = sim_matrix.copy()
        for i in range(len(_sim_matrix)):
            for j in range(len(_sim_matrix)):
                if _sim_matrix[i][j] < self.sim:
                    _sim_matrix[i][j] = 0
                else:
                    _sim_matrix[i][j] = 1

        _sim_matrix = _sim_matrix @ _sim_matrix

        for i in range(len(_sim_matrix)):
            _sim_matrix[i][i] = 0

        ROCK_matrix = pd.DataFrame(_sim_matrix)

        # corrects jaccard by checking ground truth
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix)):
                if adjacency_matrix.iloc[i, j] == 0:
                    ROCK_matrix.iloc[i, j] = 0

        return ROCK_matrix

    def __fitness_measure(labeled_numLinks, threshold):
        pairwise_fitness = []
        # function_theta can be changed to fit data set of interest, below is the most commonly used one
        function_theta = (1.0 - threshold) / (1.0 + threshold)

        pairs = combinations(labeled_numLinks, 2)

        # calculate the fitness between all combinations of cluster pairs
        for elements in pairs:
            total_links_between_clusters = labeled_numLinks[elements[0]][elements[1]]
            bothClusters = (
                sum(labeled_numLinks[elements[0]] != 0)
                + sum(labeled_numLinks[elements[1]] != 0)
            ) ** (1 + 2 * (function_theta))
            firstCluster = sum(labeled_numLinks[elements[0]] != 0) ** (
                1 + 2 * (function_theta)
            )
            secondCluster = sum(labeled_numLinks[elements[1]] != 0) ** (
                1 + 2 * (function_theta)
            )
            totalDenominator = bothClusters - firstCluster - secondCluster

            try:
                fitnessMeasure = float(total_links_between_clusters) / float(
                    totalDenominator
                )
            except ZeroDivisionError:
                raise ZeroDivisionError("No similar clusters found. Tru lowering the sim parameter"
                ) #from None
                

            # (cluster number, cluster number, fitness between clusters)
            pairwise_fitness.append((elements[0], elements[1], (fitnessMeasure * 10)))

        # sorts list so first tuple is most fit cluster pair
        pairwise_fitness.sort(key=lambda tup: tup[2], reverse=True)
        return pairwise_fitness

    def __merge_and_update(pairwise_fitness, labeled_numLinks):

        # merges clusters
        for column in labeled_numLinks:
            labeled_numLinks[column] = (
                labeled_numLinks[column] + labeled_numLinks[pairwise_fitness[0][1]]
            )

        # relabels clusters post-merging
        labeled_numLinks = labeled_numLinks.drop(pairwise_fitness[0][1], axis=1)
        labeled_numLinks = labeled_numLinks.drop(pairwise_fitness[0][1])
        labeled_numLinks.rename(
            columns={
                pairwise_fitness[0][0]: str(pairwise_fitness[0][0])
                + ","
                + str(pairwise_fitness[0][1])
            },
            inplace=True,
        )
        labeled_numLinks.rename(
            index={
                pairwise_fitness[0][0]: str(pairwise_fitness[0][0])
                + ","
                + str(pairwise_fitness[0][1])
            },
            inplace=True,
        )

        return labeled_numLinks

    def __format_output(self):
        num_communities = len(self.matrix.columns)
        self.__partition = dict()

        # checks if nodes are ints or strings and updates partition dict
        if isinstance(list(self.graph.nodes())[0], int):
            for c in range(num_communities):
                com = set(map(int, str(self.matrix.columns[c]).split(",")))
                self.__partition[c] = com
        else:
            for c in range(num_communities):
                com = set(str(self.matrix.columns[c]).split(","))
                self.__partition[c] = com

        self.__communities = list(self.__partition.values())

    def __optimize_quality_function(self):
        score = 0
        if self.quality == "modularity":
            try:
                score = modularity(self.graph, self.__communities)
            except:
                print(self.matrix)

        # elif self.quality == 'other':
        #    score = other(self.graph, self.__communities)

        if self.evaluation is not None:
            if score > self.evaluation[self.quality]:
                self.evaluation[self.quality] = score
                self.best_partition = self.__partition

        else:
            self.evaluation = {self.quality: score}
            self.best_partition = self.__partition

