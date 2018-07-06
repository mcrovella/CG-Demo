import numpy as np
import numpy.ma as ma
import scipy as sp
from scipy import stats
import networkx as nx
import pandas as pd
import logging
import warnings

class CG_decomposition:

    def __init__(self,
                     network,
                     data_matrix,
                     column_names,
                     row_names,
                     normalization,
                     loglevel=logging.INFO):

        # for detailed output call with logging.DEBUG
        logging.basicConfig(level=loglevel)

        # normalize counts data
        if normalization == 'log+zscore':
            # ignore log of zeros
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                M_log = np.log(data_matrix)
            M_log_masked = ma.masked_invalid(M_log)
            M_normalized = stats.zscore(M_log_masked)
            normalized_matrix = M_normalized.filled(0.0)	
        elif normalization == 'normal_row+log+zscore':
            matrix = matrix * 1.0
            row_normalized_UMI = matrix / matrix.sum(axis = 1)[: , np.newaxis]
            # ignore log of zeros
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                M_log = np.log(row_normalized_UMI)
            M_log_masked = ma.masked_invalid(M_log)
            M_normalized = stats.zscore(M_log_masked)
            normalized_matrix = M_normalized.filled(0.0)
        else:
            raise ValueError('unrecognized normalization')

        # intialize internal variables
        self.X = pd.DataFrame(data = normalized_matrix,
                                  index = row_names,
                                  columns = column_names)
        self.S = self.X.T.dot(self.X)
        self.M = network
        self.node_norms = self.X.apply(lambda x: np.linalg.norm(x), axis=0)

    def __update_node_norms(self, node_list):
        for i in node_list:
            self.node_norms[i] = np.linalg.norm(self.X[i])
        return

    def __average_column(self, nodes):
        return self.X[nodes].mean(axis=1)

    # given a new c and g, recompute S = X'.T @ X' in which X' = X-cg
    def __updateS(self, c, g):
        v = self.X.T.dot(c)
        w = pd.Series(np.zeros(self.X.shape[1]), index=self.X.columns)
        w[g] = c.T.dot(c)
        for i in g:
            self.S.loc[:,i] = self.S.loc[:,i] - v + w
            self.S.loc[i,:] = self.S.loc[i,:] - v 
        return 

    # create an outer product of c and g as a dataframe
    # not particularly efficient
    def __outer(self, c, g):
        outer = pd.DataFrame(np.zeros(self.X.shape),
                                 index = self.X.index,
                                 columns = self.X.columns)
        for s in g:
            outer[s] = c
        return outer

    # starting from a given seed, find the connected subnetwork of size
    # no greater than max_size, that most lowers the error X-cg 
    def __generate_candidate_graph(self, seed, max_size):
        logging.debug('seed: {}'.format(seed))
        subnetwork = [seed]
        SS = self.S.loc[seed,seed]
        decrements = [np.sqrt(SS)]
        while len(subnetwork) < max_size:
            # find the nodes that are adjacent to current network f
            boundary_nodes = nx.node_boundary(self.M, subnetwork)
            if not boundary_nodes:
                logging.debug('No further connected nodes')
                break
            maxgain = -np.inf
            i = len(subnetwork)
            # for each possible node to add to subnetwork
            for v in boundary_nodes:
                # compute the decrease in SSE due to addition of this node
                # for explanation of this computation see documentation
                SStrial = (SS
                               + 2 * np.sum(self.S.loc[subnetwork, v])
                               + self.S.loc[v, v])
                # how much would addition of this node improve the avg error?
                SStest = SStrial/(i+1) - SS/i
                if SStest > maxgain:
                    maxgain = SStest
                    bestSS = SStrial
                    best = v
            if maxgain < 0.0:
                # no adjacent nodes improve the error
                logging.debug('No further improvement possible.')
                break
            # add the node that improves error most to the growing subnetwork
            SS = bestSS
            subnetwork.append(best)
            decrements.append(np.sqrt(maxgain))
            logging.debug('subnetwork size {} -- added {}, improvement {}'.format(
                len(subnetwork), best, np.sqrt(maxgain)))
        c = self.__average_column(subnetwork)
        error = np.linalg.norm(self.X - self.__outer(c, subnetwork))
        return subnetwork, error, decrements

    # main routine; perform decomposition
    def decom(self, rank, s, max_subnetwork_size):
        logging.info("number_of_candidates: {}".format(s))
        logging.info("max_subnetwork_size: {}".format(max_subnetwork_size))
        m, n = self.X.shape
        C = pd.DataFrame(np.zeros((m,rank)), index=self.X.index)
        G = pd.DataFrame(np.zeros((rank,n)), columns=self.X.columns)
        residual_errors = [np.linalg.norm(self.X)]
        subnetwork_decrements = []
        subnetworks = []
        for r in range(rank):
            logging.debug('rank: {}'.format(r))
            seeds = sorted(self.M.nodes(),
                               key= lambda v: self.node_norms[v],
                               reverse = True)[0:s]
            candidates = [self.__generate_candidate_graph(
                seed, max_subnetwork_size) for seed in seeds]
            best_candidate = min(candidates, key = lambda cand:cand[1])
            subnetwork = best_candidate[0]
            subnetworks.append(subnetwork)
            subnetwork_decrements.append(best_candidate[2])
            logging.info('network {} is {}'.format(r,subnetwork))
            G.loc[r, subnetwork] = 1.0
            c = self.__average_column(subnetwork)
            C.loc[r] = c
            self.__updateS(c, subnetwork)
            self.X = self.X - self.__outer(c, subnetwork)
            self.__update_node_norms(subnetwork)
            residual_errors.append(np.linalg.norm(self.X))
        return C, G, subnetworks, self.X, residual_errors, subnetwork_decrements


