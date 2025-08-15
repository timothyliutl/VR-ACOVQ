import TSVQ_Modules.COVQ as COVQ
import TSVQ_Modules.TSVQ as TSVQ
import TSVQ_Modules.Generalized_Classes as Generalized_Classes
import TSVQ_Modules.VariableRateQuantizer as VariableRateQuantizer

import numpy as np
from sklearn.cluster import KMeans
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math
import itertools
import copy

class Node():
    def __init__(self):
        self.children = []
        self.quantizer = None
        self.centroids = None
        self.bits = None
    
    def quantize_source(self):
        self.quantizer.centroids = self.centroids
        self.quantizer.nn_cond()
        self.quantizer.transmission()
        self.quantizer.transmission_fb()

        if len(self.children)>0:
            for i in range(2**self.bits):
                child = self.children[i]
                index = (self.quantizer.transmission_results_fb == i)
                print(child.bits, sum(index))
                if child.bits is not None and sum(index) > 0:
                    index_array = self.quantizer.index_array
                    channel_properties = self.quantizer.channel_properties
                    channel_properties_fb = self.quantizer.channel_properties_fb
                    hamming_dist_array = self.quantizer.append_error_array
                    hamming_dist_array_fb = self.quantizer.append_error_array_fb
                    child.quantizer = Generalized_Classes.LeafTSVQFB(self.quantizer.source, index, index_array, child.bits, channel_properties, channel_properties_fb, hamming_dist_array, hamming_dist_array_fb)
                    child.quantize_source()
                else:
                    child.posterior_source = self.quantizer.source[index]
        #recurisve function to quantize and pass all values into leaf nodes
    def get_children(self):
        return self.children

class TestVRQ():
    def __init__(self, channel_properties):
        self.root = Node()
        self.channel_properties_fb = {
            'epsilon': 0,
            'delta': 0,
            'memory': 1
        }
        self.channel_properties = channel_properties
        
    def copy_centroids(self, node, vrq_node):
        #function that copies centroids from the original VRQ object to new tree structure
        if vrq_node.quantizer is not None:
            node.centroids = vrq_node.quantizer.centroids.copy()
            node.bits = vrq_node.quantizer.bits
        if len(vrq_node.children)>0:
            for i in range(len(vrq_node.children)):
                node.children.append(Node())

            for i in range(len(vrq_node.children)):
                self.copy_centroids(node.children[i], vrq_node.children[i])
    
    def initialize_root_quantizer(self, num_samples, dimension, correlation, source = None):
        if source is not None:
            self.source = source
        else:
            self.source = COVQ.generate_gaussian_source(num_samples, dimension, correlation)

        self.root.quantizer = Generalized_Classes.RootCOVQFB(self.source, self.root.bits, self.channel_properties, self.channel_properties_fb)
    
    def find_leaves(self, node):
        leaves = []
        root = self.root
        if not node.get_children():
            leaves.append(node)
        else:
            for child in node.get_children():
                leaves.extend(self.find_leaves(child))
        return leaves


    def calc_distortion(self):
        leaf_list = self.find_leaves(self.root)
        distortion = 0
        count = 0 
        for leaf in leaf_list:
            if (leaf.bits == None) or (leaf.bits == 0):
                distortion = distortion + sum(np.linalg.norm(leaf.posterior_source - (leaf.posterior_source).mean(), axis=1)**2)
                count = count + len(leaf.posterior_source)
                print(sum(np.linalg.norm(leaf.posterior_source - (leaf.posterior_source).mean(), axis=1)**2))
            else:
                distortion = distortion + sum(leaf.quantizer.calc_distortion())
                print(sum(leaf.quantizer.calc_distortion()))
                count = count + len(leaf.quantizer.source)
        print('count', count)
        return distortion/(count* self.source.shape[1])