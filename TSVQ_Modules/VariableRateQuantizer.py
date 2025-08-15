import TSVQ_Modules.COVQ as COVQ
import TSVQ_Modules.TSVQ as TSVQ
import TSVQ_Modules.Generalized_Classes as Generalized_Classes

import numpy as np
from sklearn.cluster import KMeans
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math
import itertools
import copy




class Node():
    def __init__(self, bits, channel_properties, source, epsilon_array, parent):
        self.bits = bits
        self.children = []
        self.channel_properties = channel_properties
        self.source = source
        self.epsilon_array = epsilon_array
        self.parent = parent
        self.quantizer = None
        self.name = ''

    def is_leaf(self):
        return len(self.children) == 0

    def get_children(self):
        return self.children
        
class VariableRateQuantizer():
    def __init__(self, source, average_bits, channel_properties, epsilon_array = None, fixed_rate = False, memory_constrainted = True):
        noiseless_channel_properties = {
            'epsilon': 0,
            'delta': 0,
            'memory': 1
        }

        self.source = source
        self.average_bits = average_bits # array of contraint that is an array for average bit allocation for each stage
        self.channel_properties = channel_properties
        self.epsilon_array = epsilon_array
        self.fixed_rate = fixed_rate
        self.root = Node(average_bits[0], channel_properties, source, epsilon_array, None)
        self.root.quantizer = Generalized_Classes.RootCOVQFB(source, average_bits[0], channel_properties, noiseless_channel_properties, epsilon_array = epsilon_array)
        self.root.quantizer.train_quantizer()
        self.root.name = 'root'
        if len(average_bits) > 1:
            self.initialize_new_leafs(self.root)
        self.bit_allocation_history = []
        self.average_bit_history = [] #average storage complexity for each stage
        self.average_encoding_complexity = [] #average encoding complexity for each stage
        self.memory_constrainted = memory_constrainted #boolean which determines if the quantizer is constrained by storage complexity or by encoding complexity

        for i in range(1, len(average_bits)):
            leaf_list = self.find_leaves(self.root)
            if not self.fixed_rate:
                bit_allocation, average_encoding_complexity, average_bits_used = self.calculate_optimal_bit_allocation(average_bits[i], self.memory_constrainted)
                self.average_encoding_complexity.append(average_encoding_complexity)
            else:
                bit_allocation = np.ones(len(leaf_list))*average_bits[i]
                average_bits_used = average_bits[i]
            self.bit_allocation_history.append(bit_allocation)
            self.average_bit_history.append(average_bits_used)

            if i == len(average_bits) - 1:
                self.add_new_stage(bit_allocation, leaf_list, last_stage = True)
            else:
                self.add_new_stage(bit_allocation, leaf_list, last_stage = False)
        print('Distortion: ',self.calc_distortion())
    
    def initialize_new_leafs(self, node):
        bits = node.bits
        for i in range(2**bits):
            subset_index = (node.quantizer.transmission_results == i)
            index_array = node.quantizer.index_array
           
            hamming_dist_array = node.quantizer.append_error_array #append arrays are new hamming dist arrays for leaf quantizers
            hamming_dist_array_fb = node.quantizer.append_error_array_fb

            #bits to be determined by steepest descent algorithm
            new_node = Node(None, node.channel_properties, node.quantizer.source, node.epsilon_array, node)
            new_node.subset_index = subset_index
            new_node.index_array = index_array
            new_node.hamming_dist_array = hamming_dist_array  
            new_node.hamming_dist_array_fb = hamming_dist_array_fb
            new_node.name = node.name + f'_{i}'
            node.children.append(new_node)
    
    def add_new_stage(self, bit_allocation_list, leaf_list, last_stage = False):
        #assert length of bit_allocation_list is equal to the number of leaf nodes
        if (len(bit_allocation_list) != len(leaf_list)):
            raise ValueError("Length of bit_allocation_list must be equal to the number of leaf nodes")

        for index in range(len(bit_allocation_list)):
            noiseless_channel_properties = {
                        'epsilon': 0,
                        'delta': 0,
                        'memory': 1
                    }
            if bit_allocation_list[index]>0:
                subset_index = leaf_list[index].subset_index
                index_array = leaf_list[index].index_array
                hamming_dist_array = leaf_list[index].hamming_dist_array
                hamming_dist_array_fb = leaf_list[index].hamming_dist_array_fb
                source = leaf_list[index].source
                if sum(subset_index) > 2*(2**int(bit_allocation_list[index])):
                    bits = int(bit_allocation_list[index])        
                    leaf_list[index].quantizer = Generalized_Classes.LeafTSVQFB(source, subset_index, index_array, bits, self.channel_properties, noiseless_channel_properties, hamming_dist_array = hamming_dist_array,  hamming_dist_array_fb = hamming_dist_array_fb, epsilon_array = self.epsilon_array)
                    leaf_list[index].quantizer.train_quantizer()
                else:
                    bits = 0
                leaf_list[index].bits = bits
                if not last_stage and bits > 0:
                    self.initialize_new_leafs(leaf_list[index])

    def find_leaves(self,node):
        leaves = []
        if not node.get_children():
            leaves.append(node)
        else:
            for child in node.get_children():
                leaves.extend(self.find_leaves(child))
        return leaves
    
    def calculate_optimal_bit_allocation(self, stage_average_bits, memory_constrainted = True):

        stage_average_bits = stage_average_bits if memory_constrainted else 2**stage_average_bits

        leaf_list = self.find_leaves(self.root)
        optimal_bit_allocation = np.zeros(len(leaf_list), dtype = int)
        distortion_decreases = np.zeros(len(leaf_list))
        probability_array = np.zeros(len(leaf_list))
        for i in range(len(leaf_list)):
            probability_array[i] = sum(leaf_list[i].subset_index)/len(self.source)
        average_bits = 0

        noiseless_channel_properties = {
                        'epsilon': 0,
                        'delta': 0,
                        'memory': 1
                    }
        initial_distortion_array = np.zeros(len(leaf_list)) 
        new_distortion_array = np.zeros(len(leaf_list))
        max_bit_allocation = 8
        
        #populate initial distortion list
        for i in range(len(leaf_list)):
            initial_distortion_array[i] = sum(np.linalg.norm(leaf_list[i].source[leaf_list[i].subset_index] - (leaf_list[i].source[leaf_list[i].subset_index]).mean(), axis=1)**2)
        
        for i in range(len(leaf_list)):
            if sum(leaf_list[i].subset_index) < 2*(2**(max_bit_allocation+1)):
                initial_distortion_array[i] = 0
                new_distortion_array[i] = 1
                optimal_bit_allocation[i] = 0
                distortion_decreases[i] = -1
                #if the number of source values is small, discard the quantizer and do not allocate bits to the node
            else:
                new_quantizer = Generalized_Classes.LeafTSVQFB(leaf_list[i].source, leaf_list[i].subset_index, leaf_list[i].index_array, 1, self.channel_properties, noiseless_channel_properties, leaf_list[i].hamming_dist_array, leaf_list[i].hamming_dist_array_fb)
                new_quantizer.train_quantizer()
                new_distortion = sum(new_quantizer.calc_distortion())
                new_distortion_array[i] = new_distortion

        while True:
            for i in range(len(leaf_list)):

                if distortion_decreases[i] == -1:
                    continue

                if optimal_bit_allocation[i] > max_bit_allocation:
                    distortion_decreases[i] = -1
                    continue

                #non-normalized weighted expected distortion (not divided by the total number of samples in the original source)
                initial_distortion = initial_distortion_array[i] 
                new_distortion = new_distortion_array[i]
                cost_change  = probability_array[i] if memory_constrainted else ((2**(optimal_bit_allocation[i]+1) - 2**(optimal_bit_allocation[i]))*probability_array[i])
                #cost change is the change in the given constraint
                distortion_decreases[i] = (initial_distortion - new_distortion)/cost_change 

            max_index = np.argmax(distortion_decreases)

            initial_distortion_array[max_index] = new_distortion_array[max_index] #update distortion of quantizer with new bit
            optimal_bit_allocation[max_index] = optimal_bit_allocation[max_index] + 1
            average_bits = np.dot(optimal_bit_allocation, probability_array) if memory_constrainted else np.dot(2**optimal_bit_allocation, probability_array)

            if average_bits > stage_average_bits:
                #if exceeds revert last change
                optimal_bit_allocation[max_index] = optimal_bit_allocation[max_index] - 1
                distortion_decreases[max_index] = -1 #set decrease to negative so it cannot be considered for an increase
            else:
                new_quantizer = Generalized_Classes.LeafTSVQFB(leaf_list[max_index].source, leaf_list[max_index].subset_index, leaf_list[max_index].index_array, optimal_bit_allocation[max_index]+1, self.channel_properties, noiseless_channel_properties, leaf_list[max_index].hamming_dist_array, leaf_list[max_index].hamming_dist_array_fb)
                new_quantizer.train_quantizer()
                new_distortion_array[max_index] = sum(new_quantizer.calc_distortion())

            #ensuring that average bits does not exceed the constraint
            if average_bits == stage_average_bits:
                break
            
            if all(x < 0 for x in distortion_decreases):
                break

            print(distortion_decreases)
            print(initial_distortion_array)
            print(new_distortion_array)
            print(probability_array)
            print(optimal_bit_allocation)
            print(average_bits)
        return optimal_bit_allocation, np.dot(2**optimal_bit_allocation, probability_array), np.dot(optimal_bit_allocation, probability_array)
        #apply steepest descent algorithm to find best bit allocation

    def calc_distortion(self):
        #write for edge case where a leaf doesnt have any bits allocated
        leaf_list = self.find_leaves(self.root)
        distortion = 0
        count = 0 
        for leaf in leaf_list:
            if (leaf.bits == None) or (leaf.bits == 0):
                distortion = distortion + sum(np.linalg.norm(leaf.source[leaf.subset_index] - (leaf.source[leaf.subset_index]).mean(), axis=1)**2)
                count = count + len(leaf.source[leaf.subset_index])
            else:
                distortion = distortion + sum(leaf.quantizer.calc_distortion())
                count = count + len(leaf.quantizer.source)
        return distortion/(count* self.source.shape[1])

    def is_balanced(self):
        return all(np.unique(sub_arr).size == 1 for sub_arr in self.bit_allocation_history)