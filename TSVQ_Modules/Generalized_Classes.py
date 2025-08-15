import TSVQ_Modules.COVQ as COVQ
import TSVQ_Modules.TSVQ as TSVQ
import TSVQ_Modules.MAP_Decoding as MAP
import numpy as np
from sklearn.cluster import KMeans
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math
import itertools
import copy

class WrapperTSVQ():
    def __init__(self, num_samples, correlation, bit_allocation, channel_properties, dimension, prev_centroids = None, tsvq = False, prev_tsvq_centroids = None ) -> None:
        print(num_samples, dimension, correlation)
        self.samples = COVQ.generate_gaussian_source(num_samples, dimension, correlation)
        self.quantizer_list = {}
        self.bit_allocation = bit_allocation
        self.channel_properties = channel_properties
        self.total_distortion = 0
        self.total_points = 0
        self.dimension = dimension  
        self.prev_centroids = prev_centroids #dictionary with centroids from previous quantizer, key is integer that maps to the stage with array of centroid arrays 
        self.tsvq = tsvq #code to switch from a tsvq and an acovq
        self.prev_tsvq_centroids = prev_tsvq_centroids

    def train_quantizers(self):
        for bit_index in range(len(self.bit_allocation)): #bit index synonymous with stage
            if bit_index == 0:
                print('training root quantizer')
                self.quantizer_list['root'] = RootCOVQ(self.samples, self.bit_allocation[bit_index], self.channel_properties)
                if self.prev_centroids:
                    self.quantizer_list['root'].prev_centroids = (self.prev_centroids[0])[0]
                self.quantizer_list['root'].train_quantizer()
            if bit_index == 1:
                print('training leaf quantizer')
                total_distortion = 0
                total_samples = 0
                for i in range(2**self.bit_allocation[0]):
                    index_string = COVQ.convert_to_binary(i, self.bit_allocation[0])
                    if(np.sum(self.quantizer_list['root'].feedback == i) != 0):
                        source = self.quantizer_list['root'].error_source if self.tsvq else self.quantizer_list['root'].source
                        self.quantizer_list[index_string] = LeafTSVQ(source, (self.quantizer_list['root'].feedback == i), self.bit_allocation[1], self.channel_properties, self.quantizer_list['root'].hamming_dist_array)
                        if self.prev_centroids:
                            print('adding initial centroids')
                            self.quantizer_list[index_string].select_initialization_candidate(self.prev_centroids[bit_index])
                        if (self.prev_centroids == None) and self.prev_tsvq_centroids and not self.tsvq:
                                prev_acovq_centroid = self.quantizer_list['root'].centroids[i]
                                print(self.prev_tsvq_centroids[index_string], prev_acovq_centroid)
                                print('TESTING ADD TO CODEBOOK')
                                print(TSVQ.add_2_codebook(self.prev_tsvq_centroids[index_string], prev_acovq_centroid))
                                self.quantizer_list[index_string].prev_centroids = TSVQ.add_2_codebook(self.prev_tsvq_centroids[index_string], prev_acovq_centroid) 
                        self.quantizer_list[index_string].train_quantizer()
            if bit_index > 1 and bit_index < len(self.bit_allocation):
                for prev_bit_index in range(2**sum(self.bit_allocation[0:bit_index-1])):
                    for current_bit_index in range(2**self.bit_allocation[bit_index-1]):
                        prev_index_string = COVQ.convert_to_binary(prev_bit_index, sum(self.bit_allocation[0:bit_index-1]))
                        curr_index_string = COVQ.convert_to_binary(current_bit_index, self.bit_allocation[bit_index-1])
                        #print(prev_index_string, 'index string')
                        #print('prev num of bits at this stage', sum(self.bit_allocation[0:bit_index]))
                        if (prev_index_string in self.quantizer_list.keys()):
                            prev_quantizer = self.quantizer_list[prev_index_string]
                            prev_quantizer.leaf = False
                            if (np.sum(prev_quantizer.feedback_results == current_bit_index)!=0):
                                source = prev_quantizer.error_source if self.tsvq else prev_quantizer.source
                                self.quantizer_list[prev_index_string + curr_index_string] = LeafTSVQ(source, (prev_quantizer.feedback_results == current_bit_index), self.bit_allocation[bit_index], self.channel_properties, prev_quantizer.append_error_array)
                                if self.prev_centroids:
                                    print('adding initial centroids')
                                    self.quantizer_list[prev_index_string + curr_index_string].select_initialization_candidate(self.prev_centroids[bit_index])
                                if (self.prev_centroids == None) and self.prev_tsvq_centroids and not self.tsvq:
                                    prev_acovq_centroid = self.quantizer_list[prev_index_string].centroids[current_bit_index]
                                    self.quantizer_list[prev_index_string + curr_index_string].prev_centroids = TSVQ.add_2_codebook(self.prev_tsvq_centroids[prev_index_string + curr_index_string], prev_acovq_centroid) 
                                self.quantizer_list[prev_index_string + curr_index_string].train_quantizer()
                        
        if(len(self.bit_allocation)>1):
            for i in range(2**sum(self.bit_allocation[0:len(self.bit_allocation)-1])):
                prev_index_string = COVQ.convert_to_binary(i, sum(self.bit_allocation[0:len(self.bit_allocation)-1]))
                if (prev_index_string in self.quantizer_list.keys()):
                    self.total_distortion += sum(self.quantizer_list[prev_index_string].distortion_mapping)
                    self.total_points += len(self.quantizer_list[prev_index_string].distortion_mapping)
        else:
            self.total_distortion = sum(self.quantizer_list['root'].distortion_mapping)
            self.total_points = len(self.quantizer_list['root'].distortion_mapping)

        print('done')
        print(self.quantizer_list.keys(), 'keys')
        print('dimension', self.samples.shape[1])
        print('total samples', self.total_points)
        #print('total distortion', -10*math.log10(self.total_distortion/(self.total_points*self.samples.shape[1])))
    
    def calc_distortion(self):
        return -10*math.log10(self.total_distortion/(self.total_points*self.samples.shape[1]))
    
    def calc_distortion_progression(self):
        #function to calculate the distortion after each stage
        pass
    
    def export_centroids_stage(self):
        #function to export dictionary of centroids by stage
        exported_centroids = {i: [] for i in range(len(self.bit_allocation))}
        exported_centroids[0].append(self.quantizer_list['root'].centroids)
        for stage in range(len(self.bit_allocation)-1):
            for bit_index in range(2**sum(self.bit_allocation[0:stage+1])):
                index_string = COVQ.convert_to_binary(bit_index, sum(self.bit_allocation[0:stage+1]))
                exported_centroids[stage+1].append(self.quantizer_list[index_string].centroids)
        return exported_centroids

    def export_initialization_stage(self):
        #function to export dictionary of centroids by stage
        exported_centroids = {i: [] for i in range(len(self.bit_allocation))}
        exported_centroids[0].append(self.quantizer_list['root'].initialization_centroids)
        for stage in range(len(self.bit_allocation)-1):
            for bit_index in range(2**sum(self.bit_allocation[0:stage+1])):
                index_string = COVQ.convert_to_binary(bit_index, sum(self.bit_allocation[0:stage+1]))
                exported_centroids[stage+1].append(self.quantizer_list[index_string].initialization_centroids)
        return exported_centroids

    def export_initialization_index(self):
        exported_centroids = {}
        exported_centroids['root'] = self.quantizer_list['root'].initialization_centroids
        for stage in range(len(self.bit_allocation)-1):
            for bit_index in range(2**sum(self.bit_allocation[0:stage+1])):
                index_string = COVQ.convert_to_binary(bit_index, sum(self.bit_allocation[0:stage+1]))
                exported_centroids[index_string] = copy.deepcopy(self.quantizer_list[index_string].initialization_centroids)
        return exported_centroids

    def export_centroids_index(self):
        exported_centroids = {}
        exported_centroids['root'] = self.quantizer_list['root'].centroids
        for stage in range(len(self.bit_allocation)-1):
            for bit_index in range(2**sum(self.bit_allocation[0:stage+1])):
                index_string = COVQ.convert_to_binary(bit_index, sum(self.bit_allocation[0:stage+1]))
                exported_centroids[index_string] = copy.deepcopy(self.quantizer_list[index_string].centroids)
        return exported_centroids
    


    def test_quantizers(self, num_samples, dimension, correlation):
        #function to test quantization using dataset seperate from that used in the training sequence
        #all that is needed is the centroids, channel properties, and the nn condition function
        # also need to simulate the channel
        threadsperblock = 1024
        blockspergrid = math.ceil(num_samples / threadsperblock)

        self.test_source = COVQ.generate_gaussian_source(num_samples, dimension, correlation)
        self.test_source_at_stage = {i: np.zeros(self.test_source.shape) for i in range(len(self.bit_allocation))}
        self.test_quantizer_nn_cond = {i: np.zeros(len(self.test_source), dtype=int) for i in range(len(self.bit_allocation))}
        self.test_distortion = {i: np.zeros(self.test_source.shape[0]) for i in range(len(self.bit_allocation))}
        self.test_feedback = {i: np.zeros(len(self.test_source), dtype=int) for i in range(len(self.bit_allocation))}
        self.test_num_errors = {i: np.zeros(len(self.test_source), dtype=int) for i in range(len(self.bit_allocation))}

        self.test_hamming_dist_array = {i: (np.zeros((self.test_source.shape[0], sum(self.bit_allocation[0:i+1])), dtype=int)) for i in range(len(self.bit_allocation))}
        #dictionary contains feedback, quantization, and source at each stage

        for bit_index in range(len(self.bit_allocation)):
            threadsperblock_2d = (16, 16)
            blockspergrid_x = math.ceil(num_samples / threadsperblock_2d[0])
            blockspergrid_y = math.ceil(self.bit_allocation[bit_index] / threadsperblock_2d[1])
            blockspergrid_2d = (blockspergrid_x, blockspergrid_y)

            if bit_index == 0:
                COVQ.nn_cond_vectors_gpu[blockspergrid, threadsperblock](self.test_source, self.quantizer_list['root'].centroids, self.bit_allocation[0], self.quantizer_list['root'].transition_matrix, self.test_quantizer_nn_cond[0], self.test_distortion[0])
                binary_array = np.zeros((len(self.test_source), self.bit_allocation[bit_index]), dtype=int)
                binary_array_feedback = np.zeros((len(self.test_source), self.bit_allocation[bit_index]), dtype=int)

                #simulating channel
                rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=np.random.randint(1000))
                COVQ.simulate_channel[blockspergrid, threadsperblock](self.test_quantizer_nn_cond[0], self.quantizer_list['root'].transition_matrix, self.test_feedback[0], rng_states)
                
                #calculating error and hamming distance from quantization and feedback
                TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.test_quantizer_nn_cond[0], self.bit_allocation[0], binary_array)
                TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.test_feedback[0], self.bit_allocation[0], binary_array_feedback)
                TSVQ.find_hamming_distance[blockspergrid_2d, threadsperblock_2d](self.bit_allocation[0], binary_array, binary_array_feedback, self.test_hamming_dist_array[0])
                TSVQ.find_num_errors[blockspergrid, threadsperblock](self.bit_allocation[0], self.test_hamming_dist_array[0], self.channel_properties['memory'], self.test_num_errors[0])
                TSVQ.calculate_error[blockspergrid, threadsperblock](self.test_source, self.test_feedback[0], self.quantizer_list['root'].centroids, self.test_source_at_stage[0])

            else:
                combination_list = []
                for sub_index in range(bit_index):
                    combination_list.append([j for j in range(2**self.bit_allocation[sub_index])])
                combinations = list(itertools.product(*combination_list))

                for tuple_value in combinations:
                    current_quantization = np.ones(self.test_quantizer_nn_cond[0].shape)
                    quantizer_string = ''
                    for stage_index in range(len(tuple_value)):
                        quantizer_string = quantizer_string + COVQ.convert_to_binary(tuple_value[stage_index], self.bit_allocation[stage_index])
                        current_quantization = np.logical_and(current_quantization, self.test_quantizer_nn_cond[stage_index] == tuple_value[stage_index])
                        print(np.sum(current_quantization),tuple_value[stage_index], np.sum(self.test_quantizer_nn_cond[stage_index] == tuple_value[stage_index]))
                    current_source = self.test_source_at_stage[bit_index-1][current_quantization]
                    current_errors = self.test_num_errors[bit_index-1][current_quantization]

                    temp_nn_cond_array = np.zeros(current_source.shape[0], dtype=int)
                    temp_distortion_array = np.zeros(current_source.shape[0])
                    TSVQ.nn_cond_vectors_gpu[blockspergrid, threadsperblock](current_source, self.quantizer_list[quantizer_string].centroids, self.bit_allocation[bit_index], self.quantizer_list[quantizer_string].transition_matrix, current_errors, temp_nn_cond_array, temp_distortion_array)
                    print(temp_nn_cond_array.shape, self.test_quantizer_nn_cond[bit_index][current_quantization].shape)
                    self.test_quantizer_nn_cond[bit_index][current_quantization] = temp_nn_cond_array
                    self.test_distortion[bit_index][current_quantization] = temp_distortion_array
                    print(temp_distortion_array[0:10])
                    print(np.sum(current_quantization), tuple_value)

                    binary_array = np.zeros((len(current_source), self.bit_allocation[bit_index]), dtype=int)
                    binary_array_feedback = np.zeros((len(current_source), self.bit_allocation[bit_index]), dtype=int)
                    
                    temp_feedback = np.zeros(temp_nn_cond_array.shape, dtype=int)
                    rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=np.random.randint(1000))
                    TSVQ.simulate_channel[blockspergrid, threadsperblock](self.test_quantizer_nn_cond[bit_index][current_quantization], self.quantizer_list[quantizer_string].transition_matrix, current_errors, temp_feedback, rng_states)
                    self.test_feedback[bit_index][current_quantization] = temp_feedback
                    
                    TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.test_quantizer_nn_cond[bit_index][current_quantization], self.bit_allocation[bit_index], binary_array)
                    TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.test_feedback[bit_index][current_quantization], self.bit_allocation[bit_index], binary_array_feedback)

                    current_hamming_dist_array = np.zeros((len(current_source), self.bit_allocation[bit_index]), dtype=int)
                    TSVQ.find_hamming_distance[blockspergrid_2d, threadsperblock_2d](self.bit_allocation[bit_index], binary_array, binary_array_feedback, current_hamming_dist_array)
                    
                    self.test_hamming_dist_array[bit_index][current_quantization] = np.concatenate((self.test_hamming_dist_array[bit_index-1][current_quantization], current_hamming_dist_array), axis=1)
                    
                    temp_array = np.zeros(self.test_num_errors[bit_index][current_quantization].shape, dtype=int)
                    TSVQ.find_num_errors[blockspergrid, threadsperblock](self.bit_allocation[bit_index], self.test_hamming_dist_array[bit_index][current_quantization], self.channel_properties['memory'], self.test_num_errors[bit_index][current_quantization], temp_array)
                    self.test_num_errors[bit_index][current_quantization] = temp_array

                    temp_error_array = np.zeros(current_source.shape)
                    TSVQ.calculate_error[blockspergrid, threadsperblock](current_source, self.test_feedback[bit_index][current_quantization], self.quantizer_list[quantizer_string].centroids, temp_error_array)
                    self.test_source_at_stage[bit_index][current_quantization] = temp_error_array
                

class RootCOVQ():
    def __init__(self, samples, bits, channel_properties, prev_centroids = None ) -> None:
        self.source = samples
        self.bits = bits
        self.nn_cond_vectors = np.zeros(samples.shape[0], dtype=int)
        self.distortion_mapping = np.zeros(samples.shape[0])
        self.centroids = np.zeros((2**bits, samples.shape[1]))
        self.feedback = np.zeros(samples.shape[0], dtype=int) #simulation results
        self.transition_matrix = COVQ.create_transition_matrix(bits, channel_properties['memory'], channel_properties['epsilon'], channel_properties['delta'], '', '')
        self.prev_centroids = prev_centroids
        self.memory = channel_properties['memory']
        self.hamming_dist_array = np.zeros((len(self.source), self.bits), dtype=int)
        self.error_source = np.zeros(samples.shape)
        self.leaf = False
        self.initialization_centroids = None
        self.array_index = np.array(range(len(samples)))
   
    def train_quantizer(self):
        if(self.prev_centroids is None):
            print('generating initial centroids for root')
            kmeans = KMeans(n_clusters=2**self.bits, n_init="auto", random_state = 0).fit(self.source)
            self.centroids = kmeans.cluster_centers_
            self.initialization_centroids = kmeans.cluster_centers_
        else:
            self.centroids = self.prev_centroids
            self.initialization_centroids = self.prev_centroids
        

        distortion1 = 10
        distortion2 = 10000

        threadsperblock = 1024
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)
        #training quantizer

        while(abs(distortion2-distortion1)/distortion2 > 0.001):
            COVQ.nn_cond_vectors_gpu[blockspergrid, threadsperblock](self.source, self.centroids, self.bits, self.transition_matrix, self.nn_cond_vectors, self.distortion_mapping)
            self.centroids = COVQ.centroid_cond_vectors(self.source, self.nn_cond_vectors, self.bits, self.source.shape[1], self.transition_matrix)
            #updating distortion
            distortion1 = distortion2
            distortion2 = self.distortion_mapping.mean()/self.source.shape[1]
            print("distortion", distortion2)
        
        #simulate channel
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=np.random.randint(1000))
        COVQ.simulate_channel[blockspergrid, threadsperblock](self.nn_cond_vectors, self.transition_matrix, self.feedback, rng_states)

        binary_array = np.zeros((len(self.source), self.bits), dtype=int)
        binary_array_feedback = np.zeros((len(self.source), self.bits), dtype=int)

        threadsperblock_2d = (16, 16)
        blockspergrid_x = math.ceil(len(self.source) / threadsperblock_2d[0])
        blockspergrid_y = math.ceil(self.bits / threadsperblock_2d[1])
        blockspergrid_2d = (blockspergrid_x, blockspergrid_y)

        TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.nn_cond_vectors, self.bits, binary_array)
        TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.feedback, self.bits, binary_array_feedback)
        TSVQ.find_hamming_distance[blockspergrid_2d, threadsperblock_2d](self.bits, binary_array, binary_array_feedback, self.hamming_dist_array)
        #TSVQ.find_num_errors[blockspergrid, threadsperblock](self.bits, self.hamming_dist_array, self.memory, self.num_errors)

        TSVQ.calculate_error[blockspergrid, threadsperblock](self.source, self.feedback, self.centroids, self.error_source)
        print('finished training first quantizer')


class LeafTSVQ():
    def __init__(self, source, subset_index, bits, channel_properties, hamming_dist_array, prev_centroids = None ) -> None:
        self.total_source = source
        self.subset_index = subset_index

        self.source = source[subset_index]
        self.bits = bits
        self.nn_cond_vectors = np.zeros(self.source.shape[0], dtype=int)
        self.distortion_mapping = np.zeros(self.source.shape[0])
        self.centroids = np.zeros((2**bits, self.source.shape[1]))

        self.hamming_dist_array = hamming_dist_array[subset_index] #just an array, append 

        self.transition_matrix = TSVQ.create_transition_matrix(bits, min(hamming_dist_array.shape[1],channel_properties['memory']), channel_properties['epsilon'], channel_properties['delta'])
        self.prev_centroids = prev_centroids
        self.num_errors_array = np.zeros(self.source.shape[0], dtype=int)
        self.memory = channel_properties['memory']
        self.feedback_results = np.zeros(len(self.source))

        self.append_error_array = None
        self.error_source = np.zeros(self.source.shape)
        self.leaf = True
        self.initialization_centroids = None

    
    def select_initialization_candidate(self, initialization_candidates):
        #given an array of candidate centroids, this function selects the centroids that minimizes the expected distortion for the source
        min_distortion = 1000000
        min_centroids = None
        threadsperblock = 1024
        #using noiseless transition matrix because initial codebooks from before were optimized for noiseless environment, this will prevent duplicate issue
        noiseless_transition_matrix = TSVQ.create_transition_matrix(self.bits, 1, 0, 0)
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)
        for centroid_candidate in initialization_candidates:
            temp = np.zeros(self.source.shape[0], dtype=int)
            distortion_mapping = np.zeros(self.source.shape[0])
            TSVQ.nn_cond_vectors_gpu[blockspergrid, threadsperblock](self.source, centroid_candidate, self.bits, noiseless_transition_matrix, self.num_errors_array, temp, distortion_mapping)
            distortion_current = sum(distortion_mapping)
            if distortion_current < min_distortion:
                min_centroids = centroid_candidate
                min_distortion = distortion_current
                print(distortion_current, min_centroids)
        self.prev_centroids = min_centroids #assign candidate that has the smallest expected distortion for the source as the initialized centroids


    def train_quantizer(self): #only used to simulate channel and get errors
        threadsperblock = 1024
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)

        print(len(self.hamming_dist_array), len(self.source))
        TSVQ.find_num_errors[math.ceil(len(self.hamming_dist_array) / threadsperblock), threadsperblock](self.hamming_dist_array.shape[1], self.hamming_dist_array, self.memory, self.num_errors_array)
        #finding number of errors
        print(self.num_errors_array[0:20]) #need to convert this back to a decimal number

        if((self.prev_centroids) is None):
            print('generating initial centroids')
            kmeans = KMeans(n_clusters=2**self.bits, n_init="auto", random_state = 0).fit(self.source)
            self.centroids = kmeans.cluster_centers_
            self.initialization_centroids = kmeans.cluster_centers_
        else:
            print('importing centroids')
            self.centroids = self.prev_centroids
            self.initialization_centroids = self.prev_centroids

        distortion1 = 10
        distortion2 = 10000


        while(abs(distortion1-distortion2)/distortion2 > 0.001):
            TSVQ.nn_cond_vectors_gpu[blockspergrid, threadsperblock](self.source, self.centroids, self.bits, self.transition_matrix, self.num_errors_array, self.nn_cond_vectors, self.distortion_mapping)
            #print(self.source.shape, self.nn_cond_vectors.shape, self.bits, self.source.shape[1], self.transition_matrix.shape)
            self.centroids = TSVQ.centroid_cond_vectors(self.source, self.nn_cond_vectors, self.num_errors_array, self.bits, self.source.shape[1], self.transition_matrix)
            print('distortion:', distortion2)
            distortion1 = distortion2
            distortion2 = np.sum(self.distortion_mapping)/(self.source.shape[1]*len(self.source))
        #code to simulate the channel
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=np.random.randint(1000))
        TSVQ.simulate_channel[blockspergrid, threadsperblock](self.nn_cond_vectors, self.transition_matrix, self.num_errors_array, self.feedback_results, rng_states)

        binary_array = np.zeros((len(self.source), self.bits), dtype=int)
        binary_array_feedback = np.zeros((len(self.source), self.bits), dtype=int)

        threadsperblock_2d = (64, 16)
        blockspergrid_x = math.ceil(len(self.source) / threadsperblock_2d[0])
        blockspergrid_y = math.ceil(self.bits / threadsperblock_2d[1])
        blockspergrid_2d = (blockspergrid_x, blockspergrid_y)

        TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.nn_cond_vectors, self.bits, binary_array)
        TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.feedback_results, self.bits, binary_array_feedback)
        
        current_hamming_dist_array = np.zeros((len(self.source), self.bits), dtype=int)
        TSVQ.find_hamming_distance[blockspergrid_2d, threadsperblock_2d](self.bits, binary_array, binary_array_feedback, current_hamming_dist_array)

        self.append_error_array = np.concatenate((self.hamming_dist_array, current_hamming_dist_array), axis=1)
        # print(self.hamming_dist_array[0:10])
        # print(current_hamming_dist_array[0:10])
        # print(self.append_error_array[0:10])
        # TODO: need to replace centroids and feedback results, error source calculations does not need quantized indicies
        # can create for loop that seperates source with corresponding encoders
        TSVQ.calculate_error[blockspergrid, threadsperblock](self.source, self.feedback_results, self.centroids, self.error_source)

        #TODO: add way to get difference between quantized and source values
        print('code done') 


class WrapperTSVQNoisyFB():
    #only going to be using acovq
    def __init__(self, num_samples, correlation, bit_allocation, channel_properties, channel_properties_fb, dimension, map_decode = False, training_channel_properties = None, prev_centroids = None, epsilon_array = None, ria = False) -> None:
        print(num_samples, dimension, correlation)
        self.samples = COVQ.generate_gaussian_source(num_samples, dimension, correlation)
        self.quantizer_list = {}
        self.bit_allocation = bit_allocation
        self.channel_properties = channel_properties
        self.channel_properties_fb = channel_properties_fb
        self.training_channel_properties = training_channel_properties
        self.total_distortion = 0
        self.total_points = 0
        self.dimension = dimension  
        self.prev_centroids = prev_centroids #dictionary with centroids from previous quantizer, key is integer that maps to the stage with array of centroid arrays 

        self.map_decode = map_decode #boolean that says whether quantizer is going to use map decoding

        self.epsilon_array = epsilon_array #array of epsilon values for increase decrease method
        self.ria = ria #boolean that states whether the transition matrix will use random indexing

    def train_quantizers(self):
        for bit_index in range(len(self.bit_allocation)): #bit index synonymous with stage
            if bit_index == 0:
                print('training root quantizer')
                self.quantizer_list['root'] = RootCOVQFB(self.samples, self.bit_allocation[bit_index], self.channel_properties, self.channel_properties_fb, training_channel_properties = self.training_channel_properties, map_decode = self.map_decode, epsilon_array = self.epsilon_array, ria = self.ria)
                if self.prev_centroids:
                    self.quantizer_list['root'].prev_centroids = self.prev_centroids['root']
                self.quantizer_list['root'].train_quantizer()
            if bit_index == 1:
                print('training leaf quantizer')
                total_distortion = 0
                total_samples = 0
                for i in range(2**self.bit_allocation[0]):
                    index_string = COVQ.convert_to_binary(i, self.bit_allocation[0])
                    received_subset = (self.quantizer_list['root'].map_array == i) if self.map_decode else (self.quantizer_list['root'].transmission_results_fb == i)
                    if(np.sum(received_subset) != 0):
                        source = self.quantizer_list['root'].source
                        self.quantizer_list[index_string] = LeafTSVQFB(source, received_subset, self.quantizer_list['root'].index_array, self.bit_allocation[1], self.channel_properties, self.channel_properties_fb, self.quantizer_list['root'].hamming_dist_array, self.quantizer_list['root'].hamming_dist_array_fb,map_decode = self.map_decode, training_channel_properties = self.training_channel_properties, epsilon_array = self.epsilon_array, ria = self.ria)
                        if self.prev_centroids:
                            print('adding initial centroids')
                            self.quantizer_list[index_string].prev_centroids = self.prev_centroids[index_string]
                        self.quantizer_list[index_string].train_quantizer()
            if bit_index > 1 and bit_index < len(self.bit_allocation):
                for prev_bit_index in range(2**sum(self.bit_allocation[0:bit_index-1])):
                    for current_bit_index in range(2**self.bit_allocation[bit_index-1]):
                        prev_index_string = COVQ.convert_to_binary(prev_bit_index, sum(self.bit_allocation[0:bit_index-1]))
                        curr_index_string = COVQ.convert_to_binary(current_bit_index, self.bit_allocation[bit_index-1])
                        if (prev_index_string in self.quantizer_list.keys()):
                            prev_quantizer = self.quantizer_list[prev_index_string]
                            prev_quantizer.leaf= False
                            received_subset = (prev_quantizer.map_array == current_bit_index) if self.map_decode else (prev_quantizer.transmission_results_fb == current_bit_index)
                            source = prev_quantizer.source
                            self.quantizer_list[prev_index_string + curr_index_string] = LeafTSVQFB(source, received_subset, prev_quantizer.index_array, self.bit_allocation[bit_index], self.channel_properties, self.channel_properties_fb, prev_quantizer.append_error_array, prev_quantizer.append_error_array_fb, map_decode = self.map_decode, training_channel_properties = self.training_channel_properties, epsilon_array = self.epsilon_array, ria = self.ria)
                            if self.prev_centroids:
                                print('adding initial centroids')
                                self.quantizer_list[prev_index_string + curr_index_string].prev_centroids = self.prev_centroids[prev_index_string + curr_index_string]
                            if (np.sum(prev_quantizer.transmission_results_fb == current_bit_index)!=0):
                                self.quantizer_list[prev_index_string + curr_index_string].train_quantizer()
                            else:
                                print("ERROR EMPTY CELL")
                        
        if(len(self.bit_allocation)>1):
            for i in range(2**sum(self.bit_allocation[0:len(self.bit_allocation)-1])):
                prev_index_string = COVQ.convert_to_binary(i, sum(self.bit_allocation[0:len(self.bit_allocation)-1]))
                if (prev_index_string in self.quantizer_list.keys()) and len(self.quantizer_list[prev_index_string].source)>0:
                    self.total_distortion += sum(self.quantizer_list[prev_index_string].calc_distortion())
                    self.total_points += len(self.quantizer_list[prev_index_string].distortion_mapping)
        else:
            self.total_distortion = sum(self.quantizer_list['root'].calc_distortion())
            self.total_points = len(self.quantizer_list['root'].distortion_mapping)

        print('done')
        print(self.quantizer_list.keys(), 'keys')
        print('dimension', self.samples.shape[1])
        print('total samples', self.total_points)
        #print('total distortion', -10*math.log10(self.total_distortion/(self.total_points*self.samples.shape[1])))
    
    def calc_distortion(self):
        return -10*math.log10(self.total_distortion/(self.total_points*self.samples.shape[1]))
    
    def calc_distortion_progression(self):
        #function to calculate the distortion after each stage
        pass
    
    def export_centroids_stage(self):
        #function to export dictionary of centroids by stage
        exported_centroids = {i: [] for i in range(len(self.bit_allocation))}
        exported_centroids[0].append(self.quantizer_list['root'].centroids)
        for stage in range(len(self.bit_allocation)-1):
            for bit_index in range(2**sum(self.bit_allocation[0:stage+1])):
                index_string = COVQ.convert_to_binary(bit_index, sum(self.bit_allocation[0:stage+1]))
                exported_centroids[stage+1].append(self.quantizer_list[index_string].centroids)
        return exported_centroids

    def export_initialization_stage(self):
        #function to export dictionary of centroids by stage
        exported_centroids = {i: [] for i in range(len(self.bit_allocation))}
        exported_centroids[0].append(self.quantizer_list['root'].initialization_centroids)
        for stage in range(len(self.bit_allocation)-1):
            for bit_index in range(2**sum(self.bit_allocation[0:stage+1])):
                index_string = COVQ.convert_to_binary(bit_index, sum(self.bit_allocation[0:stage+1]))
                exported_centroids[stage+1].append(self.quantizer_list[index_string].initialization_centroids)
        return exported_centroids

    def export_initialization_index(self):
        exported_centroids = {}
        exported_centroids['root'] = self.quantizer_list['root'].initialization_centroids
        for stage in range(len(self.bit_allocation)-1):
            for bit_index in range(2**sum(self.bit_allocation[0:stage+1])):
                index_string = COVQ.convert_to_binary(bit_index, sum(self.bit_allocation[0:stage+1]))
                exported_centroids[index_string] = self.quantizer_list[index_string].initialization_centroids
        return exported_centroids

    def export_centroids_index(self):
        exported_centroids = {}
        exported_centroids['root'] = self.quantizer_list['root'].centroids
        for stage in range(len(self.bit_allocation)-1):
            for bit_index in range(2**sum(self.bit_allocation[0:stage+1])):
                index_string = COVQ.convert_to_binary(bit_index, sum(self.bit_allocation[0:stage+1]))
                exported_centroids[index_string] = self.quantizer_list[index_string].centroids
        return exported_centroids

class RootCOVQFB():
    def __init__(self, samples, bits, channel_properties, channel_properties_fb, map_decode = False, training_channel_properties = None, prev_centroids = None, epsilon_array = None, ria = False) -> None:
        self.source = samples
        self.bits = bits
        self.index_array = np.array(range(len(samples))) #array of the index so other nodes can keep track of original value
        self.nn_cond_vectors = np.zeros(samples.shape[0], dtype=int)
        self.distortion_mapping = np.zeros(samples.shape[0])
        self.centroids = None #np.zeros((2**bits, samples.shape[1]))
        self.transmission_results = np.zeros(samples.shape[0], dtype=int) 
        self.transmission_results_fb = np.zeros(samples.shape[0], dtype=int) #simulation results
        self.transition_matrix = COVQ.create_transition_matrix(bits, channel_properties['memory'], channel_properties['epsilon'], channel_properties['delta'], '', '')
        self.transition_matrix_fb = COVQ.create_transition_matrix(bits, channel_properties_fb['memory'], channel_properties_fb['epsilon'], channel_properties['delta'], '', '')

        self.map_decode = map_decode

        self.channel_properties = channel_properties
        self.channel_properties_fb = channel_properties_fb

        self.epsilon_array = epsilon_array #epsilon array for initialization
        self.channel_properties = channel_properties

        self.prev_centroids = prev_centroids
        self.memory = channel_properties['memory']
        self.hamming_dist_array = np.zeros((len(self.source), self.bits), dtype=int)
        self.hamming_dist_array_fb = np.zeros((len(self.source), self.bits), dtype=int)
        self.leaf = False
        self.initialization_centroids = None
        self.array_index = np.array(range(len(samples)))

        self.ria = ria

        if training_channel_properties is None:
            self.training_transition_matrix = self.transition_matrix
            self.training_channel_properties = channel_properties
        else:
            self.training_channel_properties = training_channel_properties
            self.training_transition_matrix = COVQ.create_transition_matrix(bits, training_channel_properties['memory'], training_channel_properties['epsilon'], training_channel_properties['delta'], '', '')

        if self.ria:
            self.training_transition_matrix = COVQ.transition_matrix_ria(self.training_transition_matrix)
            self.transition_matrix = COVQ.transition_matrix_ria(self.transition_matrix)
    
    def nn_cond(self):
        threadsperblock = 1024
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)
        COVQ.nn_cond_vectors_gpu[blockspergrid, threadsperblock](self.source, self.centroids, self.bits, self.training_transition_matrix, self.nn_cond_vectors, self.distortion_mapping)
   
    def train_quantizer(self):
        if(self.prev_centroids is None and self.centroids is None):
            print('generating initial centroids for root')
            kmeans = KMeans(n_clusters=2**self.bits, n_init="auto", random_state = 0).fit(self.source)
            centroids = kmeans.cluster_centers_
            self.centroids = TSVQ.simulated_annealing(centroids, self.transition_matrix[:,:])
            self.initialization_centroids = kmeans.cluster_centers_
        # else:
        #     self.centroids = self.prev_centroids
        #     self.initialization_centroids = self.prev_centroids
        

        distortion1 = 10
        distortion2 = 10000

        threadsperblock = 1024
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)
        #training quantizer

        if self.epsilon_array is not None:
            assert self.epsilon_array[-1] == self.training_channel_properties['epsilon'], "Epsilon array does not match training channel properties epsilon"

            for epsilon in self.epsilon_array:
                transition_matrix = COVQ.create_transition_matrix(self.bits, self.training_channel_properties['memory'], epsilon, self.training_channel_properties['delta'], '', '')
                if self.ria:
                    transition_matrix = COVQ.transition_matrix_ria(transition_matrix)
                distortion1 = 10
                distortion2 = 10000
                while(abs(distortion2-distortion1)/distortion2 > 0.001):
                    COVQ.nn_cond_vectors_gpu[blockspergrid, threadsperblock](self.source, self.centroids, self.bits, transition_matrix, self.nn_cond_vectors, self.distortion_mapping)
                    self.centroids = COVQ.centroid_cond_vectors(self.source, self.nn_cond_vectors, self.bits, self.source.shape[1], transition_matrix)
                    #updating distortion
                    distortion1 = distortion2
                    distortion2 = self.distortion_mapping.mean()/self.source.shape[1]
                    print("distortion using increase decrease", distortion2)
        else:
            while(abs(distortion2-distortion1)/distortion2 > 0.001):
                COVQ.nn_cond_vectors_gpu[blockspergrid, threadsperblock](self.source, self.centroids, self.bits, self.training_transition_matrix, self.nn_cond_vectors, self.distortion_mapping)
                self.centroids = COVQ.centroid_cond_vectors(self.source, self.nn_cond_vectors, self.bits, self.source.shape[1], self.training_transition_matrix)
                #updating distortion
                distortion1 = distortion2
                distortion2 = self.distortion_mapping.mean()/self.source.shape[1]
                print("distortion", distortion2)
        self.transmission()
        self.transmission_fb()

        if self.map_decode:
            self.MAP_decoding()
        
    def transmission(self):
        threadsperblock = 1024
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)
        #simulate channel
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=np.random.randint(1000))
        COVQ.simulate_channel[blockspergrid, threadsperblock](self.nn_cond_vectors, self.transition_matrix, self.transmission_results, rng_states)

        binary_array = np.zeros((len(self.source), self.bits), dtype=int)
        binary_array_feedback = np.zeros((len(self.source), self.bits), dtype=int)

        threadsperblock_2d = (16, 16)
        blockspergrid_x = math.ceil(len(self.source) / threadsperblock_2d[0])
        blockspergrid_y = math.ceil(self.bits / threadsperblock_2d[1])
        blockspergrid_2d = (blockspergrid_x, blockspergrid_y)

        TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.nn_cond_vectors, self.bits, binary_array)
        TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.transmission_results, self.bits, binary_array_feedback)
        TSVQ.find_hamming_distance[blockspergrid_2d, threadsperblock_2d](self.bits, binary_array, binary_array_feedback, self.hamming_dist_array)
        #TSVQ.find_num_errors[blockspergrid, threadsperblock](self.bits, self.hamming_dist_array, self.memory, self.num_errors)
        self.append_error_array = self.hamming_dist_array.copy() #do this just for the root quantizer
        print('finished training first quantizer')
    
    def transmission_fb(self):
        threadsperblock = 1024
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)
        #simulate channel
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=np.random.randint(1000))
        COVQ.simulate_channel[blockspergrid, threadsperblock](self.transmission_results, self.transition_matrix_fb, self.transmission_results_fb, rng_states)

        binary_array = np.zeros((len(self.source), self.bits), dtype=int)
        binary_array_feedback = np.zeros((len(self.source), self.bits), dtype=int)

        threadsperblock_2d = (16, 16)
        blockspergrid_x = math.ceil(len(self.source) / threadsperblock_2d[0])
        blockspergrid_y = math.ceil(self.bits / threadsperblock_2d[1])
        blockspergrid_2d = (blockspergrid_x, blockspergrid_y)

        TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.transmission_results, self.bits, binary_array)
        TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.transmission_results_fb, self.bits, binary_array_feedback)
        TSVQ.find_hamming_distance[blockspergrid_2d, threadsperblock_2d](self.bits, binary_array, binary_array_feedback, self.hamming_dist_array_fb)
        self.append_error_array_fb = self.hamming_dist_array_fb.copy()
    
    def calc_distortion(self):
        quantized_source = self.centroids[self.transmission_results]
        distortion = np.linalg.norm(quantized_source - self.source, axis=1)**2
        return distortion
    
    def calc_noiseless_distortion(self):
        quantized_source_noiseless = self.centroids[self.nn_cond_vectors]
        distortion = np.linalg.norm(quantized_source_noiseless - self.source, axis=1)**2
        return distortion

    def MAP_decoding(self):
        threadsperblock = 1024
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)

        self.oplus_noise_binary = self.hamming_dist_array ^ self.hamming_dist_array_fb #binary array of fb noise oplus forward noise
        decimal_array = np.zeros(len(self.oplus_noise_binary), dtype=int)
        map_noise_array = np.zeros(len(self.oplus_noise_binary), dtype=int)
        self.map_array = np.zeros(len(self.oplus_noise_binary), dtype=int)
        #i have no idea why, but without this, this will have an incompatability error when column dimension is 1 
        binary_array = self.oplus_noise_binary if self.bits==1 else np.ascontiguousarray(np.flip(self.oplus_noise_binary, 1))
        TSVQ.msb_left_binary_2_decimal[blockspergrid, threadsperblock](self.oplus_noise_binary.shape[1], binary_array, self.oplus_noise_binary.shape[1], decimal_array)

        MAP_decoder = MAP.MAP_Decoder(self.channel_properties['epsilon'], self.channel_properties_fb['epsilon'], self.channel_properties['delta'], self.channel_properties_fb['delta'], self.oplus_noise_binary.shape[1])
        MAP_decoder.calc_transition_matrix()
        transition_matrix = MAP_decoder.transition_matrix

        #find max index of send then do modulo addition to estimate y
        TSVQ.MAP_decoding[blockspergrid, threadsperblock](decimal_array, transition_matrix, map_noise_array)
        #figure out how to take the last 2 bits
        divide_by = int(transition_matrix.shape[1]/2**(self.bits))
        self.map_array = (map_noise_array)^self.transmission_results_fb #map decoding should only be used in the testing set
        


class LeafTSVQFB(LeafTSVQ):
    #figure out how to do inheritance in python
    def __init__(self, source, subset_index, index_array, bits, channel_properties, channel_properties_fb, hamming_dist_array, hamming_dist_array_fb, map_decode = False, training_channel_properties = None, prev_centroids = None, epsilon_array = None, ria = False) -> None:
        self.total_source = source
        self.index_array = index_array[subset_index]
        self.prev_centroids = prev_centroids

        self.map_decode = map_decode

        #quantizer parameters
        self.source = source[subset_index]
        self.bits = bits
        self.nn_cond_vectors = np.zeros(self.source.shape[0], dtype=int)
        self.distortion_mapping = np.zeros(self.source.shape[0])
        self.centroids = None#np.zeros((2**bits, self.source.shape[1]))

        #noise from previous transmissions for both forward and feedback channel
        self.hamming_dist_array = hamming_dist_array[subset_index] #just an array, append 
        self.hamming_dist_array_fb = hamming_dist_array_fb[subset_index] 

        self.channel_properties = channel_properties
        self.channel_properties_fb = channel_properties_fb

        #forward channel parameters
        self.transition_matrix = TSVQ.create_transition_matrix(bits, min(hamming_dist_array.shape[1],channel_properties['memory']), channel_properties['epsilon'], channel_properties['delta'])
        self.num_errors_array = np.zeros(self.source.shape[0], dtype=int)
        self.memory = channel_properties['memory']
        self.transmission_results = np.zeros(len(self.source), dtype=int)
        self.transmission_results_fb = np.zeros(len(self.source), dtype=int)

        self.epsilon_array = epsilon_array
        self.channel_properties = channel_properties

        #feedback channel parameters
        self.transition_matrix_fb = TSVQ.create_transition_matrix(bits, min(hamming_dist_array.shape[1],channel_properties_fb['memory']), channel_properties_fb['epsilon'], channel_properties_fb['delta'])
        self.num_errors_array_fb = np.zeros(self.source.shape[0], dtype=int)
        self.memory_fb = channel_properties_fb['memory']
        self.noise_feedback_results = np.zeros(len(self.source))

        self.append_error_array = None
        self.append_error_array_fb = None
        self.oplus_noise = None #mod 2 addition of forward and backward noise

        self.leaf = True
        self.initialization_centroids = None
        self.ria = ria

        #creating transition matrix that will be used for training
        if training_channel_properties is None:
            self.training_transition_matrix = self.transition_matrix
            self.training_channel_properties = channel_properties
        else:
            self.training_channel_properties = training_channel_properties
            self.training_transition_matrix = TSVQ.create_transition_matrix(bits, min(hamming_dist_array.shape[1],training_channel_properties['memory']), training_channel_properties['epsilon'], training_channel_properties['delta'])
        if self.ria:
            self.training_transition_matrix = TSVQ.transition_matrix_ria(self.training_transition_matrix)
            self.transition_matrix = TSVQ.transition_matrix_ria(self.transition_matrix)

        threadsperblock = 1024
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)

        #populating the memory of both forwards and feedback channel
        TSVQ.find_num_errors[math.ceil(len(self.hamming_dist_array) / threadsperblock), threadsperblock](self.hamming_dist_array.shape[1], self.hamming_dist_array, self.memory, self.num_errors_array)
        TSVQ.find_num_errors[math.ceil(len(self.hamming_dist_array) / threadsperblock), threadsperblock](self.hamming_dist_array.shape[1], self.hamming_dist_array_fb, self.memory_fb, self.num_errors_array_fb)
    
    def train_quantizer(self):
        threadsperblock = 1000
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)

        print(len(self.hamming_dist_array), len(self.source))
        TSVQ.find_num_errors[math.ceil(len(self.hamming_dist_array) / threadsperblock), threadsperblock](self.hamming_dist_array.shape[1], self.hamming_dist_array, self.memory, self.num_errors_array)

        if((self.prev_centroids) is None and self.centroids is None):
            print('generating initial centroids')
            kmeans = KMeans(n_clusters=2**self.bits, n_init="auto", random_state = 0).fit(self.source)
            centroids = kmeans.cluster_centers_
            self.centroids = TSVQ.simulated_annealing(centroids, self.transition_matrix[:,:,0])
            self.initialization_centroids = centroids
        # else:
        #     print('importing centroids')
        #     self.centroids = self.prev_centroids
        #     self.initialization_centroids = self.prev_centroids

        distortion1 = 10
        distortion2 = 10000

        if self.epsilon_array is not None:
            assert self.epsilon_array[-1] == self.training_channel_properties['epsilon'], "Epsilon array does not match training channel properties epsilon"
            
            for epsilon in self.epsilon_array:
                print("epsilon", epsilon)
                transition_matrix = TSVQ.create_transition_matrix(self.bits, min(self.hamming_dist_array.shape[1],self.channel_properties['memory']), epsilon, self.channel_properties['delta'])
                if self.ria:
                    transition_matrix = TSVQ.transition_matrix_ria(transition_matrix)
                distortion1 = 10
                distortion2 = 10000
                while(abs(distortion1-distortion2)/distortion2 > 0.001/4):
                    TSVQ.nn_cond_vectors_gpu[blockspergrid, threadsperblock](self.source, self.centroids, self.bits, transition_matrix, self.num_errors_array, self.nn_cond_vectors, self.distortion_mapping)
                    self.centroids = TSVQ.centroid_cond_vectors(self.source, self.nn_cond_vectors, self.num_errors_array, self.bits, self.source.shape[1], transition_matrix)
                    distortion1 = distortion2
                    distortion2 = np.sum(self.distortion_mapping)/(self.source.shape[1]*len(self.source))
                    print("distortion using increase decrease", distortion2)
        else:
            while(abs(distortion1-distortion2)/distortion2 > 0.0001):
                TSVQ.nn_cond_vectors_gpu[blockspergrid, threadsperblock](self.source, self.centroids, self.bits, self.training_transition_matrix, self.num_errors_array, self.nn_cond_vectors, self.distortion_mapping)
                self.centroids = TSVQ.centroid_cond_vectors(self.source, self.nn_cond_vectors, self.num_errors_array, self.bits, self.source.shape[1], self.training_transition_matrix)
                print('distortion:', distortion2)
                distortion1 = distortion2
                distortion2 = np.sum(self.distortion_mapping)/(self.source.shape[1]*len(self.source))
        self.transmission()
        self.transmission_fb()

        if self.map_decode:
            self.MAP_decoding()
            
    def nn_cond(self):
        threadsperblock = 1024
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)
        TSVQ.nn_cond_vectors_gpu[blockspergrid, threadsperblock](self.source, self.centroids, self.bits, self.training_transition_matrix, self.num_errors_array, self.nn_cond_vectors, self.distortion_mapping)

    def transmission(self):
        threadsperblock = 1024
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)

        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=np.random.randint(10000))
        TSVQ.simulate_channel[blockspergrid, threadsperblock](self.nn_cond_vectors, self.transition_matrix, self.num_errors_array, self.transmission_results, rng_states)

        binary_array = np.zeros((len(self.source), self.bits), dtype=int)
        binary_array_feedback = np.zeros((len(self.source), self.bits), dtype=int)

        threadsperblock_2d = (64, 16)
        blockspergrid_x = math.ceil(len(self.source) / threadsperblock_2d[0])
        blockspergrid_y = math.ceil(self.bits / threadsperblock_2d[1])
        blockspergrid_2d = (blockspergrid_x, blockspergrid_y)

        TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.nn_cond_vectors, self.bits, binary_array)
        TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.transmission_results, self.bits, binary_array_feedback)
        
        current_hamming_dist_array = np.zeros((len(self.source), self.bits), dtype=int)
        TSVQ.find_hamming_distance[blockspergrid_2d, threadsperblock_2d](self.bits, binary_array, binary_array_feedback, current_hamming_dist_array)

        #noise array with new noise appended
        self.append_error_array = np.concatenate((self.hamming_dist_array, current_hamming_dist_array), axis=1)
    
    def transmission_fb(self):
        threadsperblock = 1024
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)

        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=np.random.randint(10000))
        TSVQ.simulate_channel[blockspergrid, threadsperblock](self.transmission_results, self.transition_matrix_fb, self.num_errors_array_fb, self.transmission_results_fb, rng_states)

        binary_array = np.zeros((len(self.source), self.bits), dtype=int)
        binary_array_feedback = np.zeros((len(self.source), self.bits), dtype=int)

        threadsperblock_2d = (64, 16)
        blockspergrid_x = math.ceil(len(self.source) / threadsperblock_2d[0])
        blockspergrid_y = math.ceil(self.bits / threadsperblock_2d[1])
        blockspergrid_2d = (blockspergrid_x, blockspergrid_y)

        TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.transmission_results, self.bits, binary_array)
        TSVQ.calc_binary_array[blockspergrid_2d, threadsperblock_2d](self.transmission_results_fb, self.bits, binary_array_feedback)
        
        current_hamming_dist_array = np.zeros((len(self.source), self.bits), dtype=int)
        TSVQ.find_hamming_distance[blockspergrid_2d, threadsperblock_2d](self.bits, binary_array, binary_array_feedback, current_hamming_dist_array)

        #noise array with new noise appended
        self.append_error_array_fb = np.concatenate((self.hamming_dist_array_fb, current_hamming_dist_array), axis=1)
    
    def calc_distortion(self):
        quantized_source = self.centroids[self.transmission_results]
        distortion = np.linalg.norm(quantized_source - self.source, axis=1)**2
        return distortion
    
    def MAP_decoding(self):
        threadsperblock = 1024
        blockspergrid = math.ceil(self.source.shape[0] / threadsperblock)

        self.oplus_noise_binary = (self.append_error_array ^ self.append_error_array_fb) #binary array of fb noise oplus forward noise
        decimal_array = np.zeros(len(self.oplus_noise_binary), dtype=int)
        map_noise_array = np.zeros(len(self.oplus_noise_binary), dtype=int)
        self.map_array = np.zeros(len(self.oplus_noise_binary), dtype=int)
        TSVQ.msb_left_binary_2_decimal[blockspergrid, threadsperblock](self.oplus_noise_binary.shape[1], np.ascontiguousarray((self.oplus_noise_binary)[:,::-1]), self.oplus_noise_binary.shape[1], decimal_array)

        MAP_decoder = MAP.MAP_Decoder(self.channel_properties['epsilon'],self.channel_properties_fb['epsilon'], self.channel_properties['delta'], self.channel_properties_fb['delta'], self.oplus_noise_binary.shape[1])
        MAP_decoder.calc_transition_matrix()
        transition_matrix = MAP_decoder.transition_matrix
        self.map_transition_matrix = MAP_decoder.transition_matrix
        self.decimal_array = decimal_array
        #find max index of send then do modulo addition to estimate y
        TSVQ.MAP_decoding[blockspergrid, threadsperblock](decimal_array, transition_matrix, map_noise_array)
        #figure out how to take the last 2 bits
        divide_by = int(transition_matrix.shape[1]/2**(self.bits))
        self.divide_by = int(transition_matrix.shape[1]/2**(self.bits))
        self.noise_prediction = map_noise_array #prediction the feedback noise
        self.map_array = (map_noise_array%(2**self.bits))^self.transmission_results_fb
        