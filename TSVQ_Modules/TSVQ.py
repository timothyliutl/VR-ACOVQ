from numba import jit, njit
from numba import cuda
import numpy as np
import time
import math
import TSVQ_Modules.COVQ as COVQ
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import copy


@jit(nopython=True)
def int_to_binary_string(n):
    if n == 0:
        return "0"
    binary_str = ""
    while n > 0:
        binary_str = str(n % 2) + binary_str
        n = n // 2
    return binary_str

@jit(nopython=True)
def convert_to_binary(number, bits):
    binary_string = int_to_binary_string(number)
    binary_string = (bits - len(binary_string))*'0' + binary_string #adding leading zeros
    return binary_string


@cuda.jit
def calculate_error(source, simulation_results_array, centroid_arr, return_array):
    pos = cuda.grid(1)
    if pos < len(source):
        #sum source and refinement quantization
        for dim in range(source.shape[1]):
            quantization_value = centroid_arr[int(simulation_results_array[pos])][dim]
            return_array[pos][dim] = source[pos][dim] - quantization_value

@cuda.jit
def calc_binary_array(array, bits, binary_array):
    #most significant bit will be on the right/higher index
    #x - source index
    #y - bit index
    pos_x, pos_y = cuda.grid(2)
    if pos_x<len(array) and pos_y < bits:
        num = array[pos_x]
        binary_array[pos_x, pos_y] = (num // (2**(bits - pos_y - 1))) % 2


@cuda.jit
def find_hamming_distance(bits, nn_array_binary, feedback_array_binary, num_error_array):
    #nn_mapping and feedback are an array of arrays
    #bits_array are array of corresponding bits
    #z - stage index, x - source index, y - bit index
    pos_x, pos_y = cuda.grid(2)
    if pos_y < nn_array_binary.shape[1] and pos_x < len(nn_array_binary):
        num_error_array[pos_x, pos_y] = (nn_array_binary[pos_x, pos_y] + feedback_array_binary[pos_x, pos_y]) % 2

# @cuda.jit
# def binary_to_decimal(binary_array, decimal_array):
#     #assuming msb is on the left
#     pos = cuda.grid(1)
#     num_bits = binary_array.shape[1]
#     sum_val = 0
#     if pos < binary_array.shape[0]:
#         pass

# @cuda.jit
# def find_hamming_distance_multi(bits_array, size_array, nn_mapping_binary, feedback_mapping_binary, num_error_mapping):
#     #input mappings should be flattened matricies
#     pos_x = cuda.grid(1)
#     if pos_x < len(nn_mapping_binary):
#         x_index = 0
#         sum_total = int(0)
#         for i in range(len(size_array)):
#             sum_total += size_array[i]
#             if pos_x < sum_total:
#                 x_index+= 1
#             else:
#                 break
#         sub_index = pos_x - sum_total
#         y_index = sub_index // bits_array[x_index]
#         z_index = sub_index % bits_array[x_index]
#         num_error_mapping[x_index, y_index, z_index] = (nn_mapping_binary[pos_x] + feedback_mapping_binary[pos_x])%2

# @cuda.jit
# def find_num_errors(bits_array, hamming_dist_mapping, memory, error_arr):
#     pos_x = cuda.grid(1)
#     if pos_x < len(hamming_dist_mapping[0]):
#         sum_value = 0
#         for i in range(len(bits_array)):
#             sum_value += bits_array[i]

#         memory_actual = int(min(memory, sum_value))

#         bits_used_index = len(bits_array) - 1
#         bits_used_sub_index = bits_array[bits_used_index] - 1
#         total_error = int(0)
#         for i in range(memory_actual):
#             total_error += hamming_dist_mapping[bits_used_index, pos_x, bits_used_sub_index]
#             bits_used_sub_index -= 1
#             if bits_used_sub_index == -1:
#                 bits_used_index -= 1
#                 bits_used_sub_index = bits_array[bits_used_index] - 1
#         error_arr[pos_x] = total_error

@cuda.jit
def find_num_errors(bits, hamming_dist_mapping, memory, error_arr):
    pos_x = cuda.grid(1)
    if pos_x < len(hamming_dist_mapping):
        sum_value = 0
        effective_memory = min(memory, hamming_dist_mapping.shape[1])
        for i in range(effective_memory):
            sum_value += int((2**(effective_memory-i-1))* hamming_dist_mapping[pos_x, bits - i - 1])
        error_arr[pos_x] = sum_value
            
@cuda.jit
def msb_left_binary_2_decimal(bits, hamming_dist_mapping, memory, error_arr):
    pos_x = cuda.grid(1)
    if pos_x < hamming_dist_mapping.shape[0]:
        sum_value = 0
        for i in range(hamming_dist_mapping.shape[1]):
            sum_value += int((2**(i))* hamming_dist_mapping[pos_x, i])
        error_arr[pos_x] = sum_value

@jit(nopython=True)
def centroid_cond_vectors(source, nn_arr, error_array, bits, dimension, transition_matrix):
    new_centroids = np.zeros((2**bits, dimension))
    for rec_index in range(2**bits):
        numerator = np.zeros(dimension)
        denominator = float(0)
        for quant_index in range(2**bits):
            for index in np.where(nn_arr == quant_index)[0]:
                #print(error_array[index], 'skdfhj')
                #print(source[index], transition_matrix[rec_index, quant_index, error_array[index]])
                numerator = numerator + source[index]*transition_matrix[rec_index, quant_index, error_array[index]]
                denominator = denominator + transition_matrix[rec_index, quant_index, error_array[index]]
        new_centroids[rec_index] = numerator/denominator
    return new_centroids

# @jit(nopython=True)
# def expected_val_centroids(source, source_subset_index, rec_index, nn_arr, error_array, transition_matrix):
#     numerator = np.zeros_like(source[0])
#     denominator = 0
#     for i in range(len(source_subset_index)):
#         index = source_subset_index[i]
#         quant_index = nn_arr[index]
#         # print(rec_index, quant_index, error_array[index])
#         numerator = numerator + source[index]*transition_matrix[rec_index, quant_index, error_array[index]]
#         denominator = denominator + transition_matrix[rec_index, quant_index, error_array[index]]
#     return numerator, denominator


@cuda.jit(device=True)
def find_distortion_cuda(point, quantized_index, bits, centroids, transition_matrix, num_errors):
        #need to make a way to find previous quantized values
        distortion = float(0)
        for index in range(len(centroids)):
            norm = float(0)
            for i in range(point.shape[0]):
                norm = norm + (point[i] - centroids[index][i])**2
            distortion = distortion + transition_matrix[index, quantized_index, num_errors]*norm
        return distortion

@cuda.jit
def nn_cond_vectors_gpu(source, centroids, bits, transition_matrix, error_array, nn_array, distortion_array):
    pos = cuda.grid(1)
    if pos < len(source):
        min_index = 0 
        min_distortion = 100000
        for centroid_index in range(len(centroids)):
            current_dist = find_distortion_cuda(source[pos], centroid_index, bits, centroids, transition_matrix, error_array[pos])
            if current_dist < min_distortion:
                min_index = centroid_index
                min_distortion = current_dist
        nn_array[pos] = int(min_index)
        distortion_array[pos] = min_distortion


@cuda.jit
def calc_expected_value(source, centroids, bits, transition_matrix, error_array, nn_array, return_average_array):
    pos = cuda.grid(1)
    if pos < len(source):
        for i in range(source.shape[1]):
            value = float(0)
            for index in range(2**bits):
                value += transition_matrix[index, nn_array[pos], error_array[pos]]*centroids[nn_array[pos]]
            return_average_array[pos, i] = value

def create_transition_matrix(bits, effective_memory, epsilon, delta): #need to pass effective memory in here
    transition_matrix = np.zeros((2**bits, 2**bits, 2**effective_memory))
    for i in range(2**bits):
        for j in range(2**bits):
            for num_errors in range(2**effective_memory):
                i_bin_string = COVQ.convert_to_binary(i, bits)
                j_bin_string = COVQ.convert_to_binary(j, bits)
                prev_noise_string = COVQ.convert_to_binary(num_errors, effective_memory)
                #memory + 10 is to ensure we are using the entire number of errors
                transition_matrix[i,j,num_errors] = COVQ.transition_prob(bits, effective_memory, epsilon, delta,i_bin_string, j_bin_string, '0'*effective_memory, prev_noise_string) #'0'*effective_memory, prev_noise_string
    return transition_matrix

def transition_matrix_ria(matrix):
    mat = matrix.copy()
    for index in range(matrix.shape[2]):
        mat[:,:,index] = COVQ.transition_matrix_ria(matrix[:,:,index])
    return mat

@cuda.jit
def expected_quantization(nn_mapping, centroids, bits, transition_matrix, error_array, return_array):
    pos = cuda.grid(1)
    if pos < len(nn_mapping):
        for index in range(centroids.shape[1]):
            expected_quant = float(0)
            for j in range(2**bits):
                expected_quant += transition_matrix[nn_mapping[pos], j, error_array[pos]]*centroids[j][index]
            return_array[pos, index] = expected_quant

@cuda.jit
def simulate_channel(nn_mapping, transition_matrix, feedback_error, result_mapping, rng_states):
    pos = cuda.grid(1)
    if pos < len(nn_mapping):
        transition_row = transition_matrix[int(nn_mapping[pos]),:,feedback_error[pos]]
        rand_num = xoroshiro128p_uniform_float32(rng_states, pos)
        sum_val = float(0)
        for i in range(transition_row.shape[0]):
            sum_val += transition_row[i]
            if rand_num < sum_val:
                result_mapping[pos] = i
                break


def tsvq_2_acovq_codebook(tsvq_codebook, tsvq_final_codebook, bit_allocation, export_w_index = False):
    #code to convert tsvq codebook to acovq equivalent codebook
    #get cumulative sum of bits for bit allocation
    bit_allocation_array = np.array(bit_allocation)
    cumulative_sum = np.cumsum(bit_allocation_array)
    bits = sum(bit_allocation)

    converted_codebook = copy.deepcopy(tsvq_codebook)

    for i in range(2**bits):
        binary_string = convert_to_binary(i, bits)
        #find string for each branch path
        binary_index_array = []
        #include the string and integer that corresponds to the quantized index
        for j in range(len(bit_allocation)-1):
            binary_index_array.append((binary_string[0:sum(bit_allocation[0:j+1])], int(binary_string[sum(bit_allocation[:j+1]):sum(bit_allocation[:j+2])],2)))

        cumulative_sum = tsvq_final_codebook['root'][int(binary_string[:(bit_allocation[0])],2)]
        print(int(binary_string[:(bit_allocation[0])],2), binary_index_array)
        for index in range(len(binary_index_array)):
            binary_string = binary_index_array[index][0]
            codebook_index = binary_index_array[index][1]
            converted_codebook[binary_string][codebook_index] = tsvq_codebook[binary_string][codebook_index] + cumulative_sum    
            cumulative_sum = cumulative_sum + tsvq_final_codebook[binary_string][codebook_index]

    merged_data = {}
    if export_w_index:
        return converted_codebook
    #print(converted_codebook['101'][0], tsvq_codebook['root'][1] + tsvq_codebook['1'][0] + tsvq_codebook['10'][1] + tsvq_codebook['101'][0])
    
    merged_data[0] = [tsvq_codebook['root']]
    prev_key_length = -1
    stage = 0
    for key, value in converted_codebook.items():
        if (key =='root'):
            continue
        key_length = len(key)
        if key_length != prev_key_length:
            prev_key_length = key_length
            stage = stage + 1
        
        # If key length exists, concatenate arrays, else initialize with the current value
        if stage in merged_data:
            merged_data[stage].append(value)
        else:
            merged_data[stage] = [value]

    
    
    return merged_data

def add_2_codebook(codebook, value):
    array = copy.deepcopy(codebook)
    for i in range(len(array)):
        array[i] = array[i] + value
    return array

@jit(nopython=True)
def shuffle_array(input_arr):
    arr = input_arr.copy()
    n = len(arr)
    l = min(3, n)

    idx = np.empty(l, dtype=np.int64)
    chosen = set()
    for i in range(l):
        r = np.random.randint(0, n)
        while r in chosen:
            r = np.random.randint(0, n)
        idx[i] = r
        chosen.add(r)
    
    perm = np.arange(l)
    for i in range(l):
        j = np.random.randint(i, l)
        perm[i], perm[j] = perm[j], perm[i]
    
    # Swap only the chosen indices
    temp = arr[idx].copy()
    for i in range(l):
        arr[idx[i]] = temp[perm[i]]
    return arr
    
@jit(nopython= True)
def calculate_channel_noise(codebook, transition_matrix):
    dimension = transition_matrix.shape[1]
    distortion = 0
    for i in range(dimension):
        for j in range(dimension):
            distortion += np.linalg.norm(codebook[i] - codebook[j]) * transition_matrix[i, j]
    return distortion


@jit(nopython= True)
def simulated_annealing(codebook, transition_matrix):
    dimension = codebook.shape[1]
    count = 0
    count_success = 0
    count_failure = 0

    temp_init = 10
    temp_final = 2.5*(10**(-4))
    alpha = 0.97

    n_failure = 50000
    n_success = 5
    n_cut = 200

    temp = temp_init
    selected_permutation = shuffle_array(codebook)

    while temp > temp_final and count_failure < n_failure:
        permutation = shuffle_array(codebook)
        new_distortion = calculate_channel_noise(permutation, transition_matrix)
        selected_distortion = calculate_channel_noise(selected_permutation, transition_matrix)
        delta_distortion = new_distortion - selected_distortion

        if delta_distortion <= 0:
            selected_permutation = permutation
            count_success += 1
            count_failure = 0
        elif np.random.rand() < np.exp(-1*delta_distortion/temp):
            selected_permutation = permutation
            count_failure += 1
        else:
            count_failure +=1
        
        if count >= n_cut or count_success >= n_success:
            temp = alpha*temp
            count = 0
            count_success = 0
        count += 1
    return selected_permutation

@cuda.jit
def MAP_decoding(o_plus_array, transition_matrix, return_array):
    pos = cuda.grid(1)
    size = transition_matrix.shape[1]
    if pos <len(o_plus_array):
        max_prob = 0
        max_prob_index = 0
        #looping through all combinations to find highest probability
        for i in range(size):
            for j in range(size):
                if i^j == o_plus_array[pos]:
                    if transition_matrix[i,j] > max_prob:
                        max_prob = transition_matrix[i,j]
                        max_prob_index = i
        return_array[pos] = max_prob_index #returns highest most probable feedback noise