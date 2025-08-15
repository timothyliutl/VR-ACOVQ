from numba import jit
from numba import cuda, float32
import numpy as np
import time
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from scipy.special import gamma

@jit(nopython=True)
def generate_gaussian_source(num_samples, dimension, correlation = 0.9):
    samples = np.zeros(num_samples*dimension)
    for i in range(num_samples*dimension):
        if(i>0):
            samples[i] = (np.random.normal(0,1) + correlation*samples[i-1])
    stddev = np.std(samples)
    samples = samples/stddev
    samples = samples.reshape(num_samples, dimension)
    return samples

def gg_density(x, alpha=1, sigma=1):
    ita = ((gamma(3/alpha)/gamma(1/alpha))**(-1/2))/sigma
    term1 = (alpha * ita)/(2*gamma(1/alpha))
    term2 = np.exp(-1*ita*(np.abs(x)**alpha))
    return term1*term2


# def generate_gg_source(num_samples, dimension, alpha = 1, sigma = 1):
#     total_samples = num_samples*dimension
#     f_max = gg_density(0, alpha, sigma)
#     accepted = np.array([])
#     while accepted.shape[0] < total_samples:
#         uniform_dist_samples = np.random.uniform(-6, 6, int(total_samples*1.5))
#         candidate_samples = np.random.uniform(0, 1.2, int(total_samples*1.5))
#         mask = uniform_dist_samples[gg_density(uniform_dist_samples, alpha, sigma) > candidate_samples]
#         accepted = np.concatenate((accepted, mask))
#     return accepted[:total_samples].reshape(num_samples, dimension)
    


@jit(nopython=True)
def transition_prob(bits, memory, epsilon, delta, num1, num2, prev_seq1, prev_seq2):
        if len(prev_seq1) != len(prev_seq2):
            print("ERROR: PREV SEQUENCES NOT SAME SIZE")
        #add leading zeros
        num1 = (bits - len(num1))*'0' + num1
        num2 = (bits - len(num2))*'0' + num2
        probability = 1
        for i in range(bits):
            current_prev_seq1 = (num1[:i])[::-1] + (prev_seq1)[::-1] #reverse string because we send from left to right
            current_prev_seq2 = (num2[:i])[::-1] + (prev_seq2)[::-1]
            sum_errors = 0
            memory_length = min(len(current_prev_seq1), memory)
            #print(memory_length)
            for j in range(memory_length):
                if current_prev_seq1[j] != current_prev_seq2[j]:
                     sum_errors = sum_errors + 1
            
            #print(self.epsilon, self.delta, self.memory, memory_length, sum_errors)
            #print('prev_errors =', sum_er
            if num1[i] == num2[i]:
                probability = probability * (1 - ((epsilon + delta*sum_errors)/(1+ delta*memory_length)))
                #print((1 - ((self.epsilon + self.delta*sum_errors)/(1+ self.delta*memory_length))))
            else:
                probability = probability * ((epsilon + delta*sum_errors)/(1+ delta*memory_length))
                #print(((self.epsilon + self.delta*sum_errors)/(1+ self.delta*memory_length)))
        return probability


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

@jit(nopython=True)
def find_distortion(point, quantized_index, bits, centroids, transition_matrix):
        #need to make a way to find previous quantized values
        distortion = float(0)
        for index in range(len(centroids)):
            distortion = distortion + transition_matrix[index, quantized_index]*(np.linalg.norm(point - centroids[index]))**2
        return distortion

@jit(nopython=True)
def nn_cond_vectors(source, centroids, bits, transition_matrix):
     return_array = np.zeros(len(source))
     return_distortion_array = np.zeros(len(source))
     for point_index in range(len(source)):
        min_index = 0
        min_distortion = 10000
        for centroid_index in range(len(centroids)):
            current_dist = find_distortion(source[point_index], centroid_index, bits, centroids, transition_matrix)
            if current_dist < min_distortion:
                min_index = centroid_index
                min_distortion = current_dist
        return_array[point_index] = min_index
        return_distortion_array[point_index] = min_distortion
     return return_array, return_distortion_array


@cuda.jit(device=True)
def find_distortion_cuda(point, quantized_index, bits, centroids, transition_matrix):
        #need to make a way to find previous quantized values
        distortion = float(0)
        for index in range(len(centroids)):
            norm = float(0)
            for i in range(point.shape[0]):
                norm = norm + (point[i] - centroids[index][i])**2
            distortion = distortion + transition_matrix[index, quantized_index]*norm
        return distortion

#code nn cond on the gpu
@cuda.jit
def nn_cond_vectors_gpu(source, centroids, bits, transition_matrix, nn_array, distortion_array):
    pos = cuda.grid(1)
    if pos < len(source):
        min_index = 0
        min_distortion = 10000
        for centroid_index in range(len(centroids)):
            current_dist = find_distortion_cuda(source[pos], centroid_index, bits, centroids, transition_matrix)
            if current_dist < min_distortion:
                min_index = centroid_index
                min_distortion = current_dist
        nn_array[pos] = int(min_index)
        distortion_array[pos] = min_distortion
            

@jit(nopython=True)
def centroid_cond_vectors(source, nn_mapping, bits, dimension, transition_matrix):
    new_centroids = np.zeros((2**bits, dimension))
    for rec_index in range(2**bits):
        numerator = np.zeros(dimension)
        denominator = float(0)
        for quant_index in range(2**bits):   
            numerator = numerator + np.sum(source[nn_mapping == quant_index], axis=0)*transition_matrix[rec_index, quant_index]
            denominator = denominator + source[nn_mapping == quant_index].shape[0]*transition_matrix[rec_index, quant_index]
        new_centroids[rec_index] = numerator/denominator
    return new_centroids

def create_transition_matrix(bits, memory, epsilon, delta, prev_seq1, prev_seq2,):
    transition_matrix = np.zeros((2**bits, 2**bits))
    for i in range(2**bits):
        for j in range(2**bits):
            i_bin_string = convert_to_binary(i, bits)
            j_bin_string = convert_to_binary(j,bits)
            transition_matrix[i,j] = transition_prob(bits, memory, epsilon, delta, i_bin_string, j_bin_string, prev_seq1, prev_seq2)
    return transition_matrix

def transition_matrix_ria(matrix):
    """
    Replaces non-diagonal elements of a square matrix with the average of the
    remaining elements in their respective row.
    
    Parameters:
    matrix (numpy.ndarray): Input square matrix (2D array)
    
    Returns:
    numpy.ndarray: Modified matrix with non-diagonal elements replaced by row averages
    
    Raises:
    ValueError: If input is not a square matrix
    """
    # Create a copy to avoid modifying the original matrix
    mat = matrix.copy()
    
    # Verify matrix is square
    n_rows, n_cols = mat.shape
    if n_rows != n_cols:
        raise ValueError("Input must be a square matrix")
    
    # Edge case: 1x1 matrix (no non-diagonal elements)
    if n_rows == 1:
        return mat
    
    # Calculate row averages (excluding diagonal elements)
    diag_elements = np.diag(mat)
    row_sums = mat.sum(axis=1) - diag_elements
    row_averages = row_sums / (n_rows - 1)
    
    # Create mask for non-diagonal elements
    mask = np.ones_like(mat, dtype=bool)
    np.fill_diagonal(mask, False)
    
    # Replace non-diagonal elements with row averages
    mat[mask] = np.repeat(row_averages, n_rows - 1)
    
    return mat


@cuda.jit
def simulate_channel(nn_mapping, transition_matrix, result_mapping, rng_states):
    pos = cuda.grid(1)
    if pos < len(nn_mapping):
        transition_row = transition_matrix[int(nn_mapping[pos]),:]
        rand_num = xoroshiro128p_uniform_float32(rng_states, pos)
        sum_val = float(0)
        for i in range(transition_row.shape[0]):
            sum_val += transition_row[i]
            #result_mapping[pos] = transition_row[0]
            if rand_num < sum_val:
                result_mapping[pos] = i
                break




