import numpy as np
import pandas as pd
import TSVQ_Modules.COVQ as COVQ

class MAP_Decoder():
    def __init__(self, epsilon, epsilon_fb, delta, delta_fb, seq_length):
        self.epsilon = epsilon
        self.epsilon_fb = epsilon_fb
        self.delta = delta
        self.delta_fb = delta_fb
        self.seq_length = seq_length
    
    def calc_transition_matrix(self):
        noise_distribution = np.zeros(2**self.seq_length)
        noise_distribution_fb = np.zeros(2**self.seq_length)

        for i in range(2**self.seq_length):
            zeros = '0'*self.seq_length
            binary_string = COVQ.convert_to_binary(i, self.seq_length)
            prob = COVQ.transition_prob(self.seq_length, 1, self.epsilon, self.delta, binary_string, zeros, '', '')
            prob_fb = COVQ.transition_prob(self.seq_length, 1, self.epsilon_fb, self.delta_fb, binary_string, zeros, '', '')
            noise_distribution[i] = prob
            noise_distribution_fb[i] = prob_fb

        noise_distribution = np.matrix(noise_distribution)
        noise_distribution_fb = np.matrix(noise_distribution_fb)

        self.transition_matrix = noise_distribution_fb.transpose()* noise_distribution

    def evaluate_decoder(self):
        sum_i = 0
        sum_j = 0
        for noise in range(2**self.seq_length):
            max_prob = 0
            max_i = 0
            max_j = 0
            for i in range(2**self.seq_length):
                j = noise^i
                if self.transition_matrix[i,j] > max_prob:
                    max_prob = self.transition_matrix[i,j]
                    max_i = i
                    max_j = j
                    sum_i += i
                    sum_j += j

        useless_decoder = ((sum_i == 0) or (sum_j == 0))
        return useless_decoder
