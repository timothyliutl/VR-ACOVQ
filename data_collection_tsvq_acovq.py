import TSVQ_Modules.Generalized_Classes as Generalized_Classes
import TSVQ_Modules.TSVQ as TSVQ
import numpy as np
import pandas as pd

import warnings
from numba.core.errors import NumbaPerformanceWarning
import sys
import copy
import os
warnings.simplefilter('ignore', NumbaPerformanceWarning)

arg1 = sys.argv[1] 

# Create data/tsvq_acovq_comparison directory if it doesn't exist
os.makedirs('./data/tsvq_acovq_comparison', exist_ok=True)

bit_allocation_list = {
    0: [1,1,1,1],
    1: [1,3],
    2: [2,2],
    3: [3,1],
    4: [4],
    5: [2, 1, 1],
    6: [1, 2, 1],
    7: [1, 1, 2],
}

epsilon_array = np.arange(0.00,0.11, 0.01)
delta_array = [0,5,10]
memory_array = [1]
export_dataframe = pd.DataFrame(columns=['bit_allocation', 'dimension', 'epsilon', 'delta', 'memory', 'snr_tsvq', 'snr_acovq', 'tsvq_centroids','acovq_equiv', 'acovq_centroids', 'tsvq_init', 'acovq_init'])

def calculate_centroid_difference(codebook1, codebook2, l2 = False, avg = False):
    keys = list(codebook1.keys())
    keys.remove('root')
    difference = 0
    max_difference = 0
    count = 0
    max_difference_key = None
    for key in keys:
        centroids1 = codebook1[key]
        centroids2 = codebook2[key]
        for i in range(len(centroids1)):
            if np.linalg.norm(centroids1[i] - centroids2[i], ord= 2 if l2 else 1) > max_difference:
                max_difference = np.linalg.norm(centroids1[i] - centroids2[i], ord= 2 if l2 else 1)
                max_difference_key = key
            count +=1
            difference += np.linalg.norm(centroids1[i] - centroids2[i], ord= 2 if l2 else 1)
    
    return difference/count if avg else (max_difference, max_difference_key)

for dimension in [1,4]:
    for delta in delta_array:
        for memory in memory_array:
            for epsilon in epsilon_array:
                channel_properties = {
                'epsilon': epsilon,
                'delta': delta,
                'memory': memory
                }
                
                quantizer = Generalized_Classes.WrapperTSVQ(1000000, 0, bit_allocation_list[int(arg1)], channel_properties, dimension, tsvq = True)
                source = quantizer.samples
                quantizer.train_quantizers()
                SNR = quantizer.calc_distortion()
                initialization = copy.deepcopy(quantizer.export_initialization_index())
                tsvq_codebook = copy.deepcopy(quantizer.export_centroids_index())
                tsvq_codebook_init = copy.deepcopy(quantizer.export_initialization_index())

                quantizer_acovq = Generalized_Classes.WrapperTSVQ(1000000, 0, bit_allocation_list[int(arg1)], channel_properties, dimension, tsvq = False, prev_tsvq_centroids=tsvq_codebook_init)
                quantizer_acovq.samples = source
                quantizer_acovq.train_quantizers()
                SNR2 = quantizer_acovq.calc_distortion()

                export_init1 = copy.deepcopy(quantizer.export_initialization_stage())
                export_init2 = copy.deepcopy(quantizer_acovq.export_initialization_stage())

                tsvq_centroids = copy.deepcopy(quantizer.export_centroids_index())
                codebook1 = copy.deepcopy(quantizer.export_centroids_index())
                codebook1 = TSVQ.tsvq_2_acovq_codebook(codebook1, codebook1, bit_allocation_list[int(arg1)], export_w_index=True)
                codebook2 = copy.deepcopy(quantizer_acovq.export_centroids_index())


                append_df = pd.DataFrame({
                    'bit_allocation': [bit_allocation_list[int(arg1)]],
                    'dimension': [dimension],
                    'epsilon': [epsilon],
                    'delta': [delta],
                    'memory': [memory],
                    'snr_tsvq': [SNR],
                    'snr_acovq': [SNR2],
                    'tsvq_centroids': [tsvq_centroids],
                    'acovq_equiv': [codebook1],
                    'acovq_centroids': [codebook2],
                    'acovq_init': [export_init2],
                    'tsvq_init': [export_init1]
                })
                
                export_dataframe = pd.concat([export_dataframe, append_df])
                export_dataframe.to_pickle('data/tsvq_acovq_comparison/tsvq_acovq_comparison' + str(bit_allocation_list[int(arg1)]) + '.pkl')

