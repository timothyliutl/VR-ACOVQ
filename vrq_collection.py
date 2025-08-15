import TSVQ_Modules.Generalized_Classes as Generalized_Classes
import TSVQ_Modules.COVQ as COVQ
import TSVQ_Modules.TSVQ as TSVQ
import TSVQ_Modules.VariableRateQuantizer as VariableRateQuantizer
import numpy as np
import pandas as pd
import gc
import TSVQ_Modules.TestingVRQ as TestingVRQ

import warnings
from numba.core.errors import NumbaPerformanceWarning
import math
import sys
import os
warnings.simplefilter('ignore', NumbaPerformanceWarning)
from datetime import datetime

arg1 = int(sys.argv[1])
folder_name = datetime.now().strftime("%Y-%m-%d")

# Create data/vrq directory if it doesn't exist
os.makedirs('./data/vrq', exist_ok=True)

bit_allocation_list = {
    0: [1,1,1,1],
    1: [1,3],
    2: [2,2],
    3: [3,1],
    4: [4],
    5: [2, 1, 1],
    6: [1, 2, 1],
    7: [1, 1, 2]
}

# bit_allocation_list = {
#     0: [3,1,1,1],
#     1: [1,3,1,1],
#     2: [1,1,3,1],
#     3: [1,1,1,3],
#     4: [2,2,1,1],
#     5: [1,2,2,1],
#     6: [1,1,2,2],
#     7: [4,1,1],
#     8: [1,4,1],
#     9: [1,1,4]
# }


export_dataframe = pd.DataFrame(columns=['bit_allocation', 'dimension', 'epsilon', 'delta', 'memory', 'vrq_snr', 'frq_snr', 'vrq_distortion', 'frq_distortion', 'bit_allocation_list', 'balanced_tree'])

epsilon_array = np.array([0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])

delta_array = [0,5, 10]
epsilon_array = np.concatenate([epsilon_array, epsilon_array[::-1], epsilon_array, epsilon_array[::-1]])

#change these parameters to test different values
correlation = 0
source_type = 'gaussian'

for dimension in [1,2,4]:
    for delta in delta_array:
        for epsilon_index in range(len(epsilon_array)):
            channel_properties = {
                'epsilon': epsilon_array[epsilon_index],
                'delta': delta,
                'memory': 1
            }
            sequence = epsilon_array[:epsilon_index+1]
            if source_type == 'laplacian':
                source = np.random.laplace(0,1, 4000000*dimension).reshape(4000000, dimension)
            elif source_type == 'gaussian':
                source = COVQ.generate_gaussian_source(4000000, dimension, correlation)
        
            vrq = VariableRateQuantizer.VariableRateQuantizer(source, bit_allocation_list[arg1], channel_properties, epsilon_array = sequence, fixed_rate= False)
            distortion = vrq.calc_distortion()

            frq = VariableRateQuantizer.VariableRateQuantizer(source, bit_allocation_list[arg1], channel_properties, epsilon_array = sequence, fixed_rate= True)
            frq_distortion = frq.calc_distortion()

            #copying centroids to dataframe
            test_vrq = TestingVRQ.TestVRQ(channel_properties)
            test_vrq.copy_centroids(test_vrq.root, vrq.root)

            test_frq = TestingVRQ.TestVRQ(channel_properties)
            test_frq.copy_centroids(test_frq.root, frq.root)

            append_df = pd.DataFrame({
                            'bit_allocation': [bit_allocation_list[int(arg1)]],
                            'dimension': [dimension],
                            'epsilon': [epsilon_array[epsilon_index]],
                            'delta': [delta],
                            'memory': [1],
                            'vrq_snr': [-10*math.log10(distortion/np.var(source))],
                            'frq_snr': [-10*math.log10(frq_distortion/np.var(source))],
                            'vrq_distortion': [distortion],
                            'frq_distortion': [frq_distortion],
                            'bit_allocation_list': [vrq.bit_allocation_history],
                            'bit_average': [vrq.average_bit_history],
                            'balanced_tree': [vrq.is_balanced()],
                            'source': [source_type],
                            'correlation': [correlation],
                            'vrq_test': [test_vrq],
                            'frq_test': [test_frq],
                        })
            export_dataframe = pd.concat([export_dataframe, append_df])
            export_dataframe.to_pickle('./data/vrq/vrq' + f'c{correlation}_{source_type}_' + folder_name + str(bit_allocation_list[int(arg1)]) + '.pkl')
            del vrq
            del frq
            del test_frq
            del test_vrq
            del source
            gc.collect()
        
