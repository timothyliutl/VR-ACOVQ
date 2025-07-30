# VR-ACOVQ Simulation Code
This repository contains code used to run simulation results for the performance of variable rate adaptive channel optimized vector quantizers. All class definitions are stored in the `TSVQ_Modules/` folder. To run a simulation, execute the `vrq_collection.py` file and provide the index corresponding to the bit allocation dictionary within the Python file. Additionally, shell scripts are included in the `shell_scripts/` folder for use on CAC servers.


## Setup and Running Instructions

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU (for CUDA acceleration)
- Required packages listed in `requirements.txt`

### Installation
1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv310
   source venv310/bin/activate  # On Windows: venv310\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulation
1. **Local execution**: To run a simulation with a specific bit allocation index (0-7), use one of the following commands:
   ```bash
   python vrq_collection.py <index>
   ```
   or
   ```bash
   python data_collection_tsvq_acovq.py <index>
   ```

2. **Cluster execution (CAC servers)**: Use the SLURM script for parallel processing:
   To run on the CAC cluster using SLURM, use the appropriate shell script depending on your simulation:

   **For VRQ simulations:**
   ```bash
   cd shell_scripts
   sbatch acovq.sh
   ```

   **For TSVQ/ACOVQ comparison simulations:**
   ```bash
   cd shell_scripts
   sbatch comparison.sh
   ```

### Bit Allocation Configuration
The bit allocation can be modified directly in the `vrq_collection.py` or `data_collection_tsvq_acovq.py` file. The current configurations include various 4-bit bit allocations, and you can add or modify these according to your needs.

### Results
Simulation results are automatically saved as pickle files in either the `data/vrq/` folder or the `data/tsvq_acovq_comparison/` folder, depending on which script you run. 

- For VRQ simulations, results are saved in `data/vrq/` with filenames in the format: `vrq{correlation}_{source_type}_{date}_{bit_allocation}.pkl`
- For TSVQ/ACOVQ comparison simulations, results are saved in `data/tsvq_acovq_comparison/` with filenames in the format: `tsvq_acovq_comparison[bit_allocation].pkl`

