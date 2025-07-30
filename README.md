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
1. **Local execution**: Run with a specific bit allocation index (0-7):
   ```bash
   python vrq_collection.py <index>
   ```

2. **Cluster execution (CAC servers)**: Use the SLURM script for parallel processing:
   ```bash
   cd shell_scripts
   sbatch script.sh
   ```

### Bit Allocation Configuration
The bit allocation can be modified directly in the `vrq_collection.py` file. The current configurations include various 4-bit bit allocations, and you can add or modify these according to your needs.

### Results
Simulation results are automatically saved as pickle files in the `data/vrq/` folder. The files are named with the format: `vrq{correlation}_{source_type}_{date}_{bit_allocation}.pkl`

