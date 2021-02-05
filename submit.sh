# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:8
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
# activate conda env
conda activate my_env  
# run script from above
python my_test_script_above.py
