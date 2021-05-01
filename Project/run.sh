#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=[name]
#SBATCH --mail-user=[your email]
#SBATCH --mail-type=BEGIN,END
#SBATCH --account=[account_name]
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --mem=92160m
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20


# The application(s) to execute along with its input arguments and options:
module purge
module load python3.8-anaconda
python [].py
