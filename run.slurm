#!/bin/bash
# These lines are for slurm

# specify a partition by its name in which to run your job.
#SBATCH -p batch
# specify number of nodes you are requesting to run your job
#SBATCH -N 1
# specify number of tasks/processes
##SBATCH -n 8
# specify the job name
#SBATCH -J cryptics

# specify the filename to be used for writing output.
# You need to put the correct path and the output file name. %N is name of node; %j is job number
#SBATCH -o /home-mscluster/whill/research/cryptic-crossword-rationale/experiments/cluster_output.txt
# specify the filename for stderr
#SBATCH -e /home-mscluster/whill/research/cryptic-crossword-rationale/experiments/cluster_error.txt

# The following are shell commands -- which will be executed according to the settings specified above

# run the script
cd $SLURM_SUBMIT_DIR # should be .../cryptic-crossword-rationale/

# activate python virtual environment
source env/bin/activate

# move to the folder where start running script
cd cryptics

# run the script
echo $1
echo $2
exit
python3 run_cryptics.py $1 $2

# move the output and error files
mv ~/research/cryptic-crossword-rationale/experiments/cluster_output.txt ~/research/cryptic-crossword-rationale/experiments/$1/cluster_output.txt
mv ~/research/cryptic-crossword-rationale/experiments/cluster_error.txt ~/research/cryptic-crossword-rationale/experiments/$1/cluster_error.txt

# deactivate python virtual environment
deactivate