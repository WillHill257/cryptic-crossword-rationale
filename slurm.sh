#!/bin/bash

# run the script
cd $SLURM_SUBMIT_DIR # should be .../cryptic-crossword-rationale/

# activate python virtual environment
source env/bin/activate

# move to the folder where start running script
cd cryptics

# run the script
if [ $# != 2 ]
then
    echo "pass experiment_folder and configuration_file as arguments"
    exit
fi
python3 run_cryptics.py $1 $2

# deactivate python virtual environment
deactivate
