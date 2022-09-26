#!/bin/bash
# file to manage running of slurm scripts

# expect 2 inputs
if [ $# != 3 ]
then
    echo "Usage: $0 <partition> <experiment folder> <configuration file>"
    echo "Example: $0 batch random/i_o configurations/random_i_o.json"
    exit
fi

# specify the filepath to use
filepath="./curriculum_models/$2"
# make the directories as needed
mkdir -p "$filepath"

# run the file
sbatch --partition=$1 --nodes=1 --job-name=$2 --output=$filepath/cluster_output.log --error=$filepath/cluster_error.log slurm.sh $2 $3
