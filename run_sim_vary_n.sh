"""This bash script runs the simulations over various values of n

This bash script calls the python script `simulate_and_score.py`.
This script is meant to parallelize runs over values of n on a 
TORQUE system. This bash script consumes a file `depths.txt` 
that contains a different depth on every line. Depth is used to represent the
number of leaves, with n = 2**(depth - 1). An example run would then use:

    qsub -t 1-10 run_sim_vary_n.sh

if there are 10 lines in `depths.txt`. For each arrayed job generated, the 
simulations for that n will be run with all combinations of algorithms, 
topologies, and reconstruction metrics specified. Populates a folder 
`results/vary_n`, which should be made prior to running.
"""

#!/bin/bash

# *** "#PBS" lines must come before any non-blank, non-comment lines ***### Set the job name
### Job's Name
#PBS -N Theory Paper simulation
### Redirect stdout and stderr by first telling Torque to redirect do /dev/null and then redirecting yourself via exec. This is the way the IT recommends.
#PBS -e localhost:/dev/null
#PBS -o localhost:/dev/null
### Set the queue to which to send
#PBS -q yosef3
### Limit the resources used
#PBS -l nodes=1:ppn=1
### Change the walltime and cpu time limit from their default (the default is currently an hour)
#PBS -l walltime=2000:00:00
#PBS -l cput=20000:00:00
### Move all your environment variables to the job
#PBS -V


### Change to the directory where the job was submitted
workdir=$PBS_O_WORKDIR

### Get the array ID
t=$PBS_ARRAYID

### Set the log location to record stderr and stdout
exec 2> $workdir/log/_log.stderr_$t > $workdir/log/_log.stdout_$t

### Specify the python script that runs the simulation script
SIM_SCRIPT="$workdir/scripts/simulate_and_score.py"

### Specify the strings indicating the topologies, algorithms, and 
### reconstruction metrics to be included in the run
topologies=(complete_binary)
algs=(shared_mutation) 
metrics=(rf)

### Specify the lambda parameter and the number of states (1/q) in the uniform
### state distribution
lamb=0.5
num_states=20

for topology in ${topologies[@]};
do
    for alg in ${algs[@]};
    do

        ### Make the folder for this topology if it doesn't exist
        if [ ! -d "$workdir/results/vary_n/${topology}" ]; then
            mkdir -p "$workdir/results/vary_n/${topology}";
        fi


        for metric in ${metrics[@]};
        do

            ### Read in the depth from line t from the file of depths indicating
            ### tree size
            content=`sed -n "$t p" $workdir/depths.txt`
            depth=$(cut -f1 <<<"$content")

            ### Generate the output file at the output folder. All arrayed jobs
            ### will write to the same output file
            SAVE_FILE="$workdir/results/vary_n/${topology}/${alg}_${metric}.txt"

            ### Run the simulation script with the specified parameters, logging the command
            cmd="python ${SIM_SCRIPT} $lamb $num_states $depth $topology $alg $metric ${SAVE_FILE}"
            echo $cmd
            ${cmd}
        done
    done
done