"""This bash script runs the simulations over various values of lambda and q.

This bash script calls the python script `simulate_and_score.py`.
This script is meant to parallelize runs over values of lambda and q on a 
TORQUE system. This bash script consumes a file `lambda_q_combinations.txt` 
that contains a different tabbed combination of lambda and number of states on 
every line, for all combinations of interest. q = 1/(number of states), as we
are using a uniform state distribution. An example run would then use:

    qsub -t 1-108 run_sim_vary_lambda_q.sh

if there are 108 lines in `lambda_q_combinations.txt`. For each arrayed job
generated, the simulations for that lambda and number of states combination will
be run with all combinations of algorithms, topologies, and reconstruction metrics
specified. Populates a folder in `results/vary_lambda_q`, which should be made
prior to running.
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
topologies=(exponential_plus_c)
algs=(percolation)
metrics=(rf)

### Specify the depth of the tree, with
depth=9

for topology in ${topologies[@]};
do
    for alg in ${algs[@]};
    do

        ### Make the folder for this topology if it doesn't exist
        if [ ! -d "$workdir/results/vary_lambda_q/${topology}" ]; then
            mkdir -p "$workdir/results/vary_lambda_q/${topology}";
        fi

        for metric in ${metrics[@]};
        do
            ### Skip over combinations that are not meant to be run
            if [[ $alg == "shared_mutation" ]] && [[ $metric == "d_triplets" ]]; then
                echo "skipped d_triplets + shared_mutation combo"
                continue
            fi

            if [[ $alg == "shared_mutation" ]] && [[ $metric == "d_triplets_20" ]]; then
                echo "skipped d_triplets_20 + shared_mutation combo"
                continue
            fi

            ### Read in lambda and q values from line t from the file of 
            ### lambda and q combinations
            content=`sed -n "$t p" $workdir/lambda_q_combinations.txt`
            lamb=$(cut -f1 <<<"$content")
            num_states=$(cut -f2 <<<"$content")

            ### Make the folder for this alg/metric combination if it doesn't exist
            if [ ! -d "$workdir/results/vary_lambda_q/${topology}/${alg}_${metric}" ]; then
                mkdir -p "$workdir/results/vary_lambda_q/${topology}/${alg}_${metric}";
            fi

            ### Generate the output file at the output folder, with a name 
            ### specified by the job index
            SAVE_FILE="$workdir/results/vary_lambda_q/${topology}/${alg}_${metric}/${t}.txt"

            ### Run the simulation script with the specified parameters, logging the command
            cmd="python ${SIM_SCRIPT} $lamb $num_states $depth $topology $alg $metric ${SAVE_FILE}"
            echo $cmd
            ${cmd}
        done
    done
done