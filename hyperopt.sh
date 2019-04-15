#!/bin/bash

# make sure to have -rwxrw-r-- by typing >> chmod u+x hyperopt.sh
# specify the space as used by hyperopt.py: e.g. one of [1k 5k 10k]
SPACE=$1
GOLD=$2
LOGTO=$3
echo "Usage: sh hyperopt.sh [dist_space_file] [goldstandard] [logto_folder]";


FLAT=(log log2 log10)
KC=(1 2 4 6 8)
PROJ=(6 8 10 12 14 16 18 20)
HASH=(4 8 12 16 20)

RUN=0
for F in ${FLAT[@]};
    do
    for K in ${KC[@]};
    	do
    	#for H in ${HASH[@]};
        for P in ${PROJ[@]};
    	    do
    	    #for P in ${PROJ[@]};
            for H in ${HASH[@]};
    	    	do
    	    	# performs a grid search with only one value for each parameter
    	    	# example log destination: ../gridsearch_5k 
    	    	RUN=$((RUN + 1))
    	    	echo \[bash script message\] starting run number ${RUN}
    	    	python3 hyperopt.py ${SPACE} "-testset" ${GOLD} "-logto" ${LOGTO} ${F} "-kc" ${K} ${K} 1 "-proj" ${P} ${P} 1 "-hash" ${H} ${H} 1 "-no-summary" "-v" &
    	    	done
            #wait #activate this wait-statement for runs with large spaces, e.g. 10k dims
            done
        wait
        done
    done

echo ... done. Number of runs: ${RUN}
