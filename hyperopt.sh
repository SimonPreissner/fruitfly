#!/bin/bash

# make sure to have -rwxrw-r-- by typing >> chmod u+x hyperopt.sh
# specify the space as used by hyperopt.py: e.g. one of [1k 5k 10k]
SPACE=$1
LOGTO=$2
echo "Usage: sh hyperopt.sh [dist_space_file] [logto_folder]";


FLAT=(log log2 log10)
KC=(1 2 4)
PROJ=(6 8 10 12 14 16)
HASH=(4 8 12 16 20)

RUN=0
for F in ${FLAT[@]};
    do
    for K in ${KC[@]};
    	do
    	for P in ${PROJ[@]};
    	    do
    	    for H in ${HASH[@]};
    	    	do
    	    	# performs a grid search with only one value for each parameter
    	    	# example log destination: ../gridsearch_5k 
    	    	RUN=$((RUN + 1))
    	    	echo \[bash script message\] starting run number ${RUN}
    	    	python3 hyperopt.py ${SPACE} "-logto" ${LOGTO} ${F} "-kc" ${K} ${K} 1 "-proj" ${P} ${P} 1 "-hash" ${H} ${H} 1 "-no-summary" "-v" &
    	    	done
    	    done
	    wait
        done
    done

echo \[bash script message\] done. Number of runs: ${RUN}
