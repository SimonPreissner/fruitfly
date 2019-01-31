import sys
import utils
import numpy as np
import MEN
from Fruitfly import Fruitfly # in order to workaround the classmethod issues

"""
this script is for hyperparameter optimizaton. 
It is basically brute-force grid search, but could be modified
to a kind of early-stopping grid search.
"""


if len(sys.argv) < 1 or sys.argv[1] not in ("bnc","wiki","rawiki"):
    print("Check your parameters! Parameter sequence: hyperopt.py [corpus] [output_file]")
    sys.exit() 

if sys.argv[1] == "bnc":
    data = "data/BNC-MEN.dm"
    column_labels = "data/BNC-MEN.cols"
    MEN_annot = "data/MEN_dataset_lemma_form_full"
elif sys.argv[1] == "wiki":
    data = "data/wiki_all.dm"
    column_labels = "data/wiki_all.cols"
    MEN_annot = "data/MEN_dataset_natural_form_full"
else:
    data = "data/wiki_abs-freq.dm"
    column_labels = "data/wiki_abs-freq.cols"
    MEN_annot = "data/MEN_dataset_natural_form_full"


if len(sys.argv) > 2: 
    log_dest = sys.argv[2]
else: 
    log_dest = performance_log.txt

in_space = utils.readDM(data) # returns dict of word : word_vector
i_to_cols, cols_to_i = utils.readCols(column_labels) # returns both-ways dicts of the vocabulary (word:pos_in_dict); important for maintenances



pn_size = len(in_space.popitem()[1]) # length of word vector (= input dimension)
kc_factor_min = 2 # number of Kenyon cells
kc_factor_max = 20
projections_min = 4 # number of connections to any given KC
projections_max = 20
hash_perc_min = 2 # percent of winners for the final hash
hash_perc_max = 20
flattening = "log" # sigmoid # softmax
all_ff_specs = {}
internal_log = {}
sp_diffs = {}




def evaluate(orig_space, result_space, goldstd, verbose=False, logfile="performance_log.txt"):
    sp_before, count_before = MEN.compute_men_spearman(orig_space, goldstd)
    sp_after, count_after = MEN.compute_men_spearman(result_space, goldstd)
    sp_diff = sp_after-sp_before

    results_statement = "evaluated items: " + str(count_after)+\
                        "\tsp_before: " + str(round(sp_before,5))+\
                        "\tsp_after: " + str(round(sp_after,5))+\
                        "\tsp_difference: " + str(round(sp_diff,5))

    # TODO extract this so that it only opens once
    with open("log/results/"+logfile, "a") as f:
        f.write(fruitfly.show_off()+"\n")
        f.write(results_statement+"\n")
    f.close()
    if(verbose):
        print(fruitfly.show_off())
        print(results_statement, "\n")

    return {"testset":count_after, "sp_before":sp_before, "sp_after":sp_after, "sp_diff":sp_diff}



""" gridsearch """
run = 0
for kc_factor in range(kc_factor_min, kc_factor_max+1):
    for projections in range(projections_min, projections_max+1):
        for hash_size in range(hash_perc_min, hash_perc_max+1):

            fruitfly = Fruitfly.from_scratch(pn_size, kc_factor*pn_size, projections, hash_size) # sets up a neural net
            print("New fruitfly -- configuration: ", fruitfly.show_off())
            out_space = fruitfly.fly(in_space, flattening) # this is where the magic happens 
            
            all_ff_specs[run] = fruitfly.get_specs() # record all training runs
            internal_log[run] = evaluate(in_space, out_space, MEN_annot, verbose=True, log_dest)
            sp_diffs[run] = internal_log[run]["sp_diff"] # record all performances
            run += 1

print ("Finished grid search. Number of runs:",run)



""" Find the best 10% of runs """
best_runs = sorted(sp_diffs, key=sp_diffs.get, reverse=True)[:round(0.1*len(sp_diffs))+1]
print("configurations of the best runs:")
for run in best_runs:
    print("improvement:",sp_diffs[run],"with configuration:",all_ff_specs[run])




