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
    print("Check your parameters! Parameter sequence: hyperopt.py [corpus] [log_destination]")
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
    log_dest = None

in_space = utils.readDM(data) # returns dict of word : word_vector
i_to_cols, cols_to_i = utils.readCols(column_labels) # returns both-ways dicts of the vocabulary (word:pos_in_dict); important for maintenances


# params: corpus flattening kc_factor projections hash_percent
# corpus: UKWAC 100million

pn_size = len(in_space.popitem()[1]) # length of word vector (= input dimension)
kc_factor_min = 2 # number of Kenyon cells (in steps of 4) = 5params
kc_factor_max = 20
projections_min = 4 # number of connections to any given KC (in steps of 4) = 5params
projections_max = 20
hash_perc_min = 2 # percent of winners for the final hash (e.g. 2-10 in steps of 2) = 5params
hash_perc_max = 10
flattening = ["log", "log2", "log10"] # "log" -> implement multiple logs (ln, log2, log10) = 3params
all_ff_specs = {}
internal_log = {}
sp_diffs = {}




def evaluate(orig_space, result_space, goldstd, pair_cos=False):
    sp_before, count_before = MEN.compute_men_spearman(orig_space, goldstd)
    sp_after, count_after = MEN.compute_men_spearman(result_space, goldstd)
    sp_diff = sp_after-sp_before

    results = {"testset":count_after, "sp_before":sp_before, "sp_after":sp_after, "sp_diff":sp_diff}

    if (pair_cos):
        word_pairs, humans = MEN.readMEN(goldstd) # list of tuples (word1, word2) and list of floats
        pair_cosines = {}
        for i in range(len(word_pairs)):
            human=humans[i]
            a,b=word_pairs[i]
            if a in result_space and b in result_space:
                cos=utils.cosine_similarity(result_space[a], result_space[b])
                pair_cosines[a+"_"+b] = [round(human,5), round(cos,5), cos-human]
                #\#TODO get the values into a useful pattern/format
        return results, pair_cosines

    else: return results


def log_results(results, flattening, ff_config, pair_cosines=None, logfile=None, verbose=True):
    pns = ff_config["pn_size"]
    kcs = ff_config["kc_size"]
    proj= ff_config["proj_size"]
    hp  = ff_config["hash_percent"]
    if (logfile is None): # provide a filename containing all parameters
        logfile = str(int(kcs/pns))+"-"+str(proj)+"-"+str(int((hp*kcs)/100))+"-"+flattening+".txt"

    items = results["testset"]
    spb = round(results["sp_before"], 5)
    spa = round(results["sp_after"], 5)
    diff = round(results["sp_diff"], 5)
    
    specs_statement = "PN_size: " + str(pns)+\
                      "\nKC_fator: " + str(kcs/pns)+\
                      "\nprojections: "+ str(proj)+\
                      "\nhash_dims: "+ str((hp*kcs)/100)+\
                      "\nflattening: "+ flattening

    results_statement = "\nevaluated: " + str(items)+\
                        "\nsp_before: " + str(spb)+\
                        "\nsp_after: " + str(spa)+\
                        "\nsp_diff: " + str(diff)


    with open("log/results/"+logfile, "w") as f:
        f.write(specs_statement+"\n")
        f.write(results_statement+"\n")
        if not (pair_cosines is None): 
            for pair in sorted(pair_cosines.keys()): # log the pairwise cosines
                f.write(pair+" "+\
                        str(pair_cosines[pair][0])+" "+\
                        str(pair_cosines[pair][1])+" "+\
                        str(pair_cosines[pair][2])+"\n")
    if(verbose):
        print(specs_statement)
        print(results_statement, "\n")





""" gridsearch """
run = 0
for flat in flattening:
    for kc_factor in range(kc_factor_min, kc_factor_max+1, 4):
        for projections in range(projections_min, projections_max+1, 4):
            for hash_size in range(hash_perc_min, hash_perc_max+1, 2):

                # make and apply fruitfly
                fruitfly = Fruitfly.from_scratch(pn_size, kc_factor*pn_size, projections, hash_size) # sets up a neural net
                print("New fruitfly -- configuration: ", fruitfly.show_off())
                out_space = fruitfly.fly(in_space, flat) # this is where the magic happens 
                
                # evaluate
                internal_log[run], pair_cosines = evaluate(in_space, out_space, MEN_annot, pair_cos=True)

                # log externally and internally
                log_results(internal_log[run], flat, fruitfly.get_specs(), pair_cosines)
                sp_diffs[run] = internal_log[run]["sp_diff"] # record all performances
                all_ff_specs[run] = fruitfly.get_specs()
                all_ff_specs[run]["flattening"] = flat
                run += 1

print ("Finished grid search. Number of runs:",run)



""" Find the best 10% of runs """
best_runs = sorted(sp_diffs, key=sp_diffs.get, reverse=True)[:round(0.1*len(sp_diffs))+1]
print("configurations of the best 10 percent of runs:")
for run in best_runs:
    print("improvement:",sp_diffs[run],"with configuration:",all_ff_specs[run])




