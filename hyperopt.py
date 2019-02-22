import os
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

if len(sys.argv) < 2 or sys.argv[1] not in ["bnc","wiki","rawiki","w2v","5k"]:
    print("Check your parameters! Parameter sequence: \n\
        hyperopt.py \n\
        [corpus]              one of [bnc wiki rawiki w2v 5k]\n\
        -logto [directory]    one file in [directory] per run\n\
        [flattenings]         one or more of [log log2 log10] (default: log)\n\
                              (default: log/hyperopt/default_log)\n\
        -kc [min max steps]   expansion factor (default: 4, 20, 4)\n\
        -proj [min max steps] number of projections (default: 4, 20, 4)\n\
        -hash [min max steps] percentage of 'winner' KCs (default: 2, 10, 2)\n\
        -v                    run in verbose mode")
    sys.exit() 


#========== PARAMETER INPUT
if sys.argv[1] == "bnc":
    data = "data/BNC-MEN.dm"
    column_labels = "data/BNC-MEN.cols"
    MEN_annot = "data/MEN_dataset_lemma_form_full"
elif sys.argv[1] == "wiki":
    data = "data/wiki_all.dm"
    column_labels = "data/wiki_all.cols"
    MEN_annot = "data/MEN_dataset_natural_form_full"
elif sys.argv[1] == "rawiki":
    data = "data/wiki_abs-freq.dm"
    column_labels = "data/wiki_abs-freq.cols"
    MEN_annot = "data/MEN_dataset_natural_form_full"
elif sys.argv[1] == "w2v":
    data = "/home/simon.preissner/FFP/ukwac_100m/ukwac_100m_w2v_400.txt"
    column_labels= "/home/simon.preissner/FFP/ukwac_100m/ukwac.w2v.400.vocab"
    MEN_annot = "data/MEN_dataset_natural_form_full"
elif sys.argv[1] == "5k":
    data = "/home/simon.preissner/FFP/ukwac_100m/ukwac_5k.dm"
    column_labels= "/home/simon.preissner/FFP/ukwac_100m/ukwac_5k.cols"
    MEN_annot = "data/MEN_dataset_natural_form_full"

if "-logto" in sys.argv: 
    log_dest = sys.argv[sys.argv.index("-logto")+1]
else: 
    log_dest = "./log/hyperopt/default_log"
if not os.path.isdir(log_dest):
   os.makedirs(log_dest)

if "-kc" in sys.argv:
    kc_factor_min = int(sys.argv[sys.argv.index("-kc")+1]) 
    kc_factor_max = int(sys.argv[sys.argv.index("-kc")+2]) 
    kc_steps = int(sys.argv[sys.argv.index("-kc")+3]) 
else:
    kc_factor_min = 4 # number of Kenyon cells (in steps of 4) = 5params
    kc_factor_max = 20
    kc_steps = 4
if "-proj" in sys.argv:
    projections_min = int(sys.argv[sys.argv.index("-proj")+1]) 
    projections_max = int(sys.argv[sys.argv.index("-proj")+2]) 
    proj_steps = int(sys.argv[sys.argv.index("-proj")+3]) 
else:
    projections_min = 4 # number of connections to any given KC (in steps of 4) = 5params
    projections_max = 20
    proj_steps = 4
if "-hash" in sys.argv:
    hash_perc_min = int(sys.argv[sys.argv.index("-hash")+1]) 
    hash_perc_max = int(sys.argv[sys.argv.index("-hash")+2]) 
    hash_steps = int(sys.argv[sys.argv.index("-hash")+3]) 
else:
    hash_perc_min = 2 # percent of winners for the final hash (e.g. 2-10 in steps of 2) = 5params
    hash_perc_max = 10
    hash_steps = 2

flattening = []
if "log" in sys.argv:
    flattening.append("log")
if "log2" in sys.argv:
    flattening.append("log2")
if "log2" in sys.argv:
    flattening.append("log10")
if not flattening:
    flattening = ["log", "log2", "log10"] # flattening happens before the PN layer (ln, log2, log10) = 3params

if "-v" in sys.argv:
    verbose = True
else:
    verbose = False


in_space = utils.readDM(data) # returns dict of word : word_vector
i_to_cols, cols_to_i = utils.readCols(column_labels) # returns both-ways dicts of the vocabulary (word:pos_in_dict); important for maintenances
pn_size = len(i_to_cols) # length of word vector (= input dimension)

all_ff_specs = {}
internal_log = {}
sp_diffs = {}



#========== FUNCTIONS

def evaluate(orig_space, result_space, goldstd):
    sp_before = 0
    sp_after = 0
    sp_before, count_before = MEN.compute_men_spearman(orig_space, goldstd)
    sp_after, count_after = MEN.compute_men_spearman(result_space, goldstd)
    sp_diff = sp_after-sp_before

    results = {"testset":count_after, "sp_before":sp_before, "sp_after":sp_after, "sp_diff":sp_diff}
    return results


def log_results(results, flattening, ff_config, log_dest, result_space=None, pair_cos=True):
    pns = ff_config["pn_size"]
    kcs = ff_config["kc_size"]
    proj= ff_config["proj_size"]
    hp  = ff_config["hash_percent"]
    
    logfilepath = log_dest+"/"+sys.argv[1]+"-"+str(int(kcs/pns))+"-"\
                  +str(proj)+"-"+str(int((hp*kcs)/100))+"-"+flattening+".txt"


    items = results["testset"]
    spb = round(results["sp_before"], 5)
    spa = round(results["sp_after"], 5)
    diff = round(results["sp_diff"], 5)
    
    specs_statement = "PN_size\t" + str(pns)+\
                      "\nKC_factor\t" + str(kcs/pns)+\
                      "\nprojections\t"+ str(proj)+\
                      "\nhash_dims\t"+ str((hp*kcs)/100)+\
                      "\nflattening\t"+ flattening
    results_statement = "evaluated\t" + str(items)+\
                        "\nsp_before\t" + str(spb)+\
                        "\nsp_after\t" + str(spa)+\
                        "\nsp_diff\t" + str(diff)+"\n"

    with open(logfilepath, "w") as f:
        f.write("Evaluated corpus:\t"+data+"\n")
        f.write(specs_statement+"\n")
        f.write(results_statement+"\n")

        if (not (result_space is None) and (pair_cos is True)): 
            pairs, men_sim, fly_sim = MEN.compile_similarity_lists(result_space, MEN_annot)
            for i in range(len(pairs)):
                f.write(str(pairs[i][0])+"\t"+str(pairs[i][1])+"\t"+\
                        str(men_sim[i])+"\t"+str(fly_sim[i])+"\t"+"\n")
    if verbose is True:
        print(specs_statement)
        print(results_statement, "\n")





#========== GRID SEARCH
#TODO maybe: sort the parameters by relevance
run = 0
for flat in flattening:
    for kc_factor in range(kc_factor_min, kc_factor_max+1, kc_steps):
        for projections in range(projections_min, projections_max+1, proj_steps):
            for hash_size in range(hash_perc_min, hash_perc_max+1, hash_steps):

                # make and apply fruitfly
                fruitfly = Fruitfly.from_scratch(pn_size, kc_factor*pn_size, projections, hash_size) # sets up a neural net
                if verbose is True:
                    print("New fruitfly -- configuration: ", fruitfly.show_off(), "flattening:\t", flat)
                out_space = fruitfly.fly(in_space, flat) # this is where the magic happens 
                
                # evaluate
                internal_log[run] = evaluate(in_space, out_space, MEN_annot)

                # log externally and internally
                log_results(internal_log[run], flat, fruitfly.get_specs(), log_dest, out_space)
                sp_diffs[run] = internal_log[run]["sp_diff"] # record all performances
                all_ff_specs[run] = fruitfly.get_specs()
                all_ff_specs[run]["flattening"] = flat
                run += 1

if verbose is True:
    print ("Finished grid search. Number of runs:",run)



#========== LOG FINAL RESULTS
with open(log_dest+"/summary.txt", "w") as f:
    ranked_runs = sorted(sp_diffs, key=sp_diffs.get, reverse=True) #[:round(0.1*len(sp_diffs))+1]
    f.write("sorted list of runs on "+data+" ("+len(ranked_runs)+" runs):")
    for run in ranked_runs:
        f.write(str(round(sp_diffs[run],5))+"\tconfig: "+str(all_ff_specs[run]))
        if verbose is True:
            print("configurations of the best 10 percent of runs:")
            print("improvement:",round(sp_diffs[run],5),"with configuration:",all_ff_specs[run])


"""
"""




