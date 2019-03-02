import os
import sys
import utils
import fcntl # for file locking of the output
import numpy as np
import MEN
from Fruitfly import Fruitfly # in order to workaround the classmethod issues

"""
this script is for hyperparameter optimizaton. 
It is basically brute-force grid search, but could be modified
to a kind of early-stopping grid search.
"""

if len(sys.argv) < 2 or sys.argv[1] not in ["bnc","wiki","rawiki","w2v","1k","5k","10k"]:
    print("Check your parameters! Parameter sequence: \n\
        hyperopt.py \n\
        [space]               one of these: [bnc wiki rawiki w2v 1k 5k 10k]\n\
        -logto [directory]    one file in [directory] per run\n\
                                 default: log/hyperopt/default_log\n\
        [flattenings]         any combination of [log log2 log10]\n\
                                 default: log\n\
        -kc [min max steps]   expansion factor\n\
                                 e.g. [2 10 2]; default: [5 5 1]\n\
        -proj [min max steps] number of projections\n\
                                 e.g. [2 10 2]; default: [5 5 1]\n\
        -hash [min max steps] percentage of 'winner' KCs\n\
                                 e.g. [4 20 4]; default: [5 5 1]\n\
        -no-summary           omit creation of a summary of all runs\n\
        -v                    run in verbose mode")
    sys.exit() 


#========== FUNCTIONS
def get_text_resources_from_argv():
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
    elif sys.argv[1] == "1k":
        data = "/home/simon.preissner/FFP/ukwac_100m/ukwac_1k_GS-checked.dm"
        column_labels= "/home/simon.preissner/FFP/ukwac_100m/ukwac_1k_GS-checked.cols"
        MEN_annot = "data/MEN_dataset_natural_form_full"
    elif sys.argv[1] == "5k":
        data = "/home/simon.preissner/FFP/ukwac_100m/ukwac_5k_GS-checked.dm"
        column_labels= "/home/simon.preissner/FFP/ukwac_100m/ukwac_5k_GS-checked.cols"
        MEN_annot = "data/MEN_dataset_natural_form_full"
    elif sys.argv[1] == "10k":
        data = "/home/simon.preissner/FFP/ukwac_100m/ukwac_10k_GS-checked.dm"
        column_labels= "/home/simon.preissner/FFP/ukwac_100m/ukwac_10k_GS-checked.cols"
        MEN_annot = "data/MEN_dataset_natural_form_full"
    else: 
        print("Error reading files. Is the key word correct?")
        sys.exit()
    return data, column_labels, MEN_annot

def get_ranges_from_argv(param, minimum=5, maximum=5, steps=1):
    if param in sys.argv:
        minimum = int(sys.argv[sys.argv.index(param)+1]) 
        maximum = int(sys.argv[sys.argv.index(param)+2]) 
        steps = int(sys.argv[sys.argv.index(param)+3])
    return minimum, maximum, steps 

def get_flattening_from_argv():
    flattening = []
    if "log" in sys.argv:
        flattening.append("log")
    if "log2" in sys.argv:
        flattening.append("log2")
    if "log10" in sys.argv:
        flattening.append("log10")
    if not flattening:
        flattening = ["log"] # flattening happens before the PN layer (ln, log2, log10) = 3params
    return flattening

def get_logging_from_argv():
    if "-logto" in sys.argv: 
        log_dest = sys.argv[sys.argv.index("-logto")+1]
    else: 
        log_dest = "../hyperopt_default_log"
    if not os.path.isdir(log_dest):
        os.makedirs(log_dest, exist_ok=True)
    return log_dest



def evaluate(orig_space, result_space, goldstd):
    sp_before = 0
    sp_after = 0
    sp_before, count_before = MEN.compute_men_spearman(orig_space, goldstd)
    sp_after, count_after = MEN.compute_men_spearman(result_space, goldstd)
    print("count_before:",count_before,"\tcount_after",count_after) #CLEANUP
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
    summarydump = log_dest+"/dump.txt"

    items = results["testset"]
    spb = round(results["sp_before"], 5)
    spa = round(results["sp_after"], 5)
    diff = round(results["sp_diff"], 5)
    
    specs_statement =   "PN_size \t" + str(pns)+\
                        "\nKC_factor\t" + str(kcs/pns)+\
                        "\nprojections\t"+ str(proj)+\
                        "\nhash_dims\t"+ str((hp*kcs)/100)+\
                        "\nflattening\t"+ flattening
    results_statement = "evaluated\t" + str(items)+\
                        "\nsp_before\t" + str(spb)+\
                        "\nsp_after\t" + str(spa)+\
                        "\nsp_diff \t" + str(diff)+"\n"

    with open(logfilepath, "w") as f, open(summarydump, "a") as d:
        fcntl.flock(d, fcntl.LOCK_EX) # for multiprocessing
        f.write("Evaluated corpus:\t"+data+"\n")
        f.write(specs_statement+"\n"+results_statement+"\n")
        d.write(specs_statement+"\n"+results_statement+"\n")
        fcntl.flock(d, fcntl.LOCK_UN)

        if (not (result_space is None) and (pair_cos is True)): 
            pairs, men_sim, fly_sim = MEN.compile_similarity_lists(result_space, goldstandard)
            for i in range(len(pairs)):
                f.write(str(pairs[i][0])+"\t"+str(pairs[i][1])+"\t"+\
                        str(men_sim[i])+"\t"+str(fly_sim[i])+"\t"+"\n")
    if verbose is True:
        print(specs_statement)
        print(results_statement)


#========== PARAMETER INPUT

data, column_labels, goldstandard = get_text_resources_from_argv()
log_dest = get_logging_from_argv()

flattening = get_flattening_from_argv()

kc_factor_min,   kc_factor_max,   kc_steps   = get_ranges_from_argv("-kc") #min=4,max=20,steps=4
projections_min, projections_max, proj_steps = get_ranges_from_argv("-proj") #min=4,max=20,steps=4
hash_perc_min,   hash_perc_max,   hash_steps = get_ranges_from_argv("-hash") #min=2,max=10,steps=2

in_space = utils.readDM(data) # returns dict of word : word_vector
i_to_cols, cols_to_i = utils.readCols(column_labels) # returns both-ways dicts of the vocabulary (word:pos_in_dict); important for maintenances
pn_size = len(i_to_cols) # length of word vector (= input dimension)

# for reporting purposes
verbose = "-v" in sys.argv
no_overall_summary_wanted = "-no-summary" in sys.argv
all_ff_specs = {}
internal_log = {}
sp_diffs = {}



#========== GRID SEARCH

#TODO maybe: sort the parameters by relevance
run = 0
for flat in flattening:
    for kc_factor in range(kc_factor_min, kc_factor_max+1, kc_steps):
        for projections in range(projections_min, projections_max+1, proj_steps):
            for hash_size in range(hash_perc_min, hash_perc_max+1, hash_steps):

                # make and apply fruitfly
                fruitfly = Fruitfly.from_scratch(pn_size, kc_factor*pn_size, projections, hash_size) # sets up a neural net
                run += 1
                if verbose is True:
                    print("Run number",run,"; config:", fruitfly.show_off(), "flattening:\t", flat)
                out_space = fruitfly.fly(in_space, flat) # this is where the magic happens 
                
                # evaluate
                internal_log[run] = evaluate(in_space, out_space, goldstandard)

                # log externally and internally
                log_results(internal_log[run], flat, fruitfly.get_specs(), log_dest, out_space)
                sp_diffs[run] = internal_log[run]["sp_diff"] # record all performances
                all_ff_specs[run] = fruitfly.get_specs()
                all_ff_specs[run]["flattening"] = flat

if verbose is True:
    print ("Finished grid search. Number of runs:",run)



#========== LOG FINAL RESULTS

if no_overall_summary_wanted is False:
    with open(log_dest+"/summary.txt", "w") as f:
        ranked_runs = sorted(sp_diffs, key=sp_diffs.get, reverse=True) #[:round(0.1*len(sp_diffs))+1]
        summary_header = "Grid search on the text data "+data+" with the following parameter ranges:\n"+\
                         "KC factor (min, max, steps): "+str(kc_factor_min)+" "+str(kc_factor_max)+" "+str(kc_steps)+"\n"+\
                         "projections (min, max, steps): "+str(projections_min)+" "+str(projections_max)+" "+str(proj_steps)+"\n"+\
                         "hash percent (min, max, steps): "+str(hash_perc_min)+" "+str(hash_perc_max)+" "+str(hash_steps)+"\n"+\
                         "flattening functions: "+", ".join(flattening)+"\n"\
                         "number of runs: "+str(len(ranked_runs))+"\n\n"
        f.write(summary_header)
        for run in ranked_runs:
            f.write(str(round(sp_diffs[run],5))+"\tconfig: "+str(all_ff_specs[run]))

        if verbose is True:
            print("Best runs by performance:")
            for run in ranked_runs[:min(10, int(round(len(ranked_runs)/10+1)))]:
                print("improvement:",round(sp_diffs[run],5),"with configuration:",all_ff_specs[run])

"""

"""




