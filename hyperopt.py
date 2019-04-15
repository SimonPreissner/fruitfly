"""Countwords: Create or extend a co-occurrence matrix (and optionally a corresponding fruitfly).

Usage:
  hyperopt.py [--help] 
  hyperopt.py <space> <testset> 
  hyperopt.py <space> <testset> [--logto <log_dir>] [-v] [--no-summary]

Options:
  -h --help             show this screen
  -v --verbose          comment program status with command-line output
  --logto=<log_dir>     creates one file per run in the specified folder
  --no-summary          omit the creation of a final summary file 
  --flat=<flattenings>  flattening function(s), one of [log log2 log10], dash-separated [default: log]
  --k1=<kc_min>          KC expansion factor, min< [default: 5]
  --k2=<kc_max>                               maximum [default: 5]
  --k3=<kc_step>                            step size [default: 1]
  --p1=<proj_min>        projection numbers per KC, minimum [default: 5]
  --p2=<proj_max>                                   maximum [default: 5]
  --p3=<proj_step>                                step size [default: 1]
  --r1=<hash_min>        percentages for hashing, minimum [default: 5]
  --r2=<hash_max>                                 maximum [default: 5]
  --r3=<hash_step>                              step size [default: 1]
  
from docopt import docopt

if __name__ == '__main__':
    arguments = docopt(__doc__)
    print(arguments)
    sys.exit() # BREAKPOINT
"""
#  hyperopt.py <space> <testset> [--flat <flat_range>...] [-k <kc_range>...] [-p <proj_range>...] [-r <hash_range>...]  
#TODO solve the problem of how to input multiple optional parameters ()



"""
this script is for hyperparameter optimizaton. 
It is basically brute-force grid search, but could be modified
to a kind of early-stopping grid search.
"""
import sys

if len(sys.argv) < 2: #or sys.argv[1] not in ["bnc","wiki","rawiki","w2v","1k","5k","10k"]:
    print("Check your parameters! Parameter sequence: \n\
        hyperopt.py \n\
        [space]               e.g. one of these: [bnc wiki rawiki w2v 1k 5k 10k]\n\
        -testset [filepath]   default: data/MEN_dataset_natural_form_full\n\
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

import os
import utils
import fcntl # for file locking of the output
import numpy as np
import MEN
from tqdm import tqdm
from Fruitfly import Fruitfly # in order to workaround the classmethod issues




#========== FUNCTIONS
def get_text_resources_from_argv():
    if sys.argv[1] == "bnc":
        data = "data/BNC-MEN.dm"
        column_labels = "data/BNC-MEN.cols"
    elif sys.argv[1] == "wiki":
        data = "data/wiki_all.dm"
        column_labels = "data/wiki_all.cols"
    elif sys.argv[1] == "rawiki":
        data = "data/wiki_abs-freq.dm"
        column_labels = "data/wiki_abs-freq.cols"
    elif sys.argv[1] == "w2v":
        data = "/home/simon.preissner/FFP/ukwac_100m/ukwac_100m_w2v_400.txt"
        column_labels= "/home/simon.preissner/FFP/ukwac_100m/ukwac.w2v.400.vocab"
    elif sys.argv[1] == "1k":
        data = "/home/simon.preissner/FFP/ukwac_100m/ukwac_1k_GS-checked.dm"
        column_labels= "/home/simon.preissner/FFP/ukwac_100m/ukwac_1k_GS-checked.cols"
    elif sys.argv[1] == "5k":
        data = "/home/simon.preissner/FFP/ukwac_100m/ukwac_5k_GS-checked.dm"
        column_labels= "/home/simon.preissner/FFP/ukwac_100m/ukwac_5k_GS-checked.cols"
    elif sys.argv[1] == "10k":
        data = "/home/simon.preissner/FFP/ukwac_100m/ukwac_10k_GS-checked.dm"
        column_labels= "/home/simon.preissner/FFP/ukwac_100m/ukwac_10k_GS-checked.cols"
    else: 
        data = sys.argv[1]+".dm"
        column_labels = sys.argv[1]+".cols"
        print("Could not identify a keyword for the vector space file.\n\
            Proceeding with  ",data,"\nand  ",column_labels,"  as resources.")
        #sys.exit() #used to terminate if the keyword was incorrect.
    return data, column_labels

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

def get_testset_from_argv():
    if "-testset" in sys.argv: 
        testfile = sys.argv[sys.argv.index("-testset")+1]
    else: 
        testfile = "data/MEN_dataset_natural_form_full"
    return testfile

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
    sp_diff = sp_after-sp_before

    return (count_before, count_after, sp_before, sp_after, sp_diff)

def log_results(results, flattening, ff_config, log_dest, result_space=None, pair_cos=True):
    pns = ff_config["pn_size"]
    kcs = ff_config["kc_size"]
    proj= ff_config["proj_size"]
    hp  = ff_config["hash_percent"]
    
    logfilepath = log_dest+"/"+str(int(kcs/pns))+"-"\
                  +str(proj)+"-"+str(int((hp*kcs)/100))+"-"+flattening+".txt"
    summarydump = log_dest+"/dump.txt"

    countb = results[0]
    counta = results[1]
    spb = round(results[2], 5)# results["sp_before"]
    spa = round(results[3], 5)# results["sp_after"]
    diff = round(results[4], 5)# results["sp_diff"]
    
    specs_statement =   "PN_size \t" + str(pns)+\
                        "\nKC_factor\t" + str(kcs/pns)+\
                        "\nprojections\t"+ str(proj)+\
                        "\nhash_dims\t"+ str((hp*kcs)/100)+\
                        "\nflattening\t"+ flattening
    results_statement = "testwords_before\t" + str(countb)+\
                        "testwords_after\t" + str(counta)+\
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

data, column_labels= get_text_resources_from_argv()
goldstandard = get_testset_from_argv()
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
sp_vals = {}



#========== GRID SEARCH

#TODO maybe: sort the parameters by relevance
run = 0
for flat in flattening:
    for kc_factor in range(kc_factor_min, kc_factor_max+1, kc_steps):
        for projections in range(projections_min, projections_max+1, proj_steps):
            for hash_size in range(hash_perc_min, hash_perc_max+1, hash_steps):

                # make and apply fruitfly
                fruitfly = Fruitfly.from_scratch(pn_size, kc_factor*pn_size, projections, hash_size) # sets up a neural net
                #fruitfly = Fruitfly.from_config(...) # for using the same fruitfly
                run += 1
                if verbose is True:
                    print("Run number {0}; config: {1} flattening:\t{2}".format(run, fruitfly.show_off(), flat))
                out_space, space_dic, space_ind, t_flight = fruitfly.fly(in_space, cols_to_i, flat) # this is where the magic happens 
                
                # evaluate
                internal_log[run] = evaluate(in_space, out_space, goldstandard)

                # log externally and internally
                log_results(internal_log[run], flat, fruitfly.get_specs(), log_dest, out_space)
                #sp_diffs[run] = (internal_log[run]["sp_before"], #CLEANUP
                #                internal_log[run]["sp_diff"],
                #                internal_log[run]["sp_diff"]) # record all performances
                all_ff_specs[run] = fruitfly.get_specs()
                all_ff_specs[run]["flattening"] = flat

if verbose is True:
    print ("Finished grid search. Number of runs:",run)



#========== LOG FINAL RESULTS

if no_overall_summary_wanted is False:
    with open(log_dest+"/summary.txt", "w") as f:
        ranked_res = sorted(internal_log.items(), key=lambda x:x[1][3], reverse=True) # data type: [(run,(results))]
        summary_header = 
            "Grid search on the text data "+data+" with the following parameter ranges:\n"+\
            "KC factor (min, max, steps): {0} {1} {2}\n".format(kc_factor_min, kc_factor_max, kc_steps)+\
            "projections (min, max, steps): {0} {1} {2}\n".format(projections_min, projections_max, proj_steps)+\
            "hash percent (min, max, steps): {0} {1} {2}\n".format(hash_perc_min, hash_perc_max, hash_steps)+\
            "flattening functions: "+", ".join(flattening)+"\n"\
            "number of runs: "+str(len(ranked_res))+"\n\n"


        f.write(summary_header)
        for run in ranked_res:
            f.write("{0}\t{1}\t{2}\tconfig: {3}".format(internal_log[run[0]][2], 
                                                        internal_log[run[0]][3],
                                                        internal_log[run[0]][4],
                                                        all_ff_specs[run[0]]))

        if verbose is True:
            print("Best runs by performance:")
            for run in ranked_res[:min(10, int(round(len(ranked_res)/10+1)))]:
                print("improvement:",round(internal_log[run[0]][4],5),"with configuration:",all_ff_specs[run[0]])

"""

"""




