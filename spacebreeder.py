"""
THE DOCOPT SECTION IS UNDER CONSTRUCTION

Spacebreeder: apply a full incrementality pipeline 

Usage:
  spacebreeder.py <texts> <space> <fly_file> ([setup] | [options])

Options:
  -h --help          Show this screen.
  -v --verbose       Output most important dimensions per word.
  -t=<teststeps>     test after a certain number of incremental runs
  
  --eval=<testset>  

OBACHT!
  # Use the file names for <space> and <testset> WITHOUT file extension!
from docopt import docopt
  
"""

import sys
import utils
frim utils import timeit
import time
import Incrementor
from Incrementor import Incrementor
import Fruitfly
from Fruitfly import Fruitfly
import MEN
import numpy as np


#========== PARAMETER INPUT
corpus_file = "ukwac_100m_linewise.txt" #TODO set the correct path
matrix_file = "spaces/unhashed/" #TODO set the correct path
fly_location = "log/configs/" #TODO set the correct path
space_location = "spaces/hashed/" #TODO set the correct path for the hashed spaces

testfile = "data/MEN_pairs" #TODO set the correct path
results_location = "log/results/" #TODO set the correct path
vip_words_location = "log/results/" #TODO set the correct path
number_of_vip_words = 20
test_interval_in_words = 1000000 
allow_disconnection = True #TODO implement disconnection of least successful KCs

tokenize = True
linewise = False # DON'T SET THIS TO True! It will break the loop.
verbose  = True
max_dims = None

window = 5


#========== SETUP INITIAL FRUIT FLY
""" 
This step is taken because Incrmentor can't do custom initialization 
"""
initial_fly_file = fly_location+"first_fly.cfg" #TODO set the correct path
p,k,c,r,flat = 50, 30000, 6, 5, "log" #TODO set the parameters (according to the grid search results?)
first_fly = Fruitfly.from_scratch(pn_size=p, kc_size=k, proj_size=c, hash_percent=r):
first_fly.log_params(filename=initial_fly_file, timestamp=False):

#========== SETUP INCREMENTOR
"""
matrix_incremental=False because it would need to load an existing matrix;
however, in this setup, it can increment all the way from an initial one.
"""
breeder = Incrementor(corpus_file, matrix_file,
        corpus_tokenize=tokenize, corpus_linewise=linewise, corpus_checkvoc=testfile,
        matrix_incremental=False, matrix_maxdims=max_dims,
        fly_new=False, fly_grow=True, fly_file=initial_fly_file, 
        verbose=verbose)

#OBACHT: don't use freq! (it's against incrementality in this setup)



#========== LOOP
# forbidden single letters: p,k,c,r (already used for FF config)
runtime_zero = time.time()
performance_summary = {} # for a final summary

for i in range(0, len(breeder.words), test_interval_in_words):
    t_thisrun = time.time() # ends before logging
    run = int(i/test_interval_in_words)
    
    breeder.flyfile = fly_location+"breeder_run_"+run+".cfg" # for logging #TODO set the correct file name
    space_file = space_location+"breeder_run"+run+".dh" #TODO set the correct file name
    results_file = results_location+"breeder_run"+run+".txt" #TODO set the correct file name
    vip_words_file = vip_words_location+"breeder_run"+run".txt" #TODO set the correct file name


    #========== COUNT AND EXPAND
    try: # take a slice from the corpus
        count_these = breeder.words[i:i+test_interval_in_words]
    except IndexError as e:
        count_these = breeder.words[i:len(breeder.words)]
        print("This batch of words only has",len(count_these),"items")

    t_count = breeder.count_cooccurrences(words=count_these, window=window)[1] # [1] selects the stats tuple
    is_x, x_diff = breeder.check_overlap(checklist_filepath=testfile, wordlist=count_these)

    t_logmat = breeder.log_matrix()[1] # no update of the filepath needed
    t_logfly = breeder.log_fly()[1] # flyfile is updated above

    
    #========== FLY AND EVALUATE    
    unhashed_space = utils.readDM(breeder.outspace)
    hashed_space, t_flight = breeder.fruitfly.fly(unhashed_space, flattening=flat)

    spb,spa,sp_diff,testsize = 0,0,0,0 # be sure to start with
    spb,testsize = MEN.compute_men_spearman(unhashed_space, testfile)
    spa,testsize = MEN.compute_men_spearman(hashed_space, testfile)
    sp_diff = sp_after-sp_before
    
    connectednesses = [len(cons) for cons in breeder.fruitfly.pn_to_kc]
    avg_PN_con = round(sum(connectednesses)/breeder.fruitfly.pn_size,6)
    var_PN_con = round(np.var(connectednesses, ddof=1),6)
    std_PN_con = round(np.std(connectednesses, ddof=1),6)

    t_thisrun = time.time()-t_thisrun

    if allow_disconnection:
        pass #breeder.fruitfly.selective_disconnect(<in the notes>) #TODO implement!

    #========== LOGGING (SPACE, FLY, PERFORMANCE)
    utils.writeDH(hashed_space, space_file) # space_file is updated above

    # log most important words for each word
    with open(vip_words_file, "w") as f:
        for w in hashed_space:
            vip_words = breeder.fruitfly.important_words_for(\
                        w, breeder.fruitfly.i_to_cols, n=number_of_vip_words)
            f.write("{0} --> {1}".format(w, ", ".join(vip_words)))

    # log the whole fruitfly
    breeder.fruitfly.log_params(filename=breeder.flyfile, timestamp=False)
    
    # log the results
    with open(results_file, "w") as f:
        result_string = "RESULTS FOR RUN "+str(run)+":\n\n"+\
                        "SP_DIFF:   {0}\nSP_AFTER:  {1}\nSP_BEFORE: {2}\nTESTSIZE: {3}\n\n"\
                        .format(sp_diff, spa, spb, testsize)
        fly_specs = breeder.fruitfly.get_specs()
        t = (fly_specs["pn_size"], fly_specs["kc_size"], fly_specs["proj_size"], fly_specs["hash_percent"],flat)
        fruitfly_string = "FLY CONFIG: ({0}, {1}, {2}, {3}, {4})\n"\
                          .format(t[0],t[1],t[2],t[3],t[4])+\
                          "AVG PN CON.:  {0}\nVAR PN CON.: {1}\nSTD PN CON.: {2}\n\n"\
                          .format(avg_PN_con, var_PN_con, std_PN_con)
        time_string = "TIMES TAKEN:\nCOOC COUNT: {0}\nFLYING: {1}\nWRITE MATRIX: {2}\nLOG FLY: {3}\nTHIS RUN: {4}\n\n"\
                      .format(t_count[3], t_flight[3], t_logmat[3], t_logfly[3], t_thisrun)
        file_string = "RELATED FILES:\nFRUITFLY: {0}\nSPACE: {1}\nVIP WORDS: {2}\n"\
                      .format(breeder.flyfile, space_file, vip_words_file)

        f.write(result_string \
                +fruitfly_string \
                +time_string \
                +file_string)

    # keep internal log
    performance_summary[run] = (fly_specs["pn_size"], testsize, spb, spa, sp_diff, t_thisrun)



# make a summary file
with open(results_location+"summary.txt","w") as f:
    f.write("run, PN size, testset, spb, spa, sp_diff, time taken:\n\n")
    for k,t in performance_summary.items():
        f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n\
            ".format(t[0],t[1],t[2],t[3],t[4],t[5],t[6]))



"""
Attributes of Incrementor

verbose,
infile,
is_tokenize,
is_linewise,
required_voc,         # checklist
outspace,             # logging destination for matrix
outcols,              # logging destination for matrix word list
is_incremental,
max_dims,
is_new_fly,
is_grow_fly,
flyfile,              # logging destination for FF
cooc,                 # cooccurrence matrix
words_to_i,           # word list (index)
fruitfly,             # fruitfly
words,                # corpus (as words)
freq                  # freqdist of the whole corpus
"""