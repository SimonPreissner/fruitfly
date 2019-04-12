"""Projection: Apply the Fruitfly Algorithm to a Distributional Space.

Usage:
  projcetion.py <space> <testset> [--eval-only | [options]] [-v] [-i <runs>]

Options:
  -h --help          Show this screen.
  -f=<flattening>    Flattening function [default: log]
  -k=<kc_size>       Number of Kenyon cells [default: 4000]
  -p=<proj_size>     Number of projections to each KC [default: 6]
  -r=<hash_percent>  Percentage of KCs for hashing/reduction [default: 5]
  -i=<runs>          Number of runs with the same parameters [default: 1]
  -v --verbose       Output most important dimensions per word.
  -e --eval-only     Only evaluate; no fruitfly involved.

OBACHT!
  # Use the file names for <space> and <testset> WITHOUT file extension!
  
"""

import sys
from docopt import docopt
import utils
import Fruitfly
from Fruitfly import Fruitfly
import MEN
import numpy as np


#=============== PARAMETER INPUT
if __name__ == '__main__':
    arguments = docopt(__doc__)

data = arguments["<space>"]+".dm"
column_labels = arguments["<space>"]+".cols"
MEN_annot = arguments["<testset>"]

try:
    unhashed_space = utils.readDM(data) # returns dict of word : word_vector
    i_to_cols, cols_to_i = utils.readCols(column_labels) # returns both-ways dicts of the vocabulary (word:pos_in_dict); important for maintenances
except FileNotFoundError as e:
    print("Unable to find files for input space and/or vocabulary.\n\
           - correct file path?\n\
           - are the file extensions '.dm' and '.cols'?\n\
           - don't specify the file extension.")
    sys.exit()

evaluate_mode = arguments["--eval-only"]
flattening = arguments["-f"]
pn_size = len(cols_to_i) # length of word vector (= input dimension)
kc_size = int(arguments["-k"])
proj_size = int(arguments["-p"])
hash_percent = int(arguments["-r"])

iterations = int(arguments["-i"])
verbose = arguments["--verbose"]



#=============== EXECUTIVE CODE
all_spb = []
all_spa = []
all_spd = []
for i in range(iterations):
    if iterations > 1:
        print("\n#=== NEW RUN:",i+1,"===#")
    #=============== FOR PURE EVALUATION OF UNHASHED SPACES

    if evaluate_mode:
        spb,count = MEN.compute_men_spearman(unhashed_space, MEN_annot)
        print("Performance:",round(spb, 4), "(calculated over",count,"items.)")
        sys.exit()

    #=============== INITIATING AND OPERATING FRUITFLY

    fruitfly = Fruitfly.from_scratch(pn_size, kc_size, proj_size, hash_percent)

    space_hashed, t_flight = fruitfly.fly(unhashed_space, flattening) # a dict of word : binary_vector (= after "flying")

    #utils.writeDH(space_hashed, "testwrite.dh")
    #loaded_hashes = utils.readDH("testwrite.dh")
    #sparse_space = utils.sparsifyDH(loaded_hashes, 1000) # dims = kc_size

    if verbose: 
        for w in space_hashed:
            words = fruitfly.important_words_for(space_hashed[w], i_to_cols, n=6)
            print("{0} IMPORTANT WORDS: {1}".format(w, words))

    print("done.") #CLEANUP
    sys.exit() #BREAKPOINT

    #=============== EVALUATION SECTION

    spb,count = MEN.compute_men_spearman(unhashed_space, MEN_annot)
    print("Spearman before flying:",round(spb, 4), "(calculated over",count,"items.)")

    spa,count = MEN.compute_men_spearman(space_hashed, MEN_annot)
    print("Spearman after flying: ",round(spa,4), "(calculated over",count,"items.)")

    print("difference:",round(spa-spb, 4))

    all_spb.append(spb)
    all_spa.append(spa)
    all_spd.append(spa-spb)

if iterations > 1:
    best = sorted(all_spd, reverse=True)

    print("\nFinished all",iterations,"runs. Summary:")
    print("best and worst runs:",[round(e, 4) for e in best[:3]].extend([round(e, 4) for e in best[:-3]]))
    print("mean Sp. before:    ",round(np.average(all_spb), 4))
    print("mean Sp. after:     ",round(np.average(all_spa), 4))
    print("mean Sp. difference:",round(np.average(all_spd), 4))
    print("var of Sp. before:    ",round(np.var(all_spb, ddof=1),8))
    print("var of Sp. after:     ",round(np.var(all_spa, ddof=1),8))
    print("var of Sp. difference:",round(np.var(all_spd, ddof=1),8))
    print("std of Sp. before:     ",round(np.std(all_spb, ddof=1), 8))
    print("std of Sp. after:      ",round(np.std(all_spa, ddof=1), 8))
    print("std of Sp. difference: ",round(np.std(all_spd, ddof=1), 8))



#========== PARAMETER VALUES
"""
data = 
    "data/BNC-MEN.dm"
    "data/wiki_all.dm"
    "data/wiki_abs-freq.dm"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_100m_w2v_400.txt"
    "/home/simon.preissner/FFP/fruitfly/incrementality_sandbox/data/sandbox_w2v_400.txt"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_1k_MEN-checked.dm"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_1k_GS-checked.dm"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_5k_MEN-checked.dm"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_5k_GS-checked.dm"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_10k_MEN-checked.dm"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_10k_GS-checked.dm"
    "/home/simon.preissner/FFP/fruitfly/incrementality_sandbox/data/sandbox_1000_dim.dm"
    "/home/simon.preissner/FFP/fruitfly/incrementality_sandbox/data/sandbox_5000_dim.dm"
    
column_labels = 
    "data/BNC-MEN.cols"
    "data/wiki_all.cols"
    "data/wiki_abs-freq.cols"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac.w2v.400.vocab"
    "/home/simon.preissner/FFP/fruitfly/incrementality_sandbox/data/sandbox_w2v_400.vocab"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_1k_MEN-checked.cols"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_1k_GS-checked.cols"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_5k_MEN-checked.cols"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_5k_GS-checked.cols"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_10k_MEN-checked.cols"
    "/home/simon.preissner/FFP/ukwac_100m/ukwac_10k_GS-checked.cols"
    "/home/simon.preissner/FFP/fruitfly/incrementality_sandbox/data/sandbox_1000_dim.cols"
    "/home/simon.preissner/FFP/fruitfly/incrementality_sandbox/data/sandbox_5000_dim.cols"

MEN_annot = 
    "data/MEN_dataset_natural_form_full"
    "data/MEN_dataset_lemma_form_full"
    "incrementality_sandbox/data/sandbox_MEN_pairs"



"""
