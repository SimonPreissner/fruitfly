"""Projection: Apply the Fruitfly Algorithm to a Distributional Space.

Usage:
  projcetion.py <space> <testset> [-f <flattening>] [-k <kc_size>] [-p <proj_size>] [-h <hash_percent>] [-v]
  projcetion.py --eval-only <space> <testset> 
  projection.py --help

Options:
  --help             Show this screen.
  -f=<flattening>    Flattening function [default: log]
  -k=<kc_size>       Number of Kenyon cells [default: 4000]
  -p=<proj_size>     Number of projections to each KC [default: 6]
  -h=<hash_percent>  Percentage of KCs for hashing [default: 5]
  -v --verbose       Output most important dimensions per word.
  -e --eval-only     Only evaluate; no fruitfly involved.

OBACHT!
  # Use the file names for <space> and <testset> WITHOUT file extension!
  
"""

import sys
from docopt import docopt
#=============== PARAMETER INPUT
if __name__ == '__main__':
    arguments = docopt(__doc__)

import utils
import Fruitfly
from Fruitfly import Fruitfly
import MEN
import numpy as np

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
hash_percent = int(arguments["-h"])

verbose = arguments["--verbose"]

#=============== FOR PURE EVALUATION OF UNHASHED SPACES

if evaluate_mode:
    spb,count = MEN.compute_men_spearman(unhashed_space, MEN_annot)
    print("Performance:",round(spb, 4), "(calculated over",count,"items.)")
    sys.exit()

#=============== INITIATING AND OPERATING FRUITFLY

fruitfly = Fruitfly.from_scratch(pn_size, kc_size, proj_size, hash_percent)

space_hashed = fruitfly.fly(unhashed_space, flattening) # a dict of word : binary_vector (= after "flying")
if verbose: 
    for w in space_hashed:
        fruitfly.show_projections(w, space_hashed[w], i_to_cols)

#=============== EVALUATION SECTION

spb,count = MEN.compute_men_spearman(unhashed_space, MEN_annot)
print("Spearman before flying:",round(spb, 4), "(calculated over",count,"items.)")

spa,count = MEN.compute_men_spearman(space_hashed, MEN_annot)
print("Spearman after flying: ",round(spa,4), "(calculated over",count,"items.)")

print("difference:",round(spa-spb, 4))


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
