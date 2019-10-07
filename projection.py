""" Projection: Apply the Fruitfly Algorithm to a Distributional Space.
"""

import sys

import numpy as np
from docopt import docopt

import MEN
import utils
from Fruitfly import Fruitfly

#=============== PARAMETER INPUT
spacefiles = input("Space to be used (without file extension): ")
data = spacefiles+".dm"
column_labels = spacefiles+".cols"
MEN_annot = input("Testset to be used: ")

try:
    unhashed_space = utils.readDM(data) # returns dict of word : word_vector
    i_to_cols, cols_to_i = utils.readCols(column_labels) # returns both-ways dicts of the vocabulary (word:pos_in_dict); important for maintenances
except FileNotFoundError as e: #TODO make this try:except like in spacebreeder.
    print("Unable to find files for input space and/or vocabulary.\n\
           - correct file path?\n\
           - are the file extensions '.dm' and '.cols'?\n\
           - don't specify the file extension.")
    sys.exit()
#TODO make the input like in spacebreeder.py!

evaluate_mode = True if input("Only evaluate the space (without flying)? [y/n] ").lower()=="y" else False
flattening = input("Choose flattening function: ")
pn_size = len(cols_to_i) # length of word vector (= input dimension)
print ("Number of PNs:",pn_size)
kc_size = int(input("Number of KCs: "))
proj_size = int(input("Number of projections per KC: "))
hash_percent = int(input("Percentage of winners in the hash: "))

iterations = int(input("How many runs? "))
verbose = True if input("Verbose mode? [y/n] ").lower()=="y" else False



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

    fruitfly = Fruitfly.from_scratch(pn_size, kc_size, proj_size, hash_percent, flattening)

    space_hashed, space_dic, space_ind = fruitfly.fly(unhashed_space, cols_to_i) # a dict of word : binary_vector (= after "flying")

    #utils.writeDH(space_hashed, "testwrite.dh") #CLEANUP
    #loaded_hashes = utils.readDH("testwrite.dh") #CLEANUP
    #sparse_space = utils.sparsifyDH(loaded_hashes, 1000) # dims = kc_size #CLEANUP

    if verbose: 
        for w in space_hashed:
            words = fruitfly.important_words_for(space_hashed[w], space_ind, n=6)
            print("{0} IMPORTANT WORDS: {1}".format(w, words))

    #print("done.") #CLEANUP
    #sys.exit() #BREAKPOINT

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
