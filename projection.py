import sys
import utils
import Fruitfly
from Fruitfly import Fruitfly
import MEN
import numpy as np


"""
This script reads command line options and feeds data to a Fruitfly object. 
"""


#=============== PARAMETER INPUT

if len(sys.argv) < 5 or sys.argv[1] not in ("bnc","wiki","rawiki", "w2v"):
    print("\nUSAGE: python3 projection.py bnc|wiki|rawiki|w2v [num-kc] [size-proj] [percent-hash]\n\
    - num-kc: the number of Kenyon cells\n\
    - size-proj: how many projection neurons are used for each projection\n\
    - percent-hash: how much of the Kenyon layer to keep in the final hash.\n")
    sys.exit() 

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
    data = "/mnt/8tera/corpora/ukwac/ukwac_100m/ukwac_100m_w2v.txt"
    column_labels= "/mnt/8tera/corpora/ukwac/ukwac_100m/ukwac_100m.w2v.vocab"
    MEN_annot = "data/MEN_dataset_natural_form_full"


unhashed_space = utils.readDM(data) # returns dict of word : word_vector
i_to_cols, cols_to_i = utils.readCols(column_labels) # returns both-ways dicts of the vocabulary (word:pos_in_dict); important for maintenances



pn_size = len(cols_to_i) # length of word vector (= input dimension)
kc_size = int(sys.argv[2])
proj_size = int(sys.argv[3]) # number of connections to any given KC
hash_percent = int(sys.argv[4])
#print("SIZES PN LAYER:",pn_size,"KC LAYER:",kc_size)
#print("SIZE OF PROJECTIONS:",proj_size)
#print("SIZE OF FINAL HASH:",hash_percent,"%")

#=============== INITIATING AND OPERATING FRUITFLY


fruitfly = Fruitfly.from_scratch(pn_size, kc_size, proj_size, hash_percent)
#fruitfly = Fruitfly.from_config("ff-params_flattening_show-down.txt") # default parameters: filename="ff_config.txt"
#print(Flying with this configuration:", fruitfly.show_off())
space_hashed = fruitfly.fly(unhashed_space, "log") # a dict of word : binary_vector (= after "flying")
if len(sys.argv) == 6 and sys.argv[5] == "-v": 
    for w in space_hashed:
        fruitfly.show_projections(w, space_hashed[w], i_to_cols)


#=============== EVALUATION SECTION

#print(utils.neighbours(unhashed_space,sys.argv[1],10))
#print(utils.neighbours(space_hashed,sys.argv[1],10))

spb,count = MEN.compute_men_spearman(unhashed_space, MEN_annot)
print("Spearman before flying:",round(spb, 4), "(calculated over",count,"items.)")

spa,count = MEN.compute_men_spearman(space_hashed, MEN_annot)
print("Spearman after flying: ",round(spa,4), "(calculated over",count,"items.)")

print("difference:",round(spa-spb, 4))
#differences += spa-spb #CLEANUP



"""
for i in range(10): #CLEANUP
    print ("run number",i+1) #CLEANUP
    spb,count = MEN.compute_men_spearman(unhashed_space, MEN_annot)
    print("Run number",i+1,"; performance:",round(spb, 4), "(calculated over",count,"items.)")

"""