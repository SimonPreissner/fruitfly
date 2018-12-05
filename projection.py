import sys
import utils
import MEN
import numpy as np

'''Parameter input'''

if len(sys.argv) < 5 or sys.argv[1] not in ("bnc","wiki"):
    print("\nUSAGE: python3 projection.py bnc|wiki [num-kc] [size-proj] [percent-hash]\n\
    - num-kc: the number of Kenyon cells\n\
    - size-proj: how many projection neurons are used for each projection\n\
    - percent-hash: how much of the Kenyon layer to keep in the final hash.\n")
    sys.exit() 

if sys.argv[1] == "bnc":
    data = "data/BNC-MEN.dm"
    column_labels = "data/BNC-MEN.cols"
    MEN_annot = "data/MEN_dataset_lemma_form_full"
else:
    data = "data/wiki_all.dm"
    column_labels = "data/wiki_all.cols"
    MEN_annot = "data/MEN_dataset_natural_form_full"

english_space = utils.readDM(data) # returns dict of word : word_vector
i_to_cols, cols_to_i = utils.readCols(column_labels) # returns both-ways dicts of the vocabulary (word:pos_in_dict); important for maintenances

PN_size = len(english_space.popitem()[1]) # length of word vector (= input dimension)
KC_size = int(sys.argv[2])
proj_size = int(sys.argv[3]) # number of connections to any given KC
percent_hash = int(sys.argv[4])
print("SIZES PN LAYER:",PN_size,"KC LAYER:",KC_size)
print("SIZE OF PROJECTIONS:",proj_size)
print("SIZE OF FINAL HASH:",percent_hash,"%")

projection_layer = np.zeros(PN_size) # input layer
kenyon_layer = np.zeros(KC_size)
projection_functions = [] # will contain the connections from PN to KC 


'''Create random projections'''
print("Creating",KC_size,"random projections...")
projection_functions = {} # dict of kc_i:array_of_connections that contains the transitions from PNs to KCs
for cell in range(KC_size):
    activated_pns = np.random.randint(PN_size, size=proj_size)
    projection_functions[cell] = activated_pns

def show_projections(word,hashed_kenyon):
    important_words = {} # dict of word:number
    for i in range(len(hashed_kenyon)):
        if hashed_kenyon[i] == 1:
            activated_pns = projection_functions[i] # retrieve transitions of an activated KC
            #print(word,[i_to_cols[pn] for pn in activated_pns])
            for pn in activated_pns: # count which word helped how many times to lead to 'hashed_kenyon'
                w = i_to_cols[pn] # retrieve word from PN index
                if w in important_words:
                    important_words[w]+=1
                else:
                    important_words[w]=1
    print(word,"BEST PNS", sorted(important_words, key=important_words.get, reverse=True)[:proj_size]) # only print the most important words

def projection(projection_layer): # Doing the flattening here is possible, but not efficient.
    kenyon_layer = np.zeros(KC_size)
    for cell in range(KC_size):
        activated_pns = projection_functions[cell] # array of the connected cells
        for pn in activated_pns:
            kenyon_layer[cell]+=projection_layer[pn] # sum the activation values of the pn nodes in the kc
            #kenyon_layer[cell]+=np.log(1+3000*projection_layer[pn]) # direct flattening with intuitively chosen factor
    return kenyon_layer

def flatten_log(frequency_vector): 
    factor = zipf_approximation(len(frequency_vector)) # in order to reverse-engineer to absolute frequencies
    for i, freq in enumerate(frequency_vector):
        frequency_vector[i] = np.log(1+factor*freq) # add 1 to make sure that no value is below 1
    #print(frequency_vector[:10], "\n===s")
    return frequency_vector

def zipf_approximation(voc_size): # approximates the number of words of a text using Zipf's Law
    wordcount = 0
    rank = 1
    while round(voc_size*(1.0/rank)) >= 1:
        wordcount+=voc_size*(1.0/rank) # adding up expected occurrances of a word according to its rank
        rank+=1
    #print("approximated word count for", voc_size, "words:", wordcount)
    return wordcount

def hash_kenyon(kenyon_layer):
    #print(kenyon_layer[:100])
    kenyon_activations = np.zeros(KC_size)
    top = int(percent_hash * KC_size / 100) # number of winners (highest activation)
    activated_kcs = np.argpartition(kenyon_layer, -top)[-top:]
    for cell in activated_kcs:
        kenyon_activations[cell] = 1 # assign 1 to the winners
    return kenyon_activations

def hash_input(word):
    #projection_layer = flatten_log(english_space[word]) # get full word vector of 'word' and flatten it already
    projection_layer = flatten_(english_space[word]) # get full word vector of 'word' and flatten it already
    #projection_layer = flatten(english_space[word]) # get full word vector of 'word' and flatten it already
    kenyon_layer = projection(projection_layer)
    hashed_kenyon = hash_kenyon(kenyon_layer) # same dimensionality as 'kenyon_layer'
    if len(sys.argv) == 6 and sys.argv[5] == "-v":
        show_projections(word,hashed_kenyon)
    return hashed_kenyon # this is the pattern obtained from the FFA



english_space_hashed = {} # a dict of word : binary_vector (= after "flying")
for w in english_space: # iterate through dictionary 
    hw = hash_input(w) # has the same dimension as the KC layer, but is binary
    english_space_hashed[w]=hw

#print(utils.neighbours(english_space,sys.argv[1],10))
#print(utils.neighbours(english_space_hashed,sys.argv[1],10))

sp,count = MEN.compute_men_spearman(english_space,MEN_annot)
print ("SPEARMAN BEFORE FLYING:",sp, "(calculated over",count,"items.)")
sp,count = MEN.compute_men_spearman(english_space_hashed,MEN_annot)
print ("SPEARMAN AFTER FLYING:",sp, "(calculated over",count,"items.)")
