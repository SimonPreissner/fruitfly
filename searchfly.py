




#THIS CODE IS DEPRECATED

import sys
import utils
import random
import numpy as np

'''Parameter input'''

if len(sys.argv) < 6:
    print("\nUSAGE: python3 searchfly.py [pod-path] [num-kc] [size-proj] [percent-hash] [url]\n\
    - pod-path: the location of the pod to process\n\
    - num-kc: the number of Kenyon cells\n\
    - size-proj: how many projection neurons are used for each projection\n\
    - percent-hash: how much of the Kenyon layer to keep in the final hash.\n")
    sys.exit() 


pod_space = utils.parse_pod(sys.argv[1])

pn_size = len(pod_space.popitem()[1])
kc_size = int(sys.argv[2])
proj_size = int(sys.argv[3])
hash_percent = int(sys.argv[4])
print("SIZES PN LAYER:",pn_size,"KC LAYER:",kc_size)
print("SIZE OF PROJECTIONS:",proj_size)
print("SIZE OF FINAL HASH:",hash_percent,"%")

projection_layer = np.zeros(pn_size)
kenyon_layer = np.zeros(kc_size)
projection_functions = []


'''Create random projections'''
print("Creating",kc_size,"random projections...")
projection_functions = {}
for cell in range(kc_size):
    activated_pns = np.random.randint(pn_size, size=proj_size)
    projection_functions[cell] = activated_pns

def projection(projection_layer):
    kenyon_layer = np.zeros(kc_size)
    for cell in range(kc_size):
        activated_pns = projection_functions[cell]
        for pn in activated_pns:
            kenyon_layer[cell]+=projection_layer[pn]
    return kenyon_layer

def hash_kenyon(kenyon_layer):
    #print(kenyon_layer[:100])
    kenyon_activations = np.zeros(kc_size)
    top = int(hash_percent * kc_size / 100)
    activated_kcs = np.argpartition(kenyon_layer, -top)[-top:]
    for cell in activated_kcs:
        kenyon_activations[cell] = 1
    return kenyon_activations

def hash_input(word):
    projection_layer = pod_space[word]
    kenyon_layer = projection(projection_layer)
    hashed_kenyon = hash_kenyon(kenyon_layer)
    return hashed_kenyon

pod_space_hashed = {}

for url in pod_space:
    hw = hash_input(url)
    pod_space_hashed[url]=hw


test_url = random.choice(list(pod_space.keys()))
test_url = sys.argv[5]
print("TESTING ON:",test_url)
print(utils.neighbours(pod_space,test_url,10),'\n')
print(utils.neighbours(pod_space_hashed,test_url,10))

