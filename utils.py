import numpy as np
from math import sqrt

def readDM(dm_file): # read word vectors
    """
    read word vectors from a .dm file, where the first word of a line is 
    the actual word and all other elements, separated by space of tab, 
    are the word's vector. 
    """
    dm_dict = {}
    version = ""
    with open(dm_file, "r") as f:
        dmlines=f.readlines()

    #Make dictionary with key=word, value=vector
    for l in dmlines:
        items=l.rstrip().split() # splits at spaces and at tabs
        row=items[0] # word
        vec=[float(i) for i in items[1:]] # vector values
        vec=np.array(vec)
        dm_dict[row]=vec
    return dm_dict

def readCols(cols_file):
    """
    read a .cols file and return two dictionaries: word:index and 
    index:word. 
    """
    i_to_cols = {}
    cols_to_i = {}
    c = 0
    with open(cols_file,'r') as f:
        for l in f:
            l = l.rstrip('\n')
            i_to_cols[c] = l
            cols_to_i[l] = c
            c+=1
    return i_to_cols, cols_to_i # dimension_number : word and word : dimension_number

def readDH(dh_file):
    """
    read a hashed space file (file extension: .dh)
    hashed spaces are stored like .dm files, but values are all integers.
    """
    dh_dict = readDM(dh_file)
    for w,h in dh_dict.items():
        dh_dict[w] = [int(v) for v in h]
    return dh_dict

def writeDH(sparse_space, dh_file):
    """
    write a hashed space to a file with the extension .hs;
    """
    dense_space = {w:np.nonzero(h)[0] for w,h in sparse_space.items()}

    with open(dh_file, "w") as f:
        for w,h in dense_space.items():
            vectorstring = " ".join([str(v) for v in h])
            f.write("{0} {1}\n".format(w, vectorstring))

def sparsifyDH(dense_space, dims):
    """
    return the sparse representations of dense hashes. This means that 
    each number in the dense hash implies 1 at the sparse hash's index 
    of that number.
    The size of the retured sparse hashes has to be provided.
    """
    sparse_space = {}
    for w,h in dense_space.items():
        sv = np.zeros(shape=(dims,))
        for i in h:
            sv[i] = 1
        sparse_space[w] = sv
    return sparse_space

def cosine_similarity(v1, v2):
    if len(v1) != len(v2):
        return 0.0
    num = np.dot(v1, v2)
    den_a = np.dot(v1, v1)
    den_b = np.dot(v2, v2)
    return num / (sqrt(den_a) * sqrt(den_b))

def neighbours(space,word,n):
    """
    find the n closest neighbours (by cosine) to a word in a space.
    """
    cosines={}
    vec = space[word]
    for k,v in space.items():
        cos = cosine_similarity(vec, v)
        cosines[k]=cos

    neighbours = sorted(cosines, key=cosines.get, reverse=True)[:n]
    return neighbours



"""
"""

