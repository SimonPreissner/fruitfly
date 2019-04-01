import numpy as np
from math import sqrt
from matplotlib import cm
import pandas as pd
from sklearn.decomposition import PCA
import time

def readDM(dm_file): # read word vectors
    dm_dict = {}
    version = ""
    with open(dm_file, "r") as f:
        dmlines=f.readlines()

    #Make dictionary with key=word, value=vector
    for l in dmlines:
        items=l.rstrip().split()
        row=items[0] # word
        vec=[float(i) for i in items[1:]] # vector values
        vec=np.array(vec)
        dm_dict[row]=vec
    return dm_dict

def readCols(cols_file):
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

def parse_pod(pod):
    pod_dict = {}
    with open(pod, "r") as f:
        for l in f:
            if l[0] != '#':
                try:
                    fields = l.rstrip('\n').split(',')
                    url = fields[1]
                    vector = np.array([float(i) for i in fields[4].split()])
                    pod_dict[url] = vector
                except:
                    pass
    return pod_dict


def cosine_similarity(v1, v2):
    if len(v1) != len(v2):
        return 0.0
    num = np.dot(v1, v2)
    den_a = np.dot(v1, v1)
    den_b = np.dot(v2, v2)
    return num / (sqrt(den_a) * sqrt(den_b))


def neighbours(dm_dict,word,n):
    cosines={}
    c=0
    vec = dm_dict[word]
    for k,v in dm_dict.items():
        cos = cosine_similarity(vec, v)
        cosines[k]=cos
        c+=1
    c=0
    neighbours = []
    for t in sorted(cosines, key=cosines.get, reverse=True):
        if c<n:
             #print(t,cosines[t])
             neighbours.append(t)
             c+=1
        else:
            break
    return neighbours


def make_figure(m_2d, labels):
    cmap = cm.get_cmap('nipy_spectral')

    existing_m_2d = pd.DataFrame(m_2d)
    existing_m_2d.index = labels
    existing_m_2d.columns = ['PC1','PC2']
    existing_m_2d.head()

    ax = existing_m_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(30,18), c=range(len(existing_m_2d)), colormap=cmap, linewidth=0, legend=False)
    ax.set_xlabel("A dimension of meaning")
    ax.set_ylabel("Another dimension of meaning")

    for i, word in enumerate(existing_m_2d.index):
        #print(word+" "+str(existing_m_2d.iloc[i].PC2)+" "+str(existing_m_2d.iloc[i].PC1))
        ax.annotate(
            word,
            (existing_m_2d.iloc[i].PC2, existing_m_2d.iloc[i].PC1), color='black', size='large', textcoords='offset points')

    fig = ax.get_figure()
    return fig


def run_PCA(dm_dict, words, savefile):
    m = []
    labels = []
    for w in words:
        labels.append(w)
        m.append(dm_dict[w])
    pca = PCA(n_components=2)
    pca.fit(m)
    m_2d = pca.transform(m)
    png = make_figure(m_2d,labels)
    cax = png.get_axes()[1]
    cax.set_visible(False)
    png.savefig(savefile)
                                          

def timeit(method):
    """
    wrapper for timing. Usage leads to returning of an additional value:
    a tuple containing timing information.
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        stats = (method.__name__, args, kw, te-ts)
        return result+((stats,))
    return timed # returning a function object makes it a wrapper


