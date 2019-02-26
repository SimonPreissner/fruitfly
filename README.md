# Implementing the fruit fly's similarity hashing

This is an implementation of the random indexing method described by Dasgupta et al (2017) in [A neural algorithm for a fundamental computing problem](http://science.sciencemag.org/content/358/6364/793/tab-figures-data).

The aim of this project is to create an architecture for distributional semantics that grows in parallel to the data that is available. The incrementality that we aim for will enable us to create distributional semantic spaces from growing text corpora at any stage, independently from their current size. 

### Description of the data

Your data/ directory contains several semantic spaces:

- one from the British National Corpus, containing lemmas expressed in 4000 dimensions. (BNC-MEN)
- one from a subset of Wikipedia, containing words expressed in 1000 dimensions. (wiki.all)
- one from a subset of UKWAC, containing words expressed in 1000 dimensions. (ukwac_1k_GS-checked)

The cells in the BNC and the Wikipedia space are normalised co-occurrence frequencies *without any additional weighting* (PMI, for instance); the cells in the UKWAC space are raw frequency counts. The UKWAC space contains only words that are in the test set (see below).

The directory also contains test pairs from the [MEN similarity dataset](https://staff.fnwi.uva.nl/e.bruni/MEN), both in lemmatised and natural forms.

Finally, it contains a file *generic_pod.csv*, which is a compilation of around 2400 distributional web page signatures, in [PeARS](http://pearsearch.org) format. The web pages span various topics: Harry Potter, Star Wars, the Black Panther film, the Black Panther social rights movement, search engines and various small topics involving architecture.

### Running the fruit fly code

##### projection.py
To create a Fruitfly, let it run over a space, and evaluate it, use projection.py
For detailed usage description, run it without parameters:
```
python3 projection.py
```
A standard run migh look like this (test on he BNC corpus with 8000 Kenyon cells (=KCs), 6 projections leading to each KC, and 5% hashing): 
```
python3 projection.py bnc 8000 6 5
```
Or for the Wikipedia space (4000 KCs, 4 projections, 10% hashing):
```
python3 projection.py wiki 4000 4 10
```
The program returns the Spearman correlation with the MEN similarity data, as calculated a) from the raw frequency space; and b) after running the fly's random projections, as well as the difference between the two.

##### Fruitfly.py
The Fruitfly class implements the Fruitfly algorithm and utilities for logging and inspecting any one fruitfly instance. 
 
`Fruitfly.from_scratch()` instantiates a new fruitfly,  
`Fruitfly.log_params()`  outputs a fruitfly to a text file, and with  
`Fruitfly.from_config()`, you can re-use the exact same net (including all projection connections) later on.  
`Fruitfly.fly()` implements the actual application of the algorithm (hashing a distributional space).  
`Fruitfly.extend()` implements the incremental feature of this project's variant of the Fruitfly algorithm.  


### Tuning parameters

First, get a sense for which parameters give best results on the MEN dataset, for both BNC and Wikipedia data. If you know how to code, you can do a random parameter search automatically. If not, just try different values manually and write down what you observe.


### Analysing the results

Compare results for the BNC and the Wikipedia data. You should see that results on the BNC are much better than on Wikipedia. Why is that?

To help you with the analysis, you can print a verbose version of the random projections with the -v flag. E.g.:

    python3 projection.py bnc 8000 6 1 -v

This will print out the projection neurons that are most responsible for the activation in the Kenyon layer.


### Turning the fly into a search engine

You can test the capability of the the fly's algorithm to return web pages that are similar to a given one (and crucially, dimensionality-reduced), by typing:

    python3 searchfly.py data/generic_pod.csv 2000 6 5 https://en.wikipedia.org/wiki/Yahoo!
