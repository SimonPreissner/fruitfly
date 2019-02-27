# Implementing the fruit fly's similarity hashing

This is an implementation of the random indexing method described by Dasgupta et al (2017) in [A neural algorithm for a fundamental computing problem](http://science.sciencemag.org/content/358/6364/793/tab-figures-data).

The aim of this project is to create an architecture for distributional semantics that grows in parallel to the data that is available. The incrementality that we aim for will enable us to create distributional semantic spaces from growing text corpora at any stage, independently from their current size. 

### Description of the data

The data/ directory contains several semantic spaces:

- one from the British National Corpus, containing lemmas expressed in 4000 dimensions. (BNC-MEN)
- one from a subset of Wikipedia, containing words expressed in 1000 dimensions. (wiki.all)
- one from a subset of UKWAC, containing words expressed in 1000 dimensions. (ukwac_1k_GS-checked)

The cells in the BNC and the Wikipedia space are normalised co-occurrence frequencies *without any additional weighting* (PMI, for instance); the cells in the UKWAC space are raw frequency counts. The UKWAC space contains only words that are in the test set (see below).

The directory also contains test pairs from the [MEN similarity dataset](https://staff.fnwi.uva.nl/e.bruni/MEN), both in lemmatised and natural forms.

Finally, it contains a file *generic_pod.csv*, which is a compilation of around 2400 distributional web page signatures, in [PeARS](http://pearsearch.org) format. The web pages span various topics: Harry Potter, Star Wars, the Black Panther film, the Black Panther social rights movement, search engines and various small topics involving architecture.

### Running the fruit fly code with projection.py
To create a Fruitfly, let it run over a space, and evaluate it, use this script.
For detailed usage description, run it without parameters:
```
python3 projection.py
```
A standard run might look like this (test on he BNC corpus with 8000 Kenyon cells (=KCs), 6 projections leading to each KC, and 5% hashing): 
```
python3 projection.py bnc 8000 6 5
```
Or for the Wikipedia space (4000 KCs, 4 projections, 10% hashing):
```
python3 projection.py wiki 4000 4 10
```
The program returns the Spearman correlation with the MEN similarity data, as calculated a) from the raw frequency space; and b) after running the fly's random projections, as well as the difference between the two.

If you want to set a baseline and evaluate a semantic space without "flying", use the `eval-only` option (you will still have to set the fruitfly's parameters):
```
python3 projection.py 1k 1 1 1 eval-only
```
Finally, with the -v flag ("verbose"), the script prints out the projection neurons that are most responsible for the activation in the Kenyon layer.
```
    python3 projection.py bnc 8000 6 5 -v
```

### The algorithm itself: Fruitfly.py
The Fruitfly class implements the Fruitfly algorithm and utilities for logging and inspecting any one fruitfly instance. 
 
`Fruitfly.from_scratch()` instantiates a new fruitfly,  
`Fruitfly.log_params()`  outputs a fruitfly to a text file, and with  
`Fruitfly.from_config()`, you can re-use the exact same net (including all projection connections) later on.  
`Fruitfly.fly()` implements the actual application of the algorithm (hashing a distributional space).  
`Fruitfly.extend()` implements the incremental feature of this project's variant of the Fruitfly algorithm.  


### Tuning hyperparameters with hyperopt.py
To perform a grid search on ranges of the fruitfly's hyperparameters, use hyperopt.py; For detailed usage description, run it without parameters:
```
python3 hyperopt.py
```
You will need to specify a corpus (the same options as in projection.py are available). Hyperopt.py will create files with the test result, one for each run, and a final summary. It's recommended to specify the directory for these files.
Other parameters you can provide:
- flattening options (as simple arguments)
- factor of expansion from PN layer to KC layer
- number of connecitons going to each KC
- percentage of KCs to be selected as "winners"

All parameters have default ranges, so if you run 
```
python3 hyperopt.py bnc -logto data/potato log log2 -kc 2 8 2 -hash 5 15 3
```
the script will perform a grid search for parameters on the BNC space with 2 parameters for flattening, 4 for the KC-factor, 5 for the hashing, and the default ranges for projections (again, to see the default ranges, run the script without parameters). You will find a new directory called `potato` with logs of all runs.  

There is also an option to run in verbose mode, by adding the flag `-v`  

Be careful with large numbers for the ranges of -kc and -proj! These parameters can quickly inflate the performance time.


### Compiling semantic spaces with countwords.py
Part of the project is to be able to integrate new data into an already-processed corpus. For now, `countwords.py` takes a text coprus and compiles a co-occurrence matrix of words. Run it without parameters for details.  
The most important parameters are 
- `-dim`, which limits the dimensionality of the resulting space
- `-window`, which specifies the range (before and after a word) that is used for co-occurrence counting 
- `-check`, which checks the word overlap of a specified file with the resulting space

For example, `ukwac_1k_GS-checked.dm` (and `.cols`) was compiled from `ukwac_100m.txt` (which is not available in this repository) with the following command:
```
python3 countwords.py ukwac_100m.txt ukwac_1k_GS-checked -dim 1000 -window 5 -check data/MEN_dataset_natural_form_full
```

Countwords.py is intended to implement incremental development of a Fruitfly object; that is, whenever a new word is observed, the Fruitfly object's parts (PN layer, KC layer, and projection connections) adapt to that new word.
