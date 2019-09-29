"""
This is the full pipeline of the incremental Fruitfly.
It reads a corpus, grows a Fruitfly alongside,
then applies the FFA to the counts,
optionally runs Word2Vec on the same corpus,
evaluates the results, and logs them.
"""

import os
import time

import math
import numpy as np
from tqdm import tqdm

import MEN
import utils
from Fruitfly import Fruitfly
from Incrementor import Incrementor

#TODO only make those variables global that need to be global!
global t_thisrun, verbose


#TODO are the ones below really needed globally?
#global errorlog, corpus_dir, w2v_exe_file, testset_file, overlap_file,\
#       pipedir, results_summary_file, results_location, w2v_results_file,\
#       matrix_filename, fly_location, space_location, vip_words_location,\
#       w2v_corpus_file, w2v_space_location, number_of_vip_words, test_interval_in_words,\
#       pns, kcs, con, red, flat, max_pns, tokenize, postag_simple, window

#========== Initial parameter input and setup

def param_input_files():
    #TODO reduce the global variables as much as possible
    global corpus_dir, w2v_exe_file, testset_file, overlap_file, results_summary_file, results_location, w2v_results_file, matrix_filename, fly_location, space_location, vip_words_location, w2v_corpus_file, w2v_space_location
    print("=== Resources and Output Directories ===")
    s = input("Path to text resources (default: data/chunks_wiki): ")
    corpus_dir = s if len(
        s) > 0 else "data/chunks_wiki"  # "/mnt/8tera/shareclic/fruitfly/ukwac_100m_tok-tagged.txt" # "../ukwac_100m/ukwac_100m.txt" # "test/pride_postagged.txt" #CLEANUP
    s = input("Path to Word2Vec code (optional): ")
    w2v_exe_file = s if len(s) > 0 else None  # "./../share/word2vec" #CLEANUP
    s = input(
        "Path to test set (default: data/MEN_dataset_natural_form_full): ")  # "data/MEN_dataset_lemma_form_full" # "./data/MEN_dataset_natural_form_full" # CLEANUP
    testset_file = s if len(s) > 0 else "data/MEN_dataset_natural_form_full"
    s = input("Path to a word list to be checked for overlap (optional): ")
    overlap_file = s if len(
        s) > 0 else None  # "./data/MEN_lemma_vocabulary" # "./data/MEN_natural_vocabulary" # CLEANUP
    s = input("Path to the resulting directories (default: results/pipe_00): ")
    pipedir = s if len(s) > 0 else "results/pipe_00"  # "/mnt/8tera/shareclic/fruitfly/pipe_postagged/" #CLEANUP
    results_summary_file = pipedir + "/" + "summary.tsv"
    results_location = pipedir + "/" + "ffa/results/stats/"
    w2v_results_file = pipedir + "/" + "w2v/results.txt" if w2v_exe_file is not None else None
    matrix_filename = pipedir + "/" + "ffa/cooc"
    fly_location = pipedir + "/" + "ffa/configs/"
    space_location = pipedir + "/" + "ffa/spaces/"
    vip_words_location = pipedir + "/" + "ffa/results/words/"
    w2v_corpus_file = pipedir + "/" + "w2v/resource.txt" if w2v_exe_file is not None else None
    w2v_space_location = pipedir + "/" + "w2v/" if w2v_exe_file is not None else None
    for f in [fly_location, space_location, results_location, vip_words_location, w2v_space_location]:
        if not os.path.isdir(f) and f is not None:
            os.makedirs(f, exist_ok=True)

def param_input_FFA():
    # TODO reduce the global variables as much as possible
    global pns, kcs, con, red, flat, max_pns
    print("=== Fruitfly Parameters ===")
    # Initial Fruitfly parameters (taken from the ukwac_100m gridsearch on 10k dims)
    pns, kcs, con, red, flat, max_pns = 50, 40000, 20, 8, "log", 10000
    d = True if input("use default parameters (", ", ".join([pns, kcs, con, red, flat, max_pns]),
                      ") [y/n]?").upper() == "Y" else False
    if d is False:
        pns = int(input("Initial number of PNs:"))
        kcs = int(input("Number of KCs:"))
        con = int(input("Connections per KC:"))
        red = int(input("WTA-percentage (whole number):"))
        flat = input("Flattening function (log/log2/log10):")
        max_pns = int(input("Maximum number of PNs:"))

def param_input_misc():
    # TODO reduce the global variables as much as possible
    global window, min_occs_in_text, max_dims, test_interval_in_words, test_filewise, number_of_vip_words, tokenize, postag_simple, verbose
    print("=== Other Parameters ===")
    s = input("Window size (to each side) for counting (default: 5):")
    window = int(s) if len(s) > 0 else 5  # one-directional window size for counting (+/- 5 words)
    s = input("Periodic deletion of infrequent words from the count (optional) -- minimum occurrences:")
    min_occs_in_text = int(s) if (
            len(s) > 0 and int(s) > 1) else None  # this will keep the count matrix' dimensions smaller
    s = input("Maximum vocabulary size for the count (skip this for true incrementality):")
    max_dims = int(s) if len(s) > 0 else None
    test_interval_in_words = 1000000
    s = input("Test interval in words (default:", test_interval_in_words, "-- file-wise testing with 'f'):")
    # TODO test with file-wise testing
    # TODO test with slice size larger than file size
    try:
        test_interval_in_words = int(s) if len(s) > 0 else None
        test_filewise = False
    except ValueError:
        test_interval_in_words = None
        test_filewise = True  # this is redundant, but more readable than handling test_interval_in_words==None
    s = input("Number of important words to be extracted (default: 50):")
    number_of_vip_words = int(s) if len(s) > 0 else 50
    tokenize = False if input("Tokenize the input text? [y/n]").upper() == "N" else True
    postag_simple = True if input(
        "Only count nouns/verbs/adjectives? [y/n]").upper() == "Y" else False  # TODO test this (True) # if true, it will only count nouns, verbs, and adjectives.
    # linewise = False # DON'T SET THIS TO True! It will break the loop. #CLEANUP
    verbose = False if input("Be verbose while running? [y/n]").upper() == "N" else True

def setup_loop_environment():
    global corpus_files, breeder #TODO resolve global variables
    """ make a list of file paths to be passed one by one to an Incrementor object """
    corpus_files = []
    if os.path.isfile(corpus_dir):
        corpus_files = [corpus_dir]
    else:
        for (dirpath, dirnames, filenames) in os.walk(corpus_dir):
            corpus_files.extend([dirpath + "/" + f for f in filenames])
    """ This step is taken because Incrementor can't do custom initialization """
    initial_fly_file = fly_location + "fly_run_0.cfg"
    first_fly = Fruitfly.from_scratch(flattening=flat, pn_size=pns, kc_size=kcs,
                                      proj_size=con, hash_percent=red, max_pn_size=max_pns)
    first_fly.log_params(filename=initial_fly_file, timestamp=False)
    """
    The constructor of Incrementor 
    - reads in text files from a directory (if specified)
    - computes a frequency distribution over the current text 
    - sets up or loads a Fruitfly object from an already-logged Fruitfly
    - sets up or loads a cooccurence matrix, with a limit if specified
    matrix_incremental=False because it would need to load an existing matrix; however, in this setup, 
    as only one Incrementor object is used, it can increment all the way from an initial matrix.
    linewise=False because it would destroy the whole loop. Therefore, it is
    also commented out in the parameter section above.
    """
    """ Initialization without corpus. File paths are passed to the Incrementor via the loop. """
    breeder = Incrementor(None, matrix_filename,
                          corpus_tokenize=tokenize, corpus_linewise=False, corpus_checkvoc=overlap_file,
                          matrix_incremental=False, matrix_maxdims=max_dims, contentwords_only=postag_simple,
                          fly_new=False, fly_grow=True, fly_file=initial_fly_file,
                          verbose=verbose)  # TODO rename breeder

#========== Core functionalities within the loop

def new_paths(run):
    """
        :param run: int
        :param verbose: bool
        :return: str, str, str, str
        """
    a = fly_location + "fly_run_" + str(run) + ".cfg"  # for logging
    b = space_location + "space_run_" + str(run) + ".dh"
    c = results_location + "stats_run_" + str(run) + ".txt"
    d = vip_words_location + "words_run_" + str(run) + ".txt"
    return a,b,c,d

def count_test_log(count_these):
    # TODO maybe improve on the verbose option within this whole method?
    global t_thisrun #, t, e # f,  #CLEANUP?

    # only counts cooccurrences of words within the freq_dist (which can be limited by matrix_maxdims)
    t_count = breeder.count_cooccurrences(words=count_these, window=window, timed=True)
    # delete words from the count matrix which are very infrequent
    nr_of_del_dims, t_cooc_del = breeder.reduce_count_size(min_occs_in_text, verbose=verbose, timed=True) # TODO make min_occs_in_text an attribute of Incrementor?
    breeder.log_matrix()

    fly_these, unhashed_space, words_to_i = prepare_flight()
    # space_dic and space_ind are the words_to_i and i_to_words of the cropped vectors (done in Fruitfly.fit_space())
    hashed_space, space_dic, space_ind, t_flight = breeder.fruitfly.fly(fly_these, words_to_i, timed=True)
    eval_and_log_FFA(count_these, hashed_space, space_ind, t_count, t_cooc_del, t_flight, unhashed_space)

    if w2v_exe_file is not None:
        w2v_min_count, w2v_space_file, w2v_vocab_file = prepare_w2v(count_these)
        t_train = execute_w2v(w2v_min_count, w2v_space_file, w2v_vocab_file)
        eval_and_log_w2v(t_train, w2v_space_file)

#========== FFA application functions

def prepare_flight():
    """ read in the count vectors etc. and choose which ones to fly """
    unhashed_space = utils.readDM(breeder.outspace)
    # print("length of unhashed_space:", len(unhashed_space))  # CLEANUP
    i_to_words, words_to_i = utils.readCols(breeder.outcols)
    # print("length of words_to_i obtained from unhashed_space: {0}".format(len(words_to_i)))  # CLEANUP
    # only select words that will be needed for evaluation:
    if overlap_file is None:
        fly_these = unhashed_space  # in this case, fly() is applied to the whole of unhashed_space
    else:
        words_for_flight = breeder.read_checklist(checklist_filepath=overlap_file,
                                                  with_pos_tags=breeder.postag_simple)
        fly_these = {w: unhashed_space[w] for w in words_for_flight if w in unhashed_space}
    # print("length of words_to_i:", len(breeder.words_to_i))  # CLEANUP
    # print("length of fly_these:", len(fly_these))  # CLEANUP
    return fly_these, unhashed_space, words_to_i

def eval_and_log_FFA(count_these, hashed_space, space_ind, t_count, t_cooc_del, t_flight, unhashed_space):
    global t_thisrun #, t #CLEANUP?
    spb, tsb = MEN.compute_men_spearman(unhashed_space, testset_file)
    spa, tsa = MEN.compute_men_spearman(hashed_space, testset_file)
    sp_diff = spa - spb
    connectednesses = [len(cons) for cons in breeder.fruitfly.pn_to_kc.values()]
    avg_PN_con = round(sum(connectednesses) / breeder.fruitfly.pn_size, 6)
    var_PN_con = round(float(np.var(connectednesses, ddof=1)), 6)
    std_PN_con = round(float(np.std(connectednesses, ddof=1)), 6)
    t_thisrun = time.time() - t_thisrun

    """ Log the hashed space, the fruitfly, and the evaluation """
    utils.writeDH(hashed_space, space_file)  # space_file is updated above
    # log most important words for each word
    if verbose: print("Logging most important words to", vip_words_file, "...")
    with open(vip_words_file, "w") as f:
        for w in tqdm(hashed_space):
            vip_words = breeder.fruitfly.important_words_for(hashed_space[w], space_ind, n=number_of_vip_words)
            vip_words_string = ", ".join(vip_words)
            f.write("{0} --> {1}\n".format(w, vip_words_string))
    # log the whole fruitfly
    breeder.log_fly()  # breeder.flyfile is updated at the very beginning of the run
    # log the results
    with open(results_file, "w") as f:
        result_string = "RESULTS FOR RUN " + str(run) + ":\n" + \
                        "\nsp_corr_diff:       " + str(sp_diff) + \
                        "\nsp_corr_after:      " + str(spa) + \
                        "\nsp_corr_before:     " + str(spb) + \
                        "\nitems_tested_after: " + str(tsa) + \
                        "\nitems_tested_before:" + str(tsb) + "\n"
        fc = breeder.fruitfly.get_specs()  # fly config
        t = [fc["flattening"], fc["pn_size"], fc["kc_size"], fc["proj_size"], fc["hash_percent"]]
        fruitfly_string = "\nFFA_CONFIG: (" + ", ".join(t) + ")" + \
                          "\navg_PN_connections: " + str(avg_PN_con) + \
                          "\nvar_PN_connections: " + str(var_PN_con) + \
                          "\nstd_PN_connections: " + str(std_PN_con) + "\n"
        time_string = "\nTIME TAKEN: " + \
                      "\ncounting: " + str(t_count) + \
                      "\nreducing: " + str(t_cooc_del) + \
                      "\nflying:   " + str(t_flight) + \
                      "\ntotal:    " + str(t_thisrun) + "\n"
        file_string = "\nRELATED FILES:" \
                      "\nfruitfly:        " + breeder.flyfile + \
                      "\nhashed_space:    " + space_file + \
                      "\nimportant_words: " + vip_words_file + "\n"
        f.write(result_string + fruitfly_string + time_string + file_string)
    # keep internal log
    performance_summary[run] = [len(count_these), fc["pn_size"], tsb, spb, spa, sp_diff, t_thisrun]

#========== Word2Vec application functions

def prepare_w2v(count_these):
    #global f #CLEANUP?
    if verbose: print("\nRunning Word2Vec ...\n")
    # add the current text slice to a file that can be used by w2v
    with open(w2v_corpus_file, "a") as f:
        f.write(" ".join(count_these) + " ")
    # choose the minimum word count for the w2v run: it's the lowest count in the FFA's PN layer
    occs = sorted([sum(vec) for vec in breeder.cooc])[:breeder.fruitfly.max_pn_size]
    w2v_min_count = math.floor(occs[0] / (window * 2))  # selects the smallest number
    w2v_space_file = w2v_space_location + "space.txt"
    w2v_vocab_file = w2v_space_location + "space.vocab"
    return w2v_min_count, w2v_space_file, w2v_vocab_file

def execute_w2v(w2v_min_count, w2v_space_file, w2v_vocab_file):
    if verbose: print("training Word2Vec with minimum count", w2v_min_count, "...")
    # run the w2v code
    # try: #CLEANUP?
    t_w2v = time.time()
    os.system(w2v_exe_file + " " + \
              "-train " + w2v_corpus_file + \
              "-output " + w2v_space_file + \
              "-size 300 " + \
              "-window " + str(window) + \
              "-sample 1e-3 " + \
              "-negative 10 " + \
              "-iter 1 " + \
              "-min-count " + str(w2v_min_count) + \
              "-save-vocab " + w2v_vocab_file)
    t_train = time.time() - t_w2v
    # except Exception as e:
    #    with open(errorlog, "a") as f:
    #        f.write(str(e)[:500]+"\n")
    #    print("An error occured while running Word2Vec. Check", errorlog, "for further information.")
    return t_train

def eval_and_log_w2v(t_train, w2v_space_file):
    #global f #CLEANUP?
    # evaluate the w2v model
    w2v_space = utils.readDM(w2v_space_file)
    spcorr, pairs = MEN.compute_men_spearman(w2v_space, testset_file)
    with open(w2v_results_file, "a+") as f:
        f.write("RUN " + run + ":" + \
                "\tSP_CORR: " + spcorr + \
                "\tTEST_PAIRS: " + pairs + \
                "\tTRAIN_TIME: " + t_train + "\n")
    # keep internal log
    performance_summary[run].extend([spcorr, pairs, t_train])

#==========

def log_final_summary(): # TODO make this prettier!
    if verbose: print("\n\nFinished all runs.")
    # make a summary file
    with open(results_summary_file, "w") as f:
        column_labels = ["run", "new_data_in_words", "PN_size", "testset",
                         "sp.corr_before", "sp.corr_after", "sp_diff", "time-taken",
                         "w2v_score", "w2v_testset"]
        f.write("\t".join(
            column_labels[:len(performance_summary[-1])]) + "\n\n")  # choose any run's stats as length indicator
        for k, stat_tuple in performance_summary.items():
            f.write(str(k) + "\t" + "\t".join([str(v) for v in stat_tuple]) + "\n")

#========== END OF FUNCTIONS

if __name__ == '__main__': #TODO RESOLVE THE GLOBAL VARIABLES!!!
    errorlog = "spacebreeder_errorlog.txt" #CLEANUP?

    # parameter input via terminal
    param_input_files()
    param_input_FFA()
    param_input_misc()
    print("=== Let's go! ===")

    # set up resource paths, initial Fruitfly, and Incrementor
    setup_loop_environment()

    runtime_zero = time.time()
    performance_summary = {} # for a final summary
    t_thisrun = 0
    wc = 0 # word count; used for slicing
    run = 0

    for (file_n, file) in enumerate(corpus_files):
        breeder.extend_corpus(file) # read in a new file (but no counting yet)

        if test_interval_in_words is not None:
            # Count and test in word intervals
            if file_n+1 == len(corpus_files) and len(breeder.words()) < wc+test_interval_in_words:
                # Last run with the remaining words of breeder.words
                t_thisrun = time.time()  # ends before logging
                run += 1
                if verbose: print("\n\nStarting the last run ...")
                breeder.flyfile, space_file, results_file, vip_words_file = new_paths(run, verbose=verbose)
                count_these = breeder.words[wc:len(breeder.words)] # take the whole rest of breeder.words()
                if verbose: print("Size of the final corpus slice:", len(count_these))
                count_test_log(count_these) # this is the actual loop content

            while len(breeder.words) >= wc+test_interval_in_words:
                # Loop as long as there are enough words for a whole slice
                t_thisrun = time.time()  # ends before logging
                run = int(wc / test_interval_in_words) + 1
                if verbose: print("\n\nStarting run",run,"...")
                breeder.flyfile, space_file, results_file, vip_words_file = new_paths(run, verbose=verbose)
                count_these = breeder.words[wc: wc + test_interval_in_words]
                if verbose: print("Size of corpus slice:", len(count_these))
                count_test_log(count_these) # this is the actual loop content
                wc += test_interval_in_words # last statement of the while loop

        elif test_filewise is True or test_interval_in_words == None:
            # Count and test once per file
            t_thisrun = time.time()  # ends before logging
            run += 1
            if verbose: print("\n\nStarting run", run, "with file", file, "...")
            breeder.flyfile, space_file, results_file, vip_words_file = new_paths(run, verbose=verbose)
            count_these = breeder.words[wc:len(breeder.words)] # count the whole newly read-in document
            if verbose: print("Size of text resource in this run:", len(count_these))
            count_test_log(count_these)  # this is the actual loop content
            wc = len(breeder.words)  #

    log_final_summary()

    print("done.")





#==========================
# try: #TODO make more try:except blocks, or get rid of them alltogether?
# except Exception as e:
#    with open(errorlog, "a") as f:
#        f.write(str(e)[:500]+"\n")
#    print("An error occured. Check", errorlog, "for further information.")

