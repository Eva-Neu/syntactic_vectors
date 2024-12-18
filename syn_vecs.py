import sys
import warnings 
import re 
import numpy as np
from numpy.linalg import norm
import pandas as pd
import spacy, benepar
import nltk 
from nltk.tree import Tree, ParentedTree 


# Read matrix file
def read_matrix(prematrix):

    matrix = []

    for i in range(len(prematrix)):
        vec = [int(x) for x in prematrix.iloc[i][0].split(" ")]
        matrix.append(np.array(vec)) 

    return matrix

# Read input sent, find target and extract label info
def read_input_sent(line):

    line_ls = line.split(',')

    # Get label 
    var_label = line_ls[0]
    present = line_ls[1].split(' ')

    # Remove hash marks, get clean sentence
    indices = []
    for item in present:
        if item == '#':
            indices.append(present.index(item))
            present.remove(item)
    clean_sent = ' '.join(present)

    # Get target
    targ = ' '.join(present[indices[0]:indices[1]])

    # Check if target occurs more than once
    if clean_sent.count(targ) > 1:
        warnings.warn('The target string "' + targ + '" occurs more than once in the input sentence "' 
                      + clean_sent + '." It is not guaranteed that the program will identify the correct position vector for this target.\n')

    return clean_sent, targ, var_label

# Assign parse to input sentence
def parse(sent):

    # Create benepar tree
    doc = nlp(sent)  
    benepar_sent = list(doc.sents)[0]
    benepar_tree = benepar_sent._.parse_string

    # Turn into NLTK tree, enforce CNF
    nltk_tree = Tree.fromstring(str(benepar_tree))
    nltk.tree.transforms.chomsky_normal_form(nltk_tree, factor='right', horzMarkov=0, vertMarkov=0, childChar='', parentChar='')

    # Turn back into string, remove <> and map onto parented tree
    # (CNF transformation cannot operate on parented tree)
    interim_tree = ' '.join(str(nltk_tree).split())
    interim_tree = re.sub("<>","",interim_tree)
    tree = ParentedTree.fromstring(interim_tree)

    # If you would like to modify the parse used by the model, just specify the structure manually, e.g.:
    # if sent == 'The squirrel said that the chipmunk will die this year': 
    #     modified_tree = '(S (NP (DT The) (NN squirrel)) (VP (VP (VBD said) (SBAR (IN that) (S (NP (DT the) (NN chipmunk)) (VP (MD will) (VP (VB die) ))))) (NP (DT this) (NN year))))'
    #     tree = ParentedTree.fromstring(modified_tree) 
    # print(tree) 

    return tree

# Create dict with position vectors for all constituents of the sentence
def modifier(node, mom_vec, dict, alpha):

    # Compute position vector of a node 
    def get_pos_vec(n, mv, a):

        # Assign base vector
        for cat in cats:
            if n.label() in cat:
                base_vec = base_vecs[cat]

        # Check that you have found a base vector 
        try:    
            base_vec
        except NameError:
            print('I do not have a base vector for category "' + str(n.label()) +'" of the following token: "' + str(n) + '"')

        # Compute position vector 
        pos_vec = (base_vec * a) + (mv * (1 - a)) 

        return pos_vec
  
    position_vec = get_pos_vec(node, mom_vec, alpha)

    # Store position vec in dict using a tuple of string, syntactic representation of constituent and index as key 
    dist_str = " ".join(node.leaves())
    dist_syn = re.sub("\n","", str(node))
    dist_syn = " ".join(dist_syn.split())
    i = len(dict.keys())
    dict[(dist_str, dist_syn, i)] = position_vec

    # If we haven't reached a terminal yet, apply modifier function recursively to daughters
    if node.subtrees() != None: 
        for subtree in node.subtrees():
            if subtree.parent() == node:
                modifier(subtree, position_vec, dict, alpha)  

    return dict

################### MAIN

# Set parameters
a = float(sys.argv[1]) # contribution of base vec to position vec 

# Define categories for base vector assignments (redefine as you like)
cats = [
    ('S',), # TP
    ('SBAR', 'SBARQ', 'SQ'), # CP
    ('VP', 'VBD', 'VBN', 'VBP', 'VB', 'MD', 'VBZ', 'TO', 'VBG'), # V(P)
    ('NP', 'NN', 'PRP', 'NNP', 'NNS'), # N(P)
    ('ADJP', 'JJ', 'JJR'), # A(P)
    ('RB', 'ADVP', 'RBR'), # Adv(P)
    ('IN', 'PP'), # P(P)/C
    ('DT', 'POS', 'PRP$'), # D (including genitive 's and possessive pronouns) 
    ('CD',), # Num
    ('WHNP', 'WDT', 'WRB', 'WP'), # Wh
    ('CC',) # Conj
]

# Read file with input sentences
infile = sys.argv[2]
with open(infile) as f:
        input = f.read()
examples = (input.split('\n'))[:-1]

# Read matrix file and get base vectors ready
prematrix = pd.read_csv('mymatrix.txt', header=None)
matrix = read_matrix(prematrix)
base_vecs = {key:val for (key,val) in zip(cats, matrix)}

# Load spacy's en model, add benepar to pipeline
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

# Set up output files (getting either the cosine similarities or directly the vectors) 
sims = pd.DataFrame(columns = ['label', 'sent', 'target', 'alpha', 'var', 'distractor', 'value'])
vecs = pd.DataFrame(columns = ['item', 'vec'])
i = 0 # row index 

# Loop over each input sentence
for line in examples:

    # Extract sentence, target and label
    sent, target, label = read_input_sent(line)

    # Parse sentence
    tree = parse(sent)

    # To initialize modifier function, find base vec of top node and create empty dict
    for cat in cats:
        if tree.label() in cat:
            init_vec = base_vecs[cat]

    init_dict = {}

    # Apply modifier function to daughters of top node (the function will then apply recursively to rest of the tree)
    for subtree in tree.subtrees():
        if subtree.parent() == tree:
            pos_vecs = modifier(subtree, init_vec, init_dict, a)

    # In dict of position vecs, find key corresponding to target 
    for key in pos_vecs: 
        if target in key:
            target_key = key
            # If more than one position vec key matches the target string (e.g., head vs phrasal level),
            # use the break statement below to retrieve either the higher or the lower one 
            # break    

    # Compute cosine sim of target to each distractor 
    cos_dict = {}
    for key in pos_vecs:
        cos_dict[key] = np.dot(pos_vecs[target_key], pos_vecs[key])/(norm(pos_vecs[target_key])*norm(pos_vecs[key]))

    # Store data in output 
    for key in pos_vecs:
        sims.loc[i, ['label']] = label
        sims.loc[i, ['sent']] = sent
        sims.loc[i, ['target']] = target
        sims.loc[i, ['alpha']] = a
        sims.loc[i, ['distractor']] = str(key)
        sims.loc[i, ['value']] = cos_dict[key]

        vecs.loc[i, ['item']] = str(key)
        vecs.loc[i, ['vec']] = str(pos_vecs[key])

        i += 1

# Export to output files
similarity_data = sims.to_csv('sample_sims.csv', index = False) 
vector_data = vecs.to_csv('sample_vecs.csv', index = False)
