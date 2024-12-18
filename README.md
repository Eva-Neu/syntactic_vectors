# Computing syntactic position vectors 

Eva Neu (UMass Amherst), Maayan Keshev (The Hebrew University of Jerusalem) and Brian Dillon (UMass Amherst)


### Background 

Agreement attraction effects are sensitive to syntactic positions of target and distractor (Franck et al., 2002). We test the hypothesis that distributed vector representations of hierarchical structure can capture this effect. We build on work by Keshev et al. (2024a) who argue that sentences are encoded in working memory by binding distributed vector representations of lexical items to distributed vector representations of syntactic positions. All item-position bindings are superimposed on the same connection matrix. Therefore, recovering an item is prone to interference from items bound to similar positions, predicting interference between syntactic position vectors that are similar to each other. Keshev et al. demonstrate this effect by directly manipulating the cosine similarities of randomly generated vectors, but their position vectors do not represent hierarchical structure in a principled way. Here we provide an algorithm to derive vector representations of syntactic positions such that higher cosine similarity between position vectors corresponds to higher rates of interference.


### The model 

We set up a constituency parse for English input sentences using the Berkeley Neural Parser and enforce binary branching. Each node in the tree is assigned an orthogonal base vector depending on its syntactic category. E.g., all N/NP nodes receive the same base vector. Using the update rule from the Temporal Context Model of position-coding in working memory (Howard and Kahana, 2002), we compute the position vector of a node as the weighted sum of its own base vector and the
position vector of its mother node (if present), using a free parameter alpha:

position vector of X = alpha * base vector of X + (1 - alpha) * position vector of X's mother

Since the function applies recursively, the position vector of said mother node in turn contains the position vector of its own mother, etc. As a result, the position vector of a node contains the base vectors (the category information) of all dominating nodes. Base vectors of more distant nodes make up a smaller part of its representation.


### How to run 

The file takes two obligatory arguments: a value for the parameter alpha and an input file. E.g.:

python3 syn_vecs.py 0.3 sample_input.csv

In the input file, do not include any punctuation in the sentence and enclose the target of agreement in hash marks. 

Additionally, the file needs access to the mymatrix.txt file which contains 16 orthogonal base vectors. If the number of base vectors you define is higher than 16, you can create a new set of 32 base vectors with the create_vec.R file. 


### References

Franck, J., Vigliocco, G., & Nicol, J. (2002). Attraction in sentence production: The role of syntactic structure. <em>Language and Cognitive Processes</em>, 17(4), 371-404.

Howard, M. W., & Kahana, M. J. (2002). A distributed representation of temporal context. <em>Journal of Mathematical Psychology</em>, 46(3), 269-299.

Keshev, M., Cartner, M., Meltzer-Asscher, A., & Dillon, B. (2024). A working memory model of sentence processing as binding morphemes to syntactic positions. In <em>Proceedings of the Annual Meeting of the Cognitive Science Society</em> (Vol. 46).

