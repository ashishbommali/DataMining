# DataMining
____________________________________________________________________________________________________________________________________________
P1 (Query Document similarity score, log weighted term frequency, log weighted inverse document term frequency)
tasks:
A single .py file.</b> We can use any standard Python library. The only non-standard library/package allowed for this assignment is NLTK. our .py file must define at least the following functions:

* getidf(token): return the inverse document frequency of a token. If the token doesn't exist in the corpus, return -1. we should stem the parameter 'token' before calculating the idf score.

* getweight(filename,token): return the normalized TF-IDF weight of a token in the document named 'filename'. If the token doesn't exist in the document, return 0. we should stem the parameter 'token' before calculating the tf-idf score.

* query(qstring): return a tuple in the form of (filename of the document, score), where the document is the query answer with respect to the weighting scheme. we should stem the query tokens before calculating similarity.

note: keep the dataset files in a single folder called US_Inaugural_Addresses and Python source file outside this US_Inaugural_Addresses folder
____________________________________________________________________________________________________________________________________________
P2 ()
