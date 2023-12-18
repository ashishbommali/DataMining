import nltk
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import math

corpusroot = './US_Inaugural_Addresses'

documents = []
document_names = []

for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        document_names.append(filename)
        file.close() 
        doc = doc.lower()
        documents.append(doc)
# print(document_names, documents)

def tokenize_and_preprocess(text):
    tokenizer = nltk.RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(text)
    #print(tokens)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    #print(filtered_tokens)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    #print(stemmed_tokens)
    return stemmed_tokens

tokenized_stemmed_documents = [tokenize_and_preprocess(doc) for doc in documents]
document_dict = dict(zip(document_names, tokenized_stemmed_documents))
# print(document_dict)

def stem_text(term):
    #print(term, "=term")
    stemmer = PorterStemmer()
    stemmed_term = stemmer.stem(term)
    #print(stemmed_term)
    return stemmed_term
stem_text('sensible')

def tfScores():
    for filename, doc_tokens in document_dict.items():
        tf = {}
        for token in doc_tokens:
            tf[token] = 1 + math.log10(doc_tokens.count(token))
        
        tf_scores.append(tf)

    return tf_scores
tf_scores = []
tfScores()

def idfScores():
    total_documents = len(document_dict)
    for filename, doc_tokens in document_dict.items():
        unique_tokens = set(doc_tokens)
        #print(unique_tokens)
        for token in unique_tokens:
            idf_scores[token] = idf_scores[token] = idf_scores.get(token, 0) + 1
    #print(idf_scores)
    for term, df in idf_scores.items():
        idf_scores[term] = math.log10(total_documents /(df))
    # print(idf_scores)
    return idf_scores
idf_scores = {}
idfScores()

def tfidfScores(idf_scores):
    
    for doc_tokens, tf in zip(tokenized_stemmed_documents, tf_scores):
        tfidf = {}
        for token, frequency in tf.items():
            tfidf[token] = frequency * idf_scores.get(token, 0)
        tfidf_scores.append(tfidf)
        
    return tfidf_scores
tfidf_scores = []
tfidfScores(idf_scores)

def getMagnitude(tfidf_dict):
    return math.sqrt(sum(tfidf_score ** 2 for tfidf_score in tfidf_dict.values()))

def normalizeTfIdf(tfidf_dict):
    magnitude = getMagnitude(tfidf_dict)
    return {token: score / magnitude for token, score in tfidf_dict.items()}

normalised_tfidf_scores = [normalizeTfIdf(tfidf_dict) for tfidf_dict in tfidf_scores]
normalised_tfidf_scores = dict(zip(document_names, normalised_tfidf_scores))
# print(normalised_tfidf_scores)

def gettf(filename, token):
    token = stem_text(token)
    if filename in os.listdir(corpusroot):
        #print(filename)
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close() 
        #print(doc)
        tf = doc.count(token)
        if tf > 0:
            l_tf = 1 + math.log10(tf)
            return l_tf
    return 0
gettf('12_jackson_1833.txt','union')

def getidf(token):
    
    token = stem_text(token)
    #print(len(tokenized_stemmed_documents))
    doc_count = sum(1 for doc in tokenized_stemmed_documents if token in doc)
    #print(doc_count)
    #print(doc_count)
    
    if doc_count == 0:
        return -1 
    else:
        total_documents = len(tokenized_stemmed_documents)
        idf = math.log10(total_documents /(doc_count))
        return idf
# print(getidf('british'),getidf('union'),getidf('war'),getidf('military'))
# getidf('great')

def getweight(filename, token):
    
    token = stem_text(token)
    # print(token)
    
    l_tf = gettf(filename, token)
    idf = getidf(token)
    
    #print(l_tf*idf)
    
    #if l_tf<1:
     #   return 0
    #print("after if")
    #tf_idf = l_tf * idf
    #normalised_tfidf = normalise(tf_idf)
    #print("if b4")
    if filename in normalised_tfidf_scores:
        #print("if1")
        if token in normalised_tfidf_scores[filename]:
            #print("if2")
            return (normalised_tfidf_scores[filename][token]) # Return the normalized TF-IDF score for the token
    return 0    
    #return tf_idf
# getweight('12_jackson_1833.txt','union')

def dotProduct(query_vector, document_vector, common_tokens):
    return sum(query_vector[token] * document_vector[token] for token in common_tokens)

def getCosineSimiliarityScore(document_content, normaised_query_TfIdf):
    common_tokens = set(normaised_query_TfIdf) & set(document_content)
    #print(common_tokens)
    
    query_magnitude = getMagnitude(normaised_query_TfIdf)
    document_magnitude = getMagnitude(document_content)
    #print(query_magnitude, document_magnitude)
    
    if query_magnitude == 0 or document_magnitude == 0:
        #print("in mag=0")
        return 0
    dot_product = dotProduct(normaised_query_TfIdf, document_content, common_tokens)
    return dot_product / (query_magnitude * document_magnitude)

def query(query_text):
    query_tokens = tokenize_and_preprocess(query_text)
    query_tf_scores = {}
    total_query_tokens = len(query_tokens)
    
    for token in query_tokens:
        query_tf = query_tokens.count(token)
        query_tf_scores[token] = 1 + math.log10(query_tf)
    #print(query_tf_scores) 

    query_tf_idf_scores = {}
    for token, tf in query_tf_scores.items():
        query_tf_idf_scores[token] = tf
    #print(query_tf_idf_scores)
    
    magnitude_query = math.sqrt(sum(score ** 2 for score in query_tf_idf_scores.values()))
    normalized_query_vector = {token: tfidf / magnitude_query for token, tfidf in query_tf_idf_scores.items()}
    
    #print(normalized_query_vector)
    
    similarity_document_scores = {}
    for document_name, document_content in normalised_tfidf_scores.items():
        #print(document_name, document_content) 
        score = getCosineSimiliarityScore(document_content, normalized_query_vector)
        similarity_document_scores[document_name] = score
        
    result_tuple = sorted(similarity_document_scores.items(), key=lambda item: item[1], reverse=True)
    
    return result_tuple[0]
# query("pleasing people")

print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('war'))
print("%.12f" % getidf('military'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('02_washington_1793.txt','arrive'))
print("%.12f" % getweight('07_madison_1813.txt','war'))
print("%.12f" % getweight('12_jackson_1833.txt','union'))
print("%.12f" % getweight('09_monroe_1821.txt','british'))
print("%.12f" % getweight('05_jefferson_1805.txt','public'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))