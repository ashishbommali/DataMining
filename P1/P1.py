{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e5e9ad8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk\n",
    "import math\n",
    "from nltk.tokenize import RegexpTokenizer\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e79c2651",
   "metadata": {},
   "outputs": [],
   "source": [
    "corpusroot = './US_Inaugural_Addresses'\n",
    "\n",
    "documents = []\n",
    "for filename in os.listdir(corpusroot):\n",
    "    if filename.startswith('0') or filename.startswith('1'):\n",
    "        file = open(os.path.join(corpusroot, filename), \"r\", encoding='windows-1252')\n",
    "        doc = file.read()\n",
    "        file.close() \n",
    "        doc = doc.lower()\n",
    "        documents.append(doc)\n",
    "#print(doc, documents)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2efadf02",
   "metadata": {},
   "outputs": [],
   "source": [
    "#tokenizer = RegexpTokenizer(r'[a-zA-Z]+')\n",
    "#tokens = tokenizer.tokenize(\"CSE4334 and CSE5534 are taught together. IE3013 is an undergraduate course.\")\n",
    "#print(tokens)\n",
    "\n",
    "def tokenize_and_preprocess(text):\n",
    "    tokenizer = nltk.RegexpTokenizer(r'[a-zA-Z]+')\n",
    "    tokens = tokenizer.tokenize(text)\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    filtered_tokens = [word for word in tokens if word not in stop_words]\n",
    "    stemmer = PorterStemmer()\n",
    "    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]\n",
    "    return stemmed_tokens\n",
    "\n",
    "tokenized_documents = [tokenize_and_preprocess(doc) for doc in documents]\n",
    "print(tokenized_documents)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5372c62d",
   "metadata": {},
   "outputs": [],
   "source": [
    "nltk.download()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "afcb01bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(stopwords.words('english'))\n",
    "\n",
    "def remove_stopwords(text_tokens):\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    filtered_tokens = [word for word in text_tokens if word not in stop_words]\n",
    "    return filtered_tokens\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fbf63ce7",
   "metadata": {},
   "outputs": [],
   "source": [
    "#stemmer = PorterStemmer()\n",
    "#print(stemmer.stem('studying'))\n",
    "#print(stemmer.stem('vector'))\n",
    "#print(stemmer.stem('entropy'))\n",
    "#print(stemmer.stem('hispanic'))\n",
    "#print(stemmer.stem('ambassador'))\n",
    "\n",
    "def stem_text(text_tokens):\n",
    "    stemmer = PorterStemmer()\n",
    "    stemmed_tokens = [stemmer.stem(word) for word in text_tokens]\n",
    "    print(stemmed_tokens)\n",
    "    return stemmed_tokens"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "238ca2db",
   "metadata": {},
   "outputs": [],
   "source": [
    "def getidf(term):\n",
    "    doc_count = sum(1 for doc in documents if term in doc)\n",
    "    print(doc_count)\n",
    "    if doc_count == 0:\n",
    "        return 0\n",
    "    return math.log(len(documents) / (1 + doc_count))\n",
    "getidf('british')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d71abb15",
   "metadata": {},
   "outputs": [],
   "source": [
    "def getweight(term, document, documents):\n",
    "    tf = document.count(term)\n",
    "    idf = calculate_idf(term, documents)\n",
    "    print(tf, idf)\n",
    "    return tf * idf\n",
    "getweight('chosen', doc, documents)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4725fcd4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def query(query_text, documents):\n",
    "    query_tokens = tokenize_text(query_text)\n",
    "    query_scores = {}\n",
    "    \n",
    "    for term in query_tokens:\n",
    "        query_scores[term] = calculate_tfidf(term, query_text, documents)\n",
    "    \n",
    "    # Sort the results by TF-IDF score in descending order\n",
    "    sorted_query_scores = sorted(query_scores.items(), key=lambda x: x[1], reverse=True)\n",
    "    \n",
    "    return sorted_query_scores\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84cc2b4e",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"%.12f\" % getidf('british'))\n",
    "print(\"%.12f\" % getidf('union'))\n",
    "print(\"%.12f\" % getidf('war'))\n",
    "print(\"%.12f\" % getidf('military'))\n",
    "print(\"%.12f\" % getidf('great'))\n",
    "print(\"--------------\")\n",
    "print(\"%.12f\" % getweight('02_washington_1793.txt','arrive'))\n",
    "print(\"%.12f\" % getweight('07_madison_1813.txt','war'))\n",
    "print(\"%.12f\" % getweight('12_jackson_1833.txt','union'))\n",
    "print(\"%.12f\" % getweight('09_monroe_1821.txt','british'))\n",
    "print(\"%.12f\" % getweight('05_jefferson_1805.txt','public'))\n",
    "print(\"--------------\")\n",
    "print(\"(%s, %.12f)\" % query(\"pleasing people\"))\n",
    "print(\"(%s, %.12f)\" % query(\"british war\"))\n",
    "print(\"(%s, %.12f)\" % query(\"false public\"))\n",
    "print(\"(%s, %.12f)\" % query(\"people institutions\"))\n",
    "print(\"(%s, %.12f)\" % query(\"violated willingly\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "79d5d483",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(corpusroot)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "00504200",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
