from tqdm import tqdm
import pandas as pd

from gensim.models import TfidfModel

from gensim.models import KeyedVectors
import numpy as np

import torch

from gensim import corpora
from gensim.utils import simple_preprocess

corpus = pd.read_csv('C:\\Users\\hplis\\PycharmProjects\\data_10_04(1).csv')
word2vec = KeyedVectors.load('our_vectors.kv')
doc_tokenized = [simple_preprocess(doc) for doc in corpus['text']]

########################################################################################################################
#           Performs averaging of embeddings in every document to obtain document embeddings
#           Two techniques are employed:
#               1. Normal averaging
#               2. TFIDF scores based weighted averaging

#       To be run before the saturation

#######################################################################################
#                                 NORMAL AVERAGING                                    #
#######################################################################################

doc_embeddings = []
for i in tqdm(range(len(doc_tokenized))):
    embeddings = []
    for word in doc_tokenized[i]:
        try:
            embeddings.append(torch.Tensor(word2vec.word_vec(word)))
        except:
            continue
    try:
        average = torch.mean(torch.stack(embeddings), 0)
    except:
        average = None
    doc_embeddings.append(average)

normal_embeddings = []
for i in tqdm(range(len(doc_embeddings))):
    try:
        normal_embeddings.append(list(np.array(doc_embeddings[i])))
    except:
        normal_embeddings.append(None)

corpus['normal_embeddings'] = normal_embeddings

del doc_embeddings, normal_embeddings

#######################################################################################
#                           TFIDF WEIGHTED AVERAGING                                  #
#######################################################################################

dictionary = corpora.Dictionary()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_tokenized]
# for doc in BoW_corpus:
#    print([[dictionary[id], freq] for id, freq in doc])
import numpy as np
tfidf = TfidfModel(BoW_corpus, smartirs='ntc')
# for doc in tfidf[BoW_corpus]:
#    print([[dictionary[id], np.around(freq,decimals=2)] for id, freq in doc])


doc_embeddings = []
for i in tqdm(range(len(doc_tokenized))):
    embeddings = []
    temponary = dict(zip(  [dictionary[number[0]] for number in  tfidf[BoW_corpus][i]], [number[1] for number in  tfidf[BoW_corpus][i]]  ))
    for word in doc_tokenized[i]:
        try:
            embeddings.append(torch.Tensor(word2vec.word_vec(word)) * temponary[word])
        except:
            continue   
    try:
        average = torch.mean(torch.stack(embeddings), 0)
    except:
        average = None
    doc_embeddings.append(average)


tfidf_emb = []
for i in tqdm(range(len(doc_embeddings))):
    try:
        tfidf_emb.append(list(np.array(doc_embeddings[i])))
    except:
        tfidf_emb.append(None)

corpus['tfidf_embeddings'] = tfidf_emb


##########################################
#                 SAVE                   #
##########################################

corpus.to_csv('after_averaging.csv')




