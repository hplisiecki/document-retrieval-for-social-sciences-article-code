from gensim.models import KeyedVectors
from tqdm import tqdm
import pandas as pd
import ast
from scipy import spatial
from gensim.utils import simple_preprocess

# load the corpus and the embeddings

print('Loading the necessary files...')

corpus = pd.read_csv('after_averaging.csv')
word2vec = KeyedVectors.load('our_vectors.kv')
doc_tokenized = [simple_preprocess(doc) for doc in corpus['text']]

########################################################################################################################
#       Performs saturation calculations with regards to the word 'demokracja' (democracy)
#       Employs four different techniques:
#
#           1. Calculate the number of times the stem "demok" appears in each of the documents
#
#           2. Calculate the cosine similarity of each of the embeddings
#               in a document with the democracy embedding and sum them
#
#           3. Calculate the cosine similarity between the democracy embedding and the
#               10 most similar embeddings from each of the documents
#
#           4. Calculate the cosine similarity of each of the normal
#               document embeddings with the democracy embedding
#
#           5. Calculate the cosine similarity of each of the tfidf scores
#               based weighted document embeddings with the democracy embedding



# ######################################################################################
#                     Count the occurences of the "demok" stem                        #
# ######################################################################################
import pickle
with open("partie.txt", "rb") as fp:  # load the stopwords
   parts = pickle.load(fp)

print('Calculating the occurences of the demok stem...')

texts = []
for i in tqdm(corpus.text):
    temp = i
    for j in parts:
        if j in temp:
            temp =  temp.replace(j, '')
    texts.append(temp)

del temp

count = []
for i in tqdm(texts):
    count.append(i.count('demok'))

corpus['demok_counts'] = count

del count
del texts

#######################################################################################
#                    Cosine similarity of each word in a document                     #
#######################################################################################

print('Calculating the cosine similarity with every word in each of the documents...')

dem_val = []
missed = []
for doc in tqdm(doc_tokenized):
    val = 0
    miss = 0
    for word in doc:
        try:
            val += word2vec.similarity('demokracja', word)
        except:
            miss +=1
            continue
    dem_val.append(val)
    missed.append(miss)

corpus['dem_val'] = dem_val
corpus['misses'] = missed

dem_all = []
for i in corpus.index:
    check = corpus.lengths[i] - corpus.misses[i]
    if check > 0:
        dem_all.append(corpus.dem_val[i] / check)
    else:
        dem_all.append(None)

corpus['dem_all_words_saturation'] = dem_all




#######################################################################################
#          Cosine similarity of the 10 most similar words in a document               #
#######################################################################################

print('Calculating the cosine similarity of the 10 most similar words in each of the documents...')

dem_val = []
for doc in tqdm(doc_tokenized):
    val = []
    for word in doc:
        try:
            val.append(word2vec.similarity('demokracja', word))
        except:
            val.append(0)
            continue
    val.sort(reverse=True)
    val = sum(val[:10])
    if val == 0:
        dem_val.append(None)
    else:
        dem_val.append(val)



corpus['dem_ten_words_saturation'] = dem_val

#######################################################################################
#                Cosine similarity of the normal document embeddings                  #
#######################################################################################

print('Calculating the cosine similarity of each of the normal document embeddings...')


dem = word2vec['demokracja']

dem_val = []

for i in tqdm(corpus.normal_embeddings):
    try:
        dem_val.append(1 - spatial.distance.cosine(ast.literal_eval(i), dem))
    except:
        dem_val.append(None)

corpus['dem_norm_mean_words_saturation'] = dem_val



#######################################################################################
#   Cosine similarity of the normal TFIDF scores based weighted document embeddings   #
#######################################################################################

print('Calculating the cosine similarity of each of the TFIDF scores based weighted document embeddings...')

dem_val = []

for i in tqdm(corpus.tfidf_embeddings):
    try:
        dem_val.append(1 - spatial.distance.cosine(ast.literal_eval(i), dem))
    except:
        dem_val.append(None)

corpus['dem_tfidf_mean_words_saturation'] = dem_val

ast.literal_eval(corpus.tfidf_embeddings[0])
ast.literal_eval(corpus.normal_embeddings[0])

del dem_val

##########################################
#                 SAVE                   #
##########################################

print('Saving...')

corpus.to_csv('after_saturation.csv')

print(' The saturation calculations are finished.')



