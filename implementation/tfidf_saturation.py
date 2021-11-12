from gensim.models import Word2Vec
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel
from gensim import corpora
from scipy import spatial
from tqdm import tqdm
import numpy as np
import math


class EmbeddingSimilarity(object):
    # Class that at start-up creates tf-idf weighted document embeddings
    # and then can be used to perform saturation on the basis of a query

    def __init__(self, corpus, size = 300, window = 5, min_count = 5, workers = 4, model = None):
        # preprocess
        self.tokenized_documents = [simple_preprocess(doc) for doc in corpus]
        if model is None:
            print('Training the word embeddings')
            # train the word embeddings
            self.model = Word2Vec(sentences = self.tokenized_documents, vector_size=size, window=window, min_count=min_count, workers=workers)
        print('Computing the TF-IDF scores')
        # self.scores = self.tfidf(corpus)
        self.dictionary = corpora.Dictionary()
        # create a bag of words for each document
        self.bow = [self.dictionary.doc2bow(doc, allow_update=True) for doc in self.tokenized_documents]
        # create a tf-idf model
        self.tfidf_scores = TfidfModel(self.bow, smartirs='ntc')
        # creaste a dataframe out of the corpus
        print('Creating the Document Embeddings')
        self.df = pd.DataFrame(corpus, columns=['text'])
        # create the tf-idf weighted doc embeddings
        self.df['embeddings'] = self.create_embeddings(self.model, self.df['text'])


    def calculate_similarity(self, query):
        # calculate the cosine similarity between the embedding and all documents in the dataframe
        if len(query.split()) > 1:
            embeddings = []
            retained = ''
            # for each word in the query
            for q in query.split():
                try:
                    # get the embedding of the word
                    emb = self.model.wv[q]
                    retained = retained + q + ' '
                    # add the embedding weighted by the idf to the embedding set
                    embeddings.append(emb * self.idf(q, self.tokenized_documents))

                except:
                    print(f"{q} is not in the vocabulary. It will be omitted.")
            # average the embeddings
            query_embedding = np.mean(embeddings, axis=0)
            print(f'Performing saturation for the following query: {retained}')
        else:
            try:
                query_embedding = self.model.wv[query]
                print(f'Performing saturation for the following query: {query}')
            except:
                print('The query is not in the vocabulary.')
                return
        # create a list of cosine similarities
        similarities = self.df['embeddings'].apply(lambda row: self.cosine_similarity(query_embedding, row))
        # return the cosine similarity
        return similarities

    def cosine_similarity(self, embedding1, embedding2):
        # calculate the cosine similarity between two embeddings

        return 1 - spatial.distance.cosine(embedding1, embedding2)


    def create_embeddings(self, model, corpus):
        # create the embedding for each document

        embedding_set = []
        # for each document in the corpus
        for i in tqdm(range(len(corpus))):
            # create a dictionary of tf-idf scores and words
            temponary = dict(zip([self.dictionary[number[0]] for number in self.tfidf_scores[self.bow][i]],
                                 [number[1] for number in self.tfidf_scores[self.bow][i]]))
            # pick the document
            document = corpus[i]
            # create a list of words in the document
            words = document.split()
            # create a list of word embeddings for each word in the document
            embeddings = []
            for word in words:
                try:
                    embeddings.append(model.wv[word] * temponary[word])
                except:
                    continue
            # create an embedding by averaging the word embeddings
            embedding = np.mean(embeddings, axis=0)

            # add the embedding to the embedding set
            embedding_set.append(embedding)
        # return the embedding set
        return embedding_set

    def idf(self, word, tokenized_documents):
        # calculate the inverse document frequency of a word
        n = len(tokenized_documents)
        df = 0
        for doc in tokenized_documents:
            if word in doc:
                df += 1
        return math.log(n / df)

# load the after_averaging.csv file
df = pd.read_csv('after_averaging.csv')

# create a dataframe out of the corpus
emb_sim = EmbeddingSimilarity(df['text'])

# calculate the cosine similarity between the embedding and all documents in the dataframe
df['demokracja'] = emb_sim.calculate_similarity('demokracja')

# sort the dataframe by the cosine similarity
df.sort_values(by=['demokracja'], ascending=False, inplace=True)

# print the top 10 documents
print(df.head(10))
