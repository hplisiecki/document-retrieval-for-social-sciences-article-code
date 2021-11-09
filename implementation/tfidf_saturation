from gensim.models import Word2Vec
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel
from gensim import corpora
import numpy as np
from scipy import spatial
from tqdm import tqdm




class EmbeddingSimilarity(object):
    # Class that at start-up creates tf-idf weighted document embeddings
    # and then can be used to perform saturation on the basis of a query

    def __init__(self, corpus, size = 300, window = 5, min_count = 5, workers = 4, model = None):
        # preprocess
        doc_tokenized = [simple_preprocess(doc) for doc in corpus]
        if model is not None:
            print('Training the word embeddings')
            # train the word embeddings
            self.model = Word2Vec(sentences = doc_tokenized, vector_size=size, window=window, min_count=min_count, workers=workers)
        print('Computing the TF-IDF scores')
        # self.scores = self.tfidf(corpus)
        self.dictionary = corpora.Dictionary()
        # create a bag of words for each document
        self.bow = [self.dictionary.doc2bow(doc, allow_update=True) for doc in doc_tokenized]
        # create a tf-idf model
        self.tfidf_scores = TfidfModel(self.bow, smartirs='ntc')
        # creaste a dataframe out of the corpus
        print('Creating the Document Embeddings')
        self.df = pd.DataFrame(corpus, columns=['text'])
        # create the tf-idf weighted doc embeddings
        self.df['embeddings'] = self.create_embedding(self.model, self.df['text'])


    def calculate_similarity(self, query):
        # calculate the cosine similarity between the embedding and all documents in the dataframe

        query_embedding = self.model.wv[query]
        # create a list of cosine similarities
        similarities = self.df['embeddings'].apply(lambda row: self.cosine_similarity(query_embedding, row))
        # return the cosine similarity
        return similarities

    def cosine_similarity(self, embedding1, embedding2):
        # calculate the cosine similarity between two embeddings

        return 1 - spatial.distance.cosine(embedding1, embedding2)


    def create_embedding(self, model, corpus):
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


