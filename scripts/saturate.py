import pandas as pd
import data
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import csv
from scipy import sparse
from scipy.io import savemat, loadmat
import os
import torch
import pickle

# computes the saturation of the entirety of each subcorpus

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

dictionary = load_obj('prepared_set')


def get_all_data(docs, vocab):

    # Maximum / minimum document frequency
    max_df = 0.7
    min_df = 10 # choose desired value for min_df


    # Read data
    print('reading text file...')


    csv.field_size_limit(350000)





    # Create count vectorizer
    print('counting document frequency of words...')
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
    cvz = cvectorizer.fit_transform(docs).sign()

    # Get vocabulary
    print('building the vocabulary...')
    sum_counts = cvz.sum(axis=0)
    v_size = sum_counts.shape[1]
    sum_counts_np = np.zeros(v_size, dtype=int)
    for v in range(v_size):
        sum_counts_np[v] = sum_counts[0,v]

    del cvectorizer
    print('  initial vocabulary size: {}'.format(v_size))


    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])

    print('tokenizing documents and splitting into train/test/valid...')
    num_docs = cvz.shape[0]

    del cvz

    print('  vocabulary after removing words not in train: {}'.format(len(vocab)))


    docs_all = [[word2id[w] for w in docs[idx].split() if w in word2id] for idx in range(num_docs)]

    del docs



    print('  number of documents (all): {} [this should be equal to {}]'.format(len(docs_all), num_docs))


    # Remove empty documents
    print('removing empty documents...')

    def remove_empty(in_docs):
        return [doc for doc in in_docs if doc!=[]]

    docs_all = remove_empty(docs_all)





    # Getting lists of words and doc_indices
    print('creating lists of words...')

    def create_list_words(in_docs):
        return [x for y in in_docs for x in y]

    words_all = create_list_words(docs_all)

    print('  len(words_all): ', len(words_all))


    # Get doc indices
    print('getting doc indices...')

    def create_doc_indices(in_docs):
        aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
        return [int(x) for y in aux for x in y]

    doc_indices_all = create_doc_indices(docs_all)


    print('  len(np.unique(doc_indices_all)): {} [this should be {}]'.format(len(np.unique(doc_indices_all)), len(docs_all)))

    # Number of documents in each set
    n_docs_all = len(docs_all)

    # Remove unused variables
    del docs_all

    # Create bow representation
    print('creating bow representation...')

    def create_bow(doc_indices, words, n_docs, vocab_size):
        return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

    bow_all = create_bow(doc_indices_all, words_all, n_docs_all, len(vocab))


    del words_all
    del doc_indices_all



    # Split bow intro token/value pairs
    print('splitting bow intro token/value pairs...')

    def split_bow(bow_in, n_docs):
        indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
        counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
        return indices, counts

    bow_all_tokens, bow_all_counts = split_bow(bow_all, n_docs_all)
    del bow_all

    savemat('temp.mat', {'tokens': bow_all_tokens}, do_compression=True)

    tokens = loadmat('temp.mat')['tokens'].squeeze()

    savemat('temp.mat', {'counts': bow_all_counts}, do_compression=True)

    counts = loadmat('temp.mat')['counts'].squeeze()

    print('Data ready !!')
    print('*************')

    return {'tokens': tokens, 'counts': counts}


def saturate(saturation, docs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = f'core/etm_{saturation}_K_20_Htheta_800_Optim_adam_Clip_1.0_ThetaAct_relu_Lr_0.0005_Bsz_1000_RhoSize_300_trainEmbeddings_0'
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()

    vocab, train, valid, test = data.get_data(os.path.join(f'core/min_df_10{saturation}'))
    vocab_size = len(vocab)


    all = get_all_data(docs, vocab)

    # 1. training all
    all_tokens = all['tokens']
    all_counts = all['counts']
    num_docs_all = len(all_tokens)

    bow_norm = 1
    num_topics = 20

    saturation = []
    weighed_saturation = []
    with torch.no_grad():

        ## get most used topics
        indices = torch.tensor(range(num_docs_all))
        indices = torch.split(indices, 1000)
        thetaAvg = torch.zeros(1, num_topics).to(device)
        thetaWeightedAvg = torch.zeros(1, num_topics).to(device)
        cnt = 0
        for idx, ind in enumerate(indices):
            data_batch = data.get_batch(all_tokens, all_counts, ind, vocab_size, device)
            sums = data_batch.sum(1).unsqueeze(1)
            cnt += sums.sum(0).squeeze().cpu().numpy()
            if bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            theta, _ = model.get_theta(normalized_data_batch)
            saturation = saturation + [[float(sat) for sat in list(one)] for one in list(theta)]
            thetaAvg += theta.sum(0).unsqueeze(0) / num_docs_all
            weighed_theta = sums * theta
            weighed_saturation = weighed_saturation + [[float(sat) for sat in list(one)] for one in list(weighed_theta)]
            thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
            if idx % 100 == 0 and idx > 0:
                print('batch: {}/{}'.format(idx, len(indices)))
        thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
        print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

        ## show topics
        return saturation, weighed_saturation


def save_demok(saturated, saturation):

    corpus = pd.read_csv('after_saturation.csv')
    subset = corpus[corpus.index.isin(dictionary['demok'])]
    num_topics = 20

    for i in range(num_topics):
        subset[f'topic_{i}'] = [sat[i] for sat in saturated]

    dom = []
    for l in saturated:
        m = max(l)
        ind = [i for i, j in enumerate(l) if j == m]
        if len(ind) > 1:
            print('problym')
            break
        dom.append(ind[0])

    subset['dominant'] = dom

    subset.to_csv(f'core/{saturation}_results.csv')


def save(saturated, saturation, subset):

    num_topics = 20

    for i in range(num_topics):
        subset.loc[:, f'topic_{i}'] = [sat[i] for sat in saturated]

    dom = []
    for l in saturated:
        m = max(l)
        ind = [i for i, j in enumerate(l) if j == m]
        if len(ind) > 1:
            print('problym')
            break
        dom.append(ind[0])

    subset.loc[:, 'dominant'] = dom

    subset.to_csv(f'core/{saturation}_results.csv')

def save_demok_weighed(saturated, saturation):

    corpus = pd.read_csv('after_saturation.csv')
    subset = corpus[corpus.index.isin(dictionary['demok'])]
    num_topics = 20

    for i in range(num_topics):
        subset[f'topic_{i}'] = [sat[i] for sat in saturated]

    dom = []
    for l in saturated:
        m = max(l)
        ind = [i for i, j in enumerate(l) if j == m]
        if len(ind) > 1:
            print('problym')
            break
        dom.append(ind[0])

    subset['dominant'] = dom

    subset.to_csv(f'core/weighed_{saturation}_results.csv')


def save_weighed(saturated, saturation, subset):

    num_topics = 20

    for i in range(num_topics):
        subset.loc[:, f'topic_{i}'] = [sat[i] for sat in saturated]

    dom = []
    for l in saturated:
        m = max(l)
        ind = [i for i, j in enumerate(l) if j == m]
        if len(ind) > 1:
            print('problym')
            break
        dom.append(ind[0])

    subset.loc[:, 'dominant'] = dom

    subset.to_csv(f'core/weighed_{saturation}_results.csv')


#####
# set the name
saturation = 'demok'
corpus = pd.read_csv('after_saturation.csv')


docs = list(corpus[corpus.index.isin(dictionary[saturation])]['text'])

saturated, weighed_saturation = saturate(saturation, docs)

save_demok(saturated, saturation)

save_demok_weighed(weighed_saturation, saturation)

#####
corpus = pd.read_csv('after_saturation.csv')
saturation = 'ten'
corpus = corpus.sort_values(by = [f'dem_{saturation}_words_saturation'], ascending = False)

docs = list(corpus[corpus.index.isin(dictionary[saturation])]['text'])


saturated, weighed_saturation = saturate(saturation, docs)

save(saturated, saturation, corpus[corpus.index.isin(dictionary[saturation])])

save_weighed(weighed_saturation, saturation, corpus[corpus.index.isin(dictionary[saturation])])

#####
corpus = pd.read_csv('after_saturation.csv')
saturation = 'all'
corpus = corpus.sort_values(by = [f'dem_{saturation}_words_saturation'], ascending = False)

docs = list(corpus[corpus.index.isin(dictionary[saturation])]['text'])


saturated, weighed_saturation = saturate(saturation, docs)

save(saturated, saturation, corpus[corpus.index.isin(dictionary[saturation])])

save_weighed(weighed_saturation, saturation, corpus[corpus.index.isin(dictionary[saturation])])

#####
saturation = 'norm_mean'
corpus = corpus.sort_values(by = [f'dem_{saturation}_words_saturation'], ascending = False)

docs = list(corpus[corpus.index.isin(dictionary[saturation])]['text'])


saturated, weighed_saturation = saturate(saturation, docs)

save(saturated, saturation, corpus[corpus.index.isin(dictionary[saturation])])

save_weighed(weighed_saturation, saturation, corpus[corpus.index.isin(dictionary[saturation])])

#####
saturation = 'tfidf_mean'
corpus = corpus.sort_values(by = [f'dem_{saturation}_words_saturation'], ascending = False)

docs = list(corpus[corpus.index.isin(dictionary[saturation])]['text'])


saturated, weighed_saturation = saturate(saturation, docs)

save(saturated, saturation, corpus[corpus.index.isin(dictionary[saturation])])

save_weighed(weighed_saturation, saturation, corpus[corpus.index.isin(dictionary[saturation])])

