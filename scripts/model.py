from gut_etm import digest
import pickle
import os
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import multiprocessing
from time import time  # To time our operations

import argparse

from main_etm import etm_training

########################################################################################################################
# Create embeddings and preprocess data for modelling


########################################################################################################################
#                     Load the stpip install --upgrade setuptoolsopwords and define utility functions                 #
########################################################################################################################

def parser(path_load, saturation, save):
    parser = argparse.ArgumentParser(description='The Embedded Topic Model')

    ### data and file related arguments
    parser.add_argument('--dataset', type=str, default=saturation, help='name of corpus')
    parser.add_argument('--data_path', type=str, default= path_load, help='directory containing data')
    parser.add_argument('--emb_path', type=str, default= '', help='directory containing word embeddings')
    parser.add_argument('--save_path', type=str, default= save, help='path to save results')
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training')

    ### model-related arguments
    parser.add_argument('--num_topics', type=int, default=20, help='number of topics')
    parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
    parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
    parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
    parser.add_argument('--theta_act', type=str, default='relu',
                        help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
    parser.add_argument('--train_embeddings', type=int, default=0, help='whether to fix rho or train it')

    ### optimization-related arguments
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train...150 for 20ng 100 for others')
    parser.add_argument('--mode', type=str, default='train', help='train or eval model')
    parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
    parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
    parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
    parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
    parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
    parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

    ### evaluation, visualization, and logging-related arguments
    parser.add_argument('--num_words', type=int, default=10, help='number of words for topic viz')
    parser.add_argument('--log_interval', type=int, default=2, help='when to log training')
    parser.add_argument('--visualize_every', type=int, default=10, help='when to visualize results')
    parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
    parser.add_argument('--load_from', type=str,default='',help='the name of the ckpt to eval from')
    parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')
    parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')

    args = parser.parse_known_args()
    return args

##### THE BELOW FILES ARE LANGUAGE AND CORPUS SPECIFIC

with open("../stopwords.txt", "rb") as fp:  # load the stopwords
   stop_words = pickle.load(fp)

with open("../surnames.txt", "rb") as fp:
   surnames = pickle.load(fp)

with open("../partie.txt", "rb") as fp:  # load the stopwords
   parts = pickle.load(fp)


def remove_stopwords_and_surnames(texts):
    '''

    Simple list comprehension checking if a certain word is in the stop_words and returning those
        that aren't. Only works with tokenized lists of documents.

    '''
    stop_words.append('rok')
    texts = [[word for word in doc if word not in parts.values()] for doc in texts]
    texts = [[word for word in doc if word not in surnames] for doc in texts]
    return [[word for word in doc if word not in stop_words] for doc in texts]

def prepare_LDA(corpus):

    '''

    :param @data: the texts have to be passed in the form of a list of documents

    :return: @dictionary -
             @id2word
             @corpis
             @bigrams

    '''

    print("Preparing LDA")

    data = [simple_preprocess(doc) for doc in corpus]

    data = remove_stopwords_and_surnames(data)   # remove those stopwords

    cleaned = []
    for i in range(len(data)):  # remove numbers and empty strings
        while("" in data[i]):
            data[i].remove("")
        cleaned.append([item for item in data[i] if not item.isdigit()])


    return cleaned


def create_embeddings(docs, name, save):
    sentences = prepare_LDA(docs.to_list())
    cores = multiprocessing.cpu_count()
    model = Word2Vec(min_count=5,
                         window=5,
                         vector_size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores-1)
    t = time()
    model.build_vocab(sentences, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    t = time()
    model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    word_vectors = model.wv
    word_vectors.save(save + f'{name}.kv')


#######################################################################################
#              Preprocess the demok saturation and train embeddings                   #
#######################################################################################




save = 'core1/'
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

dictionary = load_obj('../prepared_set')




print('Preprocessing data 1/5...')

# Maximum / minimum document frequency

max_df = 0.7
min_df = 10

# Read meta-data
print('reading data...')


corpus = pd.read_csv('data/after_saturation.csv')
#############################
cut_off = len(corpus[corpus['demok_counts'] != 0])
names = ['count', 'separate words', 'ten words', 'norm mean', 'tfidf mean']
sorting_names = ['demok_counts', 'dem_all_words_saturation', 'dem_ten_words_saturation', 'dem_norm_mean_words_saturation', 'dem_tfidf_mean_words_saturation']

for name, sorting in zip(names, sorting_names):
    corpus = corpus.sort_values(by = [sorting], ascending = False)
    demok = corpus.index[:cut_off]
    for value in corpus.index[:cut_off]:
        dictionary.setdefault(f'{name}', set()).add(value)
##############################
# set the name
saturation = 'count'

path_save = save + './min_df_' + str(min_df) + saturation + '/'
if not os.path.isdir(path_save):
    os.mkdir(path_save)

docs = list(corpus[corpus.index.isin(dictionary[saturation])]['text'])

digest(docs, path_save, min_df, max_df)

print('Creating embeddings 1/5...')
create_embeddings(corpus[corpus.index.isin(dictionary[saturation])]['text'], saturation, save)




#######################################################################################
#              Preprocess the 'ten' saturation and train embeddings                   #
#######################################################################################

print('Preprocessing data 2/5...')

saturation = 'ten words'

path_save = save + './min_df_' + str(min_df) + saturation + '/'
if not os.path.isdir(path_save):
    os.mkdir(path_save)

docs = list(corpus[corpus.index.isin(dictionary[saturation])]['text'])

digest(docs, path_save, min_df, max_df)

print('Creating embeddings 2/5...')

create_embeddings(corpus[corpus.index.isin(dictionary[saturation])]['text'], saturation, save)

#######################################################################################
#              Preprocess the 'all' saturation and train embeddings                   #
#######################################################################################

print('Preprocessing data 3/5...')

saturation = 'separate words'

path_save = save + './min_df_' + str(min_df) + saturation + '/'
if not os.path.isdir(path_save):
    os.mkdir(path_save)

docs = list(corpus[corpus.index.isin(dictionary[saturation])]['text'])

digest(docs, path_save, min_df, max_df)

print('Creating embeddings 3/5...')

create_embeddings(corpus[corpus.index.isin(dictionary[saturation])]['text'], saturation, save)

#######################################################################################
#              Preprocess the 'tfidf' saturation and train embeddings                 #
#######################################################################################

print('Preprocessing data 4/5...')

saturation = 'tfidf mean'

path_save = save + './min_df_' + str(min_df) + saturation + '/'
if not os.path.isdir(path_save):
    os.mkdir(path_save)

docs = list(corpus[corpus.index.isin(dictionary[saturation])]['text'])

digest(docs, path_save, min_df, max_df)

print('Creating embeddings 4/5...')

create_embeddings(corpus[corpus.index.isin(dictionary[saturation])]['text'], saturation, save)

#######################################################################################
#            Preprocess the 'norm_mean' saturation and train embeddings               #
#######################################################################################

print('Preprocessing data 5/5...')

saturation = 'norm mean'

path_save = save + './min_df_' + str(min_df) + saturation + '/'
if not os.path.isdir(path_save):
    os.mkdir(path_save)

docs = list(corpus[corpus.index.isin(dictionary[saturation])]['text'])

digest(docs, path_save, min_df, max_df)

print('Creating embeddings 5/5...')

create_embeddings(corpus[corpus.index.isin(dictionary[saturation])]['text'], saturation, save)

print("The data is ready!")

#######################################################################################
#######################################################################################
#                            Time to run the models                                   #
#######################################################################################
#######################################################################################



print('Lets train the models.')

#######################################################################################
#   COUNT

print('Training the demok model')

saturation = 'count'

path_load = save + './min_df_' + str(min_df) + saturation

args = parser(path_load, saturation ,save)
args[0].mode = 'train'
etm_training(args[0], f'{saturation}.kv')

#   TEN WORDS

print('Training the ten model')

saturation = 'ten words'

path_load = save + './min_df_' + str(min_df) + saturation

args = parser(path_load, saturation ,save)
args[0].mode = 'train'
etm_training(args[0], f'{saturation}.kv')
#######################################################################################
#   SEPARATE WORDS

print('Training the all model')

saturation = 'separate words'

path_load = save + './min_df_' + str(min_df) + saturation

args = parser(path_load, saturation ,save)
args[0].mode = 'train'
etm_training(args[0], f'{saturation}.kv')
#######################################################################################
#   NORM MEAN

print('Training the norm model')

saturation = 'norm mean'

path_load = save + './min_df_' + str(min_df) + saturation

args = parser(path_load, saturation ,save)
args[0].mode = 'train'
etm_training(args[0], f'{saturation}.kv')
#######################################################################################
#   TFIDF MEAN

print('Training the tfidf model')

saturation = 'tfidf mean'

path_load = save + './min_df_' + str(min_df) + saturation

args = parser(path_load, saturation ,save)
args[0].mode = 'train'
etm_training(args[0], f'{saturation}.kv')


print('The training has been completed')
