
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
from venn import venn
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

corpus = pd.read_csv('scripts/data/after_saturation.csv')



#######################################################################################
#                 Create a Venn Diagram of all of the document sets                   #
#######################################################################################

dictionary = {}
# cut_off = 50000
cut_off = len(corpus[corpus['demok_counts'] != 0])

#15521


##############################
######       DEMOK      ######
##############################

names = ['count', 'separate words', 'ten words', 'norm mean', 'tfidf mean']
sorting_names = ['demok_counts', 'dem_all_words_saturation', 'dem_ten_words_saturation', 'dem_norm_mean_words_saturation', 'dem_tfidf_mean_words_saturation']

for name, sorting in zip(names, sorting_names):
    corpus = corpus.sort_values(by = [sorting], ascending = False)
    demok = corpus.index[:cut_off]
    for value in corpus.index[:cut_off]:
        dictionary.setdefault(f'{name}', set()).add(value)

all_indexes = []
for saturation in names:
    all_indexes.extend(list(dictionary[saturation]))
all_indexes = set(all_indexes)
picked = corpus[corpus.index.isin(

for name in names:
    temp_df = corpus.loc[dictionary[f'{name}']]
    avg_doc_length = np.mean([len(txt.split(' ')) for txt in temp_df['text'].values])
    print(f'{name}: {avg_doc_length} words per document')


##############################
######   VENN DIAGRAM  #######
##############################
hatches = ['//', '\\', '+', 'x', '.']
fig = venn(dictionary, figsize =(11, 11), hatches = hatches)
fig.figure.savefig("graphs/venn_all.jpg", format="jpg",transparent=True, dpi = 1200)


############################################################
# Venn diagram for vocabs
############################################################
def get_vocab(saturation):
    try:
        file_path = f'core1/min_df_10{saturation}/'
        file = open(os.path.join(file_path, 'vocab.pkl'), 'rb')
        vocab = pickle.load(file)
    except:
        file_path = f'scripts/core1/min_df_10{saturation}/'
        file = open(os.path.join(file_path, 'vocab.pkl'), 'rb')
        vocab = pickle.load(file)

    return vocab


names = ['count', 'ten words', 'separate words', 'norm mean', 'tfidf mean']
stack = [get_vocab(sat) for sat in names]
for vocab, name in zip(stack, names):
    print(f'{name} vocab size: {len(vocab)}')

vocab_dict = {}
for vocab, name in zip(stack, names):
    for value in vocab:
        vocab_dict.setdefault(f'{name}', set()).add(value)
venn(vocab_dict)


# see the unique dictionary of the ten
names = ['count', 'separate words', 'norm mean', 'tfidf mean', 'ten words']
loner = 'separate words'
names.remove(loner)
alone = list(vocab_dict[loner])
for word in alone.copy():
    for name in names:
        if word in vocab_dict[name]:
            if word in alone:
                alone.remove(word)

from wordcloud import WordCloud
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

translated = ' '.join(alone)
wordcloud = WordCloud(width=1600, height=800, max_font_size=50, max_words=300, background_color="white", repeat=False).generate(translated)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#######################################################################################
#                            Inspect the unique sets                                  #
#######################################################################################

# First rescale the saturation scores so that they can be easily compared
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


r_corpus = corpus.filter(['dem_tfidf_mean_words_saturation', 'dem_norm_mean_words_saturation', 'dem_ten_words_saturation', 'dem_all_words_saturation', 'demok_counts'], axis=1)

for col in r_corpus.columns:
    col_rescaled = col + '_rescaled'
    r_corpus[col_rescaled] = scaler.fit_transform(corpus[col].values.reshape(-1,1))


###################################
######       Correlate       ######
###################################


saturated = [index for value in dictionary.values() for index in value]

saturated = corpus.loc[set(saturated)]

corrMatrix = saturated.corr()
corrMatrix.to_csv("correlation_matrix.csv")

# create a new df using only the columns we want
shared_corpus = saturated.filter(['text', 'tfidf_embeddings', 'normal_embeddings', 'demok_counts', 'dem_ten_words_saturation', 'dem_all_words_saturation', 'dem_norm_mean_words_saturation', 'dem_tfidf_mean_words_saturation'], axis=1)
# reset index
shared_corpus = shared_corpus.reset_index(drop=True)
# save the df
shared_corpus.to_csv('data/shared_corpus.csv')