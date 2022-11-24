import pandas as pd

corpus = pd.read_csv(r'/hungary/data/after_saturation.csv')
cut_off = len(corpus[corpus['demok_counts'] != 0])
names = ['count', 'separate words', 'ten words', 'norm mean', 'tfidf mean', 'doc2vec']
sorting_names = ['demok_counts', 'dem_all_words_saturation', 'dem_ten_words_saturation',
                 'dem_norm_mean_words_saturation', 'dem_tfidf_mean_words_saturation']

dictionary = {}
for name, sorting in zip(names, sorting_names):
    corpus = corpus.sort_values(by=[sorting], ascending=False)
    demok = corpus.index[:cut_off]
    for value in corpus.index[:cut_off]:
        dictionary.setdefault(f'{name}', set()).add(value)

all_values = set()
for key, value in dictionary.items():
    all_values = all_values.union(value)

picked_documents_hu = corpus.loc[all_values]

picked_documents_hu = picked_documents_hu[['text', 'tfidf_embeddings', 'normal_embeddings', 'demok_counts', 'dem_ten_words_saturation',
                  'dem_all_words_saturation', 'dem_norm_mean_words_saturation', 'dem_tfidf_mean_words_saturation']]

# save
picked_documents_hu.to_csv(r'D:\PycharmProjects\saturation_experiment\for_osf\picked_documents_hu.csv', index=False)


corpus = pd.read_csv(r'D:\PycharmProjects\saturation_experiment\data\after_saturation.csv')
cut_off = len(corpus[corpus['demok_counts'] != 0])
names = ['count', 'separate words', 'ten words', 'norm mean', 'tfidf mean', 'doc2vec']
sorting_names = ['demok_counts', 'dem_all_words_saturation', 'dem_ten_words_saturation',
                 'dem_norm_mean_words_saturation', 'dem_tfidf_mean_words_saturation']

dictionary = {}
for name, sorting in zip(names, sorting_names):
    corpus = corpus.sort_values(by=[sorting], ascending=False)
    demok = corpus.index[:cut_off]
    for value in corpus.index[:cut_off]:
        dictionary.setdefault(f'{name}', set()).add(value)

all_values = set()
for key, value in dictionary.items():
    all_values = all_values.union(value)

picked_documents_pl = corpus.loc[all_values]

picked_documents_pl = picked_documents_pl[['text', 'tfidf_embeddings', 'normal_embeddings', 'demok_counts', 'dem_ten_words_saturation',
                    'dem_all_words_saturation', 'dem_norm_mean_words_saturation', 'dem_tfidf_mean_words_saturation']]
# save
picked_documents_pl.to_csv(r'D:\PycharmProjects\saturation_experiment\for_osf\picked_documents_pl.csv', index=False)