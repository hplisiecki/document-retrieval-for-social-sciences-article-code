# hollow out the dataset by removing the common set of the first 50000 documents in each saturation method

import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
from venn import venn
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


corpus = pd.read_csv('scripts/data/after_saturation.csv')

############################################################
#                 Calculate the common set                #
############################################################

# create a dictionary with the indexes of the first 50000 documents in each saturation method
dictionary = {}
cut_off = 15185

##############################
######       COUNT      ######
##############################

name = 'count'

corpus = corpus.sort_values(by=[f'demok_counts'], ascending=False)
demok = corpus.index[:cut_off]
for value in corpus.index[:cut_off]:
    dictionary.setdefault(f'{name}', set()).add(value)


#########################################
######       SEPARATE WORDS      ########
#########################################

name = 'separate words'

corpus = corpus.sort_values(by=[f'dem_all_words_saturation'], ascending=False)
all = corpus.index[:cut_off]

for value in corpus.index[:cut_off]:
    dictionary.setdefault(f'{name}', set()).add(value)

####################################
######       TEN WORDS      ########
####################################

name = 'ten words'

corpus = corpus.sort_values(by=[f'dem_ten_words_saturation'], ascending=False)
ten = corpus.index[:cut_off]

for value in corpus.index[:cut_off]:
    dictionary.setdefault(f'{name}', set()).add(value)

##################################
######       NORM MEAN     #######
##################################

name = 'norm mean'

corpus = corpus.sort_values(by=[f'dem_norm_mean_words_saturation'], ascending=False)
norm = corpus.index[:cut_off]

for value in corpus.index[:cut_off]:
    dictionary.setdefault(f'{name}', set()).add(value)

###################################
######       TFIDF MEAN      ######
###################################

name = 'tfidf mean'

corpus = corpus.sort_values(by=[f'dem_tfidf_mean_words_saturation'], ascending=False)
tfidf = corpus.index[:cut_off]
for value in corpus.index[:cut_off]:
    dictionary.setdefault(f'{name}', set()).add(value)

# gather the common set

common_set = []
for value in tqdm(corpus.index):
    common = True
    for key in dictionary.keys():
        if value not in dictionary[key]:
            common = False
            break
    if common == True:
        common_set.append(value)

# get the common set
common = corpus[corpus.index.isin(common_set)]
# save
common.to_csv('scripts/data/common_set.csv')
# create a dictionary with the indexes of the empty set

empty = {}

for key in dictionary.keys():
    for value in dictionary[key]:
        if value not in common_set:
            empty.setdefault(f'{key}', set()).add(value)

empty_df = corpus[corpus.index.isin(empty['count'])]

# the cut-off is determined by the ammount of documents with at least one occurence of the "demok" stem
cut_off = len(empty_df[empty_df['demok_counts'] > 0])

# create a dictionary with the indexes of the hollow set
prepared_set = {}

for key in empty.keys():
    subset = corpus[corpus.index.isin(empty[key])]
    try:
        subset = subset.sort_values(by=[f'dem_{key}_words_saturation'], ascending=False)
    except:
        subset = subset.sort_values(by=[f'demok_counts'], ascending=False)
    temp = subset.index[:cut_off]
    for value in temp:
        prepared_set.setdefault(f'{key}', set()).add(value)

# plot the hollow set
fig = venn(prepared_set, figsize =(11, 11) )
fig.figure.savefig("graphs/ven_hollow.svg", format="svg",transparent=True, dpi = 1200)

# save the dictionary with the indexes of the hollow set
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(prepared_set, 'prepared_set')

