# hollow out the dataset by removing the common set of the first 50000 documents in each saturation method

import pandas as pd
from venn import venn
from tqdm import tqdm


corpus = pd.read_csv('after_saturation.csv')

############################################################
#                 Calculate the common set                #
############################################################

# create a dictionary with the indexes of the first 50000 documents in each saturation method
dictionary = {}
cut_off = 50000

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

# create a dictionary with the indexes of the empty set

empty = {}

for key in dictionary.keys():
    for value in dictionary[key]:
        if value not in common_set:
            empty.setdefault(f'{key}', set()).add(value)

a = corpus[corpus.index.isin(empty['demok'])]

corpus = pd.read_csv('after_saturation.csv')

# the cut-off is determined by the ammount of documents with at least one occurence of the "demok" stem
cut_off = 7377

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
venn(prepared_set)

# save the dictionary with the indexes of the hollow set
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(prepared_set, 'prepared_set')

    

    
