import pandas as pd
from venn import venn
import seaborn as sns
import matplotlib.pyplot as plt

corpus = pd.read_csv('after_saturation.csv')


#######################################################################################
#                 Create a Venn Diagram of all of the document sets                   #
#######################################################################################

dictionary = {}
# cut_off = 50000
cut_off = 15521

#15521


##############################
######       DEMOK      ######
##############################

name = 'count'

corpus = corpus.sort_values(by = [f'demok_counts'], ascending = False)
demok = corpus.index[:cut_off]
for value in corpus.index[:cut_off]:
    dictionary.setdefault(f'{name}', set()).add(value)


##############################
######       ALL      ########
##############################

name = 'separate words'

corpus = corpus.sort_values(by = [f'dem_all_words_saturation'], ascending = False)
all = corpus.index[:cut_off]


for value in corpus.index[:cut_off]:
    dictionary.setdefault(f'{name}', set()).add(value)

##############################
######       TEN      ########
##############################

name = 'ten words'

corpus = corpus.sort_values(by = [f'dem_ten_words_saturation'], ascending = False)
ten = corpus.index[:cut_off]


for value in corpus.index[:cut_off]:
    dictionary.setdefault(f'{name}', set()).add(value)

##############################
######       NORM      #######
##############################

name = 'norm mean'

corpus = corpus.sort_values(by = [f'dem_norm_mean_words_saturation'], ascending = False)
norm = corpus.index[:cut_off]


for value in corpus.index[:cut_off]:
    dictionary.setdefault(f'{name}', set()).add(value)


##############################
######       TFIDF      ######
##############################

name = 'tfidf mean'

corpus = corpus.sort_values(by = [f'dem_tfidf_mean_words_saturation'], ascending = False)
tfidf = corpus.index[:cut_off]
for value in corpus.index[:cut_off]:
    dictionary.setdefault(f'{name}', set()).add(value)
    
    
##############################
########   save it   #########
##############################

fig = venn(dictionary, figsize =(11, 11) )
fig.figure.savefig("graphs/venn_all.svg", format="svg",transparent=True, dpi = 1200)

