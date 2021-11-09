
# computes the correlation of the results of the different methods

def correlate(saturation):
    # compare with model results
    saturated_corpus = pd.read_csv(f'core/{saturation}_results.csv')
    topic_num = 20
    for num in tqdm(range(topic_num)):
        temp = []
        for filename in common:
            temp.append(saturated_corpus[saturated_corpus.file==filename][f'topic_{num}'].values[0])
        stack_df[f'topic_{num}'] = temp



    corr_stack = []

    for num in range(topic_num):
        temp = []
        for code in codes:
            temp.append(stack_df[code].corr(stack_df[f'topic_{num}']))
        corr_stack.append(temp)

    corr_dataframe = pd.DataFrame(corr_stack, columns = codes)

    corr_dataframe.to_csv(f'core/correlation_{saturation}')

correlate('demok')
correlate('ten')
correlate('all')
correlate('norm_mean')
correlate('tfidf_mean')

demok_corr = pd.read_csv('core/correlation_demok')
ten_corr = pd.read_csv('core/correlation_ten')
all_corr = pd.read_csv('core/correlation_all')
norm_corr = pd.read_csv('core/correlation_norm_mean')
tfidf_corr = pd.read_csv('core/correlation_tfidf_mean')
