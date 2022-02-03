import sklearn

import numpy as np
import plotly.express as px

from sklearn.metrics.cluster import adjusted_rand_score


def df_preparation(df, task='wiki-wiki'):

    df = df.iloc[df['positions'].dropna().index]
    df['positions'] = df['positions'].apply(lambda x: x.split(','))
    df['positions'] = df['positions'].apply(lambda x: x[0].split('-'))
    df[df['context'].apply(lambda x: len(x.split('.'))) != 1]
    df['word_form'] = df.apply(lambda x: get_word_form(x['context'], x['positions'], task), axis=1)

    return df


def get_word_location(target, tokens):
    current = ''
    current_indices = []
    for i, token in enumerate(tokens):
        if token[:2] == '##':
            current += token[2:]
            current_indices.append(i)
        else:
            current = token
            current_indices = [i]
        if current == target:
            return current_indices
    print(target, tokens)
    return 'not found'

def get_word_form(context, position, task):
    if task == 'bts-rnc':
        raw = context[int(position[0]): int(position[1])+1]
    else:
        raw = context[int(position[0]): int(position[1])]
    fixed = ''
    for letter in raw:#.lower():
        if letter.isalpha():
            fixed += letter
#             if letter != 'й':
#                 fixed += letter
#             else:
#                 fixed += 'и'

    return fixed

def masking(context, positions, mask_string='<mask>'):
    
    with_mask = ''
    
    for i, symbol in enumerate(context):
        if i == int(positions[0]):
            with_mask += mask_string
        elif int(positions[0]) < i < int(positions[1]):
            pass
        else:
            with_mask += symbol
            
    return with_mask


def make_plot(df, score):
    title = df['word'].iloc[0] + f', {score}'
    labels_true = df['gold_sense_id'].apply(str).to_numpy()
    pca = sklearn.decomposition.PCA(n_components=3)
    d3 = pca.fit_transform(np.stack(df['embedding'].to_numpy()))
    
    fig = px.scatter_3d(d3, x=0, y=1, z=2, color=labels_true, title=title)
    fig.show()


def clustering(train_df, clusterizator_class, kwargs=None, print_every=11, task='wiki-wiki'):
    number_of_clusters = {'wiki-wiki' : 2, 'bts-rnc' : 3, 'active-dict' : 3}

    words_info = {}
    total = 0
    ari_sum = 0
    for i, word in enumerate(set(train_df['word'])):
        df = train_df[train_df['word']==word]
        n_clusters = number_of_clusters[task]
        n_contexts = df.shape[0]
        labels_true = df['gold_sense_id'].to_numpy()
        X = df['embedding'].to_numpy()
        X = np.stack(X)
        if kwargs is None:
            clusterizator = clusterizator_class(n_clusters=n_clusters)
        else:
            clusterizator = clusterizator_class(n_clusters=n_clusters, **kwargs)
        labels_pred = clusterizator.fit_predict(X)#+1
        ari = adjusted_rand_score(labels_true, labels_pred)
        words_info[word] = {'ari' : ari, 'count' : n_contexts}
        ari_sum += ari*n_contexts
        total += n_contexts

        if task == 'wiki-wiki' or i % print_every == 0:
            make_plot(df, ari) 
        
    return words_info, total, ari_sum