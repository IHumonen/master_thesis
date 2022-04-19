import sklearn

import numpy as np
import pandas as pd

from sklearn.metrics.cluster import adjusted_rand_score


def static_clusters(task='wiki-wiki', word='замок', df=None):

    number_of_clusters = {'wiki-wiki' : 2, 'bts-rnc' : 3, 'active-dict' : 3}

    return number_of_clusters[task]

def clustering(train_df, clusterizator_class, kwargs=None, task='wiki-wiki', number_of_clusters=static_clusters):

    all_preds = []
    words_info = {}
    total = 0
    ari_sum = 0
    for i, word in enumerate(set(train_df['word'])):
        df = train_df[train_df['word']==word]
        n_clusters = number_of_clusters(task=task)
        n_contexts = df.shape[0]
        labels_true = df['gold_sense_id'].to_numpy()
        X = df['embedding'].to_numpy()
        X = np.stack(X)
        if kwargs is None:
            clusterizator = clusterizator_class(n_clusters=n_clusters)
        else:
            clusterizator = clusterizator_class(n_clusters=n_clusters, **kwargs)
        labels_pred = clusterizator.fit_predict(X)
        all_preds += list(labels_pred)
        ari = adjusted_rand_score(labels_true, labels_pred)
        words_info[word] = {'ari' : ari, 'count' : n_contexts}
        ari_sum += ari*n_contexts
        total += n_contexts
    words_info['mean'] = {'ari' : ari_sum/total, 'count' : total}
    clusterizator_name = str(clusterizator_class).split('.')[-1].split('\'')[0]
    train_df[clusterizator_name] = all_preds
        
    return words_info

def results_table(words_info):

    df = pd.DataFrame(words_info).T

    return df