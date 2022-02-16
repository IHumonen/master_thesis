import sklearn

import numpy as np
import plotly.express as px


def make_plot(df, word, dimensions=3):
    df = df[df['word'] == word]
    title = word
    labels_true = df['gold_sense_id'].apply(str).to_numpy()
    pca = sklearn.decomposition.PCA(n_components=dimensions)
    d3 = pca.fit_transform(np.stack(df['embedding'].to_numpy()))
    
    if dimensions == 3:
        fig = px.scatter_3d(d3, x=0, y=1, z=2, color=labels_true, title=title, labels={'0' : 'x', '1' : 'y', '2' : 'z'})
    elif dimensions == 2:
        fig = px.scatter(d3, x=0, y=1, color=labels_true, title=title, labels={'0' : 'x', '1' : 'y'})
    fig.show()