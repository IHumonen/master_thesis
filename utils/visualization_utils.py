import sklearn

import numpy as np
import plotly.express as px

from matplotlib import pyplot as plt


def make_plot(df, word, dimensions=3):
    df = df[df['word'] == word]
    title = word
    labels_true = df['gold_sense_id'].apply(str).to_numpy()
    labels_true_plt = df['gold_sense_id'].apply(int).to_numpy()
    plt_colors = {0 : 'red',
                1 : 'blue',
                2 : 'black',
                3 : 'green',
                4 : 'yellow'}
    pca = sklearn.decomposition.PCA(n_components=dimensions)
    d3 = pca.fit_transform(np.stack(df['embedding'].to_numpy()))
    df    
    if dimensions == 3:
        fig = px.scatter_3d(d3, x=0, y=1, z=2, color=labels_true, title=title, labels={'0' : 'x', '1' : 'y', '2' : 'z'})

    elif dimensions == 2:
        fig = px.scatter(d3, x=0, y=1, color=labels_true, title=title, labels={'0' : 'x', '1' : 'y'})

        for i in range(max(labels_true_plt)):
            x = d3[:, 0][labels_true_plt == i+1]
            y = d3[:, 1][labels_true_plt == i+1]
            plt.scatter(x, y, c=plt_colors[i], label=i)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc = 'upper left')
        plt.title(label=title)
        plt.show()
    fig.show()
