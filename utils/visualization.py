import umap
from models.sentence_embedding import *
import umap.plot
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
import seaborn as sns

# check https://plotly.com/python/t-sne-and-umap-projections/ for plotly implementation


def umap_visualization(df, embedding_feature_name='sif_embeddings', n_fit_points=None,
                       n_neighbors=150, min_dist=0.15, n_components=2, metric='cosine', title='', matplot=True):
    assert hasattr(df, 'sentences_wo_punct'), "Missing attribute! The dataframe must have the column 'sentences_wo_punct'"
    assert hasattr(df, 'overall'), "Missing attribute! The dataframe must have the column 'overall'"
    assert hasattr(df,
                   embedding_feature_name), f"Missing attribute! The dataframe must have the column '{embedding_feature_name}'"
    assert n_components == 2 or n_components == 3, f"n_components must be equal to 2 or 3. Passed {n_components} instead"
    n_fit_points = df.shape[0] if n_fit_points is None or n_fit_points > df.shape[0] else n_fit_points
    train_embeddings_idx = np.random.choice(df.shape[0], n_fit_points, replace=False)
    train_embeddings = df.iloc[train_embeddings_idx][embedding_feature_name]
    umap_fit = umap.UMAP(n_neighbors=n_neighbors,
                         min_dist=min_dist,
                         n_components=n_components,
                         verbose=1,
                         metric=metric).fit(list(train_embeddings))
    projections = umap_fit.transform(list(df[embedding_feature_name]))
    if matplot:
        plt.scatter(projections[:, 0], projections[:, 1], s=np.ones((projections.shape[0]))-.9)
        plt.title(title)
        plt.show()
        plt.draw()
        plt.savefig(f"..\\images\\{embedding_feature_name}__n_neighbors_{n_neighbors}__min_dist_{min_dist}__metric_{metric}.png", dpi=100)
        return projections
    if n_components == 2:
        fig = px.scatter(
            projections, x=0, y=1,
            color=df.sentences_wo_punct,
            text=df.overall,
            render_mode='webgl'
        )
    else:
        fig = px.scatter_3d(
            projections,
            x=0, y=1, z=2,
            color=df.sentences_wo_punct,
            text=df.overall
        )
    fig.update_traces(showlegend=False)
    # fig.show()
    fig.write_html("plot.html", auto_open = False)


def tsne_visualization(df, embedding_name='sif_embeddings'):
    assert hasattr(df, 'sentences_wo_punct'), "Missing attribute! The dataframe must have the column 'sentences_wo_punct'"
    assert hasattr(df, 'overall'), "Missing attribute! The dataframe must have the column 'overall'"
    assert hasattr(df, embedding_name), f"Missing attribute! The dataframe must have the column '{embedding_name}'"

    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(np.array(list(df[embedding_name])))

    fig = px.scatter(
        projections, x=0, y=1,
        color=df.sentences_wo_punct,
        text=df.overall
    )
    fig.update_traces(showlegend=False)
    fig.show()

