import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_tsne_2d(datas, perplexity=30, n_components=2, random_state=42, save_dir = 't-sne', modality = '', n_class = 10):
    """
    use t-SNE to do dimensionality reduction visualization

    """
    data = []
    labels = []
    for i in range(len(datas)):
        data.append(datas[i][0].cpu().detach().numpy())
        labels.append(datas[i][1].cpu().numpy())
    data = np.array(data)
    labels = np.array(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_class))
    custom_cmap = ListedColormap(colors)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state
    )
    tsne_results = tsne.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        scatter = plt.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            c=labels,
            cmap=custom_cmap
        )
        plt.colorbar(scatter)
    else:
        plt.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            alpha=0.5
        )
    
    plt.title(f't-SNE Visualization {modality} {n_components}d')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(save_dir)
    return plt.gcf()

def visualize_tsne_3d(datas, perplexity=30, n_components=3, random_state=42, save_dir='t-sne', modality='',n_class = 10):
    """
    use t-SNE to do 3D visualization
    
    """
    # prepare data
    data = []
    labels = []
    for i in range(len(datas)):
        data.append(datas[i][0].cpu().detach().numpy())
        labels.append(datas[i][1].cpu().numpy())
    data = np.array(data)
    labels = np.array(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_class))
    custom_cmap = ListedColormap(colors)
    # t-SNE dimension reduction
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state
    )
    tsne_results = tsne.fit_transform(data)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        tsne_results[:, 2],
        c=labels,
        cmap=custom_cmap,
        alpha=0.6
    )

    plt.colorbar(scatter)
    ax.set_title(f't-SNE Visualization {modality} {n_components}d')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')

    ax.grid(True)

    plt.savefig(save_dir)

    for angle in range(0, 360, 45):
        ax.view_init(30, angle)
        plt.savefig(f'{save_dir}_angle_{angle}_{modality}.jpg')
    
    return fig
