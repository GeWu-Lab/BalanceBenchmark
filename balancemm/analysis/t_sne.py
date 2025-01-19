import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_tsne_2d(datas, perplexity=30, n_components=2, random_state=42, save_dir = 't-sne', modality = '', n_class = 10):
    """
    使用t-SNE进行降维可视化
    
    参数:
    data: numpy数组，形状为(n_samples, n_features)
    labels: 数据标签，可选
    perplexity: t-SNE的困惑度参数
    n_components: 降维后的维度
    random_state: 随机种子
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
    使用t-SNE进行3D可视化
    
    参数:
    datas: 数据列表，每个元素包含特征和标签
    perplexity: t-SNE的困惑度参数
    n_components: 降维后的维度（这里固定为3）
    random_state: 随机种子
    save_dir: 保存图像的路径
    modality: 模态信息，用于标题
    """
    # 数据准备
    data = []
    labels = []
    for i in range(len(datas)):
        data.append(datas[i][0].cpu().detach().numpy())
        labels.append(datas[i][1].cpu().numpy())
    data = np.array(data)
    labels = np.array(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_class))
    custom_cmap = ListedColormap(colors)
    # t-SNE降维
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state
    )
    tsne_results = tsne.fit_transform(data)
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制散点图
    scatter = ax.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        tsne_results[:, 2],
        c=labels,
        cmap=custom_cmap,
        alpha=0.6
    )
    
    # 添加颜色条
    plt.colorbar(scatter)
    
    # 设置标题和标签
    ax.set_title(f't-SNE Visualization {modality} {n_components}d')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    
    # 添加网格线
    ax.grid(True)
    
    # 保存图像
    plt.savefig(save_dir)
    
    # 保存多个角度的视图
    for angle in range(0, 360, 45):
        ax.view_init(30, angle)
        plt.savefig(f'{save_dir}_angle_{angle}_{modality}.jpg')
    
    return fig
