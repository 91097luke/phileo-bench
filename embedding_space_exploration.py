import random

from matplotlib import pyplot as plt

from models.model_FeatureExtracter import GeoPretrainedFeatureExtractor
from utils import data_protocol, load_data
import buteo as beo
import torch
from tqdm import tqdm
import vdblite
from time import time
from uuid import uuid4
from utils.visualize import render_s2_as_rgb
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def plot_fig(anchor, top_k, bottom_k, k):
    rows = k
    columns = 3
    anchor_rgb = render_s2_as_rgb(anchor, channel_first=True)

    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(2 * columns, 2 * rows))

    fig.add_subplot(rows, columns, 1)
    plt.imshow(anchor_rgb)
    plt.axis('off')

    for i in range(rows):
        for j in range(columns):
            axes[i][j].axis('off')

    for i in range(k):
        top_rgb = render_s2_as_rgb(top_k[i]['content'], channel_first=True)
        similarity = top_k[i]['score']
        ax = fig.add_subplot(rows, columns, 2 + 3*i)
        plt.imshow(top_rgb)
        ax.set_xlabel(f'sim_score: {np.format_float_positional(similarity, 2)}')
        # plt.axis('off')

        bottom_rgb = render_s2_as_rgb(bottom_k[i]['content'], channel_first=True)
        similarity = bottom_k[i]['score']
        ax = fig.add_subplot(rows, columns, 3 + 3*i)
        plt.imshow(bottom_rgb)
        ax.set_xlabel(f'sim_score: {np.format_float_positional(similarity, 2)}')
        # plt.axis('off')

    fontsize = 16
    axes[0][0].set_title('Anchor',  fontdict={'fontsize': fontsize})
    axes[0][1].set_title('Similar',  fontdict={'fontsize': fontsize})
    axes[0][2].set_title('Dissimilar',  fontdict={'fontsize': fontsize})
    fig.tight_layout()
    plt.show()


def get_k_sim(k=5):
    vdb = vdblite.Vdb()
    vdb.load('GeoAware_contrastive_testVectors_new.vdb')

    for _ in range(100):
        rand_idx = random.randint(0, len(vdb.data)-1)

        vector = vdb.data[rand_idx]['vector']

        top_k = vdb.search(vector, field='vector', count=k+5, top_k=True)
        bottom_k = vdb.search(vector, field='vector', count=k+10, top_k=False)

        plot_fig(anchor=vdb.data[rand_idx]['content'], top_k=top_k[5:], bottom_k=bottom_k[:10], k=k)

def plot_clusters():
    vdb = vdblite.Vdb()
    vdb.load('GeoAware_contrastive_testVectors_new.vdb')

    for cluster in range(200):

        results = list()
        for i in vdb.data:
            if i['cluster'] == cluster:
                results.append(i)

        rows = 10
        columns = 10
        k = 0

        fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(2 * columns, 2 * rows))

        if len(results) > 2:
            for i in range(columns):
                for j in range(rows):
                    k = k+1
                    if len(results) > 1:
                        rand_idx = random.randint(0, len(results) - 1)
                        rgb = render_s2_as_rgb(results[rand_idx]['content'], channel_first=True)
                        del results[rand_idx]

                        fig.add_subplot(rows, columns, k)
                        plt.imshow(rgb)
                        plt.axis('off')
                        axes[i][j].axis('off')

            fig.tight_layout()
            fig.savefig(f'misc/clusters_{cluster}.png')
            plt.close()


def get_k_means():
    vdb = vdblite.Vdb()
    vdb.load('GeoAware_contrastive_testVectors_new.vdb')

    vectors = []
    for i in vdb.data:
        vectors.append(i['vector'])


    vectors = np.array(vectors)
    # pca = PCA(n_components=3).fit_transform(vectors)
    kmeans = KMeans(n_clusters=200, random_state=0, n_init="auto").fit_predict(vectors)
    for i, cluster in enumerate(kmeans):
        vdb.data[i]['cluster'] = cluster
        # vdb.data[i]['pca_vector'] = pca[i]

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for c in np.unique(kmeans):
    #     i = np.where(kmeans == c)
    #     ax.scatter(pca[i, 0], pca[i, 1], pca[i, 2], label=c)
    # ax.legend()
    # plt.show()
    # fig.savefig('pca_clusters.png')
    # fig.close()

    vdb.save('GeoAware_contrastive_testVectors_new.vdb')


def gen_vbd():
    batch_size = 32
    model = GeoPretrainedFeatureExtractor(checkpoint='/home/lcamilleri/git_repos/Phileo-contrastive-geographical-expert/trained_models/contrastive/27102023_CoreEncoderMultiHead_geo_reduce_on_plateau/CoreEncoderMultiHead_best.pt', input_channels=10)
    model.eval()
    device = 'cuda' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    x_test, y_test = data_protocol.get_testset(folder='/phileo_data/downstream/downstream_dataset_patches_np/', y='building')

    ds_test = beo.Dataset(x_test, y_test, callback=load_data.callback_decoder)

    dl_test = load_data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0,
                                   drop_last=True, generator=torch.Generator(device='cpu'))

    test_pbar = tqdm(dl_test, total=len(dl_test),
                     desc=f"Test Set")

    vdb = vdblite.Vdb()

    with torch.no_grad():

        for i, (images, labels) in enumerate(test_pbar):
            images = images[:15].to(device)
            vectors = model(images)

            for j, vector in enumerate(vectors):
                info = {'vector': vector.detach().cpu().numpy(), 'uuid': str(uuid4()), 'content': images[j].detach().cpu().numpy()}
                vdb.add(info)

    vdb.save('GeoAware_contrastive_testVectors_new.vdb')









if __name__ == '__main__':
    gen_vbd()
    get_k_means()
    plot_clusters()