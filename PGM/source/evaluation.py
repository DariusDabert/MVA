import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

class Evaluator():
    def __init__(self, X, dataset, idx, model, distribution, total_count, device, y, eval_file):
        self.X = X
        self.dataset = dataset
        self.idx = idx
        self.model = model
        self.device = device
        self.y = y
        self.distribution = distribution
        self.total_count = total_count
        self.nb_classes = len(np.unique(y))
        self.eval_file = eval_file

    def evaluate(self):
        test_idx = [int(i) for i in self.idx[int(0.9*self.idx.size):]]

        self.model.load_state_dict(torch.load(f"models/{self.eval_file}.pth", map_location=self.device))
        self.model.eval()

        X_test_list = []
        for i in test_idx:
            X_test_list.append(self.X[i])
        X_test = torch.stack(X_test_list, dim=0).to(self.device)
        y_test = np.array(self.y)[test_idx]

        with torch.no_grad():
            x_latent = self.model.encoder(X_test.to_dense()) # get latent space before reparametrization

            if hasattr(self.model, 'fc_mu') and hasattr(self.model, 'fc_logvar'): # VAE
                mu = self.model.fc_mu(x_latent)
                latent = mu.cpu().numpy() # latent representation

                kmeans = KMeans(n_clusters=self.nb_classes, random_state=42).fit(latent)
                clusters = kmeans.labels_ # cluster assignments

            if hasattr(self.model, 'fc_pi'): # GMVAE
                pi = self.model.fc_pi(x_latent)
                mus = [fc_mu(x_latent) for fc_mu in self.model.fc_mus]
                clusters = torch.argmax(pi, dim=1) # cluster assignments with highest probability
                latent = np.zeros((x_latent.size(0), x_latent.size(1)))
                for i in range(x_latent.size(0)):
                    for j in range(pi.size(1)):
                        latent[i] += (pi[i, j] * mus[j][i]).cpu().numpy() # latent representation
                clusters = clusters.cpu().numpy()
                # For test
                # clusters_count = np.zeros(self.nb_classes)
                # for i in range(len(clusters)):
                #     clusters_count[clusters[i]] += 1
                # print(clusters_count)
                kmeans = KMeans(n_clusters=self.nb_classes, random_state=42).fit(latent) # k-means clustering, because GMVAE has collapsed clusters
                clusters_k = kmeans.labels_
            else:
                latent = x_latent.cpu().numpy()
                kmeans = KMeans(n_clusters=self.nb_classes, random_state=42).fit(latent)
                clusters = kmeans.labels_
            
            loss, _, _ = self.model.loss_function(X_test, self.distribution, 1, self.total_count)
            loss = loss.item()
            loss /= X_test.size(0)

        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent)

        if self.dataset == "pbmc":

            cell_types = [
                "CD19+ B cells",
                "CD34+ cells",
                "CD4+ helper T cells",
                "CD4+/CD25+ regulatory T cells",
                "CD4+/CD45RA+/CD25- naïve T cells",
                "CD4+/CD45RO+ memory T cells",
                "CD56+ natural killer cells",
                "CD8+ cytotoxic T cells",
                "CD8+/CD45RA+ naïve cytotoxic T cells"
            ]

            colors = [
                "#ff0000",  # Bright Red
                "#00ff00",  # Bright Green
                "#0000ff",  # Bright Blue
                "#ffff00",  # Yellow
                "#ff00ff",  # Magenta
                "#00ffff",  # Cyan
                "#800000",  # Maroon
                "#808000",  # Olive
                "#000000"   # Black
            ]
        elif self.dataset == "retina":
            cell_types = [
                "RBC",
                "MG",
                "BC5A",
                "BC7",
                "BC6",
                "BC5C",
                "BC1A",
                "BC3B",
                "BC1B",
                "BC2",
                "BC5D",
                "BC3A",
                "BC5B",
                "BC4",
                "BC8_9",
            ]

            colors = [
                "#ff0000",  # Bright Red
                "#00ff00",  # Bright Green
                "#0000ff",  # Bright Blue
                "#ffff00",  # Yellow
                "#ff00ff",  # Magenta
                "#00ffff",  # Cyan
                "#800000",  # Maroon
                "#808000",  # Olive
                "#000000",  # Black
                "#008000",  # Green
                "#800080",  # Purple
                "#008080",  # Teal
                "#808080",  # Gray
                "#c0c0c0",  # Silver
                "#ff8000"   # Orange
            ]


        plt.figure(figsize=(10, 8))
        for i, ctype in enumerate(cell_types):
            idxs = np.where(y_test == i)[0]
            plt.scatter(latent_2d[idxs, 0], latent_2d[idxs, 1], s=5, c=colors[i], label=ctype, alpha=0.7)

        plt.xlabel("ZtSNE1")
        plt.ylabel("ZtSNE2")
        plt.title("t-SNE of Latent Space (Test Set)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("latent_tsne_plot.png", dpi=300)
        plt.close()
        print("Plot saved as latent_tsne_plot.png")

        if hasattr(self.model, 'fc_pi'): # GMVAE
            rand_index = adjusted_rand_score(y_test, clusters)
            print(f"Adjusted Rand Index with categorical probability: {rand_index}")
            rand_index = adjusted_rand_score(y_test, clusters_k)
            print(f"Adjusted Rand Index with k-means: {rand_index}")
        else:
            rand_index = adjusted_rand_score(y_test, clusters)
            print(f"Adjusted Rand Index: {rand_index}")
        
        print(f"Loss: {loss}")