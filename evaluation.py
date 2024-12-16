import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

class Evaluator():
    def __init__(self, X, idx, model, device, y, eval_file):
        self.X = X
        self.idx = idx
        self.model = model
        self.device = device
        self.y = y
        self.nb_classes = len(np.unique(y))
        self.eval_file = eval_file

    def evaluate_tSNE(self):
        test_idx = [int(i) for i in self.idx[int(0.9*self.idx.size):]]

        self.model.load_state_dict(torch.load(self.eval_file, map_location=self.device))
        self.model.eval()

        X_test_list = []
        for i in test_idx:
            X_test_list.append(self.X[i])
        X_test = torch.stack(X_test_list, dim=0).to(self.device)
        y_test = np.array(self.y)[test_idx]

        with torch.no_grad():
            x_latent = self.model.encoder(X_test.to_dense())

            if hasattr(self.model, 'fc_mu') and hasattr(self.model, 'fc_logvar'):
                mu = self.model.fc_mu(x_latent)
                latent = mu.cpu().numpy()
                kmeans = KMeans(n_clusters=self.nb_classes, random_state=42).fit(latent)
                # get the labels of the clusters
                clusters = kmeans.labels_

            if hasattr(self.model, 'fc_pi'):
                pi = self.model.fc_pi(x_latent)
                mus = [fc_mu(x_latent) for fc_mu in self.model.fc_mus]
                clusters = torch.argmax(pi, dim=1)
                latent = np.zeros((x_latent.size(0), x_latent.size(1)))
                for i in range(x_latent.size(0)):
                    for j in range(pi.size(1)):
                        latent[i] += (pi[i, j] * mus[j][i]).cpu().numpy()
                clusters = clusters.cpu().numpy()
                # kmeans = KMeans(n_clusters=self.nb_classes, random_state=42).fit(latent)
                # clusters = kmeans.labels_
            else:
                latent = x_latent.cpu().numpy()
                kmeans = KMeans(n_clusters=self.nb_classes, random_state=42).fit(latent)
                clusters = kmeans.labels_

        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent)

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

        rand_index = adjusted_rand_score(y_test, clusters)

        print(f"Adjusted Rand Index: {rand_index}")