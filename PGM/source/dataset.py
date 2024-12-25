import torch
import requests
import os
from scipy.sparse import vstack, coo_matrix
import tarfile
from scipy.io import mmread
import numpy as np
import csv
import h5py


def coo_submatrix_pull(matr, rows, cols):
    """
    Pulls out an arbitrary i.e. non-contiguous submatrix out of
    a sparse.coo_matrix. 
    """
    if type(matr) != coo_matrix:
        raise TypeError('Matrix must be sparse COOrdinate format')
    
    gr = -1 * np.ones(matr.shape[0])
    gc = -1 * np.ones(matr.shape[1])
    
    lr = len(rows)
    lc = len(cols)
    
    ar = np.arange(0, lr)
    ac = np.arange(0, lc)
    gr[rows[ar]] = ar
    gc[cols[ac]] = ac
    mrow = matr.row
    mcol = matr.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]
    return coo_matrix((matr.data[newelem], np.array([gr[newrows], gc[newcols]])),(lr, lc))

def sub_pbmc_loader(path_to_data, small) :
    data = mmread(path_to_data).transpose()
    l,c = data.shape[0], data.shape[1]
    if small:
        data = coo_submatrix_pull(data, np.arange(0, int(l*0.1)), np.arange(0, c))
    return data

def cortex_loader(path_to_data, small) :
    rows = []
    gene_names = []
    with open(path_to_data) as csvfile:
        data_reader = csv.reader(csvfile, delimiter="\t")
        for i, row in enumerate(data_reader):
            if i == 1:
                precise_clusters = np.asarray(row, dtype=str)[2:]
            if i == 8:
                clusters = np.asarray(row, dtype=str)[2:]
            if i >= 11:
                rows.append(row[1:])
                gene_names.append(row[0])
    cell_types, labels = np.unique(clusters, return_inverse=True)
    _, precise_labels = np.unique(precise_clusters, return_inverse=True)
    data = np.asarray(rows, dtype=np.int32).T[1:]
    gene_names = np.asarray(gene_names, dtype=str)
    gene_indices = []

    extra_gene_indices = []
    gene_indices = np.concatenate([gene_indices, extra_gene_indices]).astype(np.int32)
    if gene_indices.size == 0:
        gene_indices = slice(None)

    data = data[:, gene_indices]
    if small:
        data = data[:1000, :]
    # gene_names = gene_names[gene_indices]
    return torch.Tensor(data), torch.Tensor(labels)

def retina_loader(path_to_data, small):
    f = h5py.File(path_to_data, 'r')
    m = np.array(f['matrix'])
    data = torch.Tensor(m).transpose(0, 1)
    l = np.array(f['col_attrs']['ClusterID'])
    labels = torch.Tensor(l).squeeze()

    if small:
        data = data[:3000, :]
        labels = labels[:3000]

    f.close()
    return data, labels


pbmc_definition = {
    'name': 'PBMC',
    'labels': 'from_dataset',
    'loader': sub_pbmc_loader,
    'data' : {
        'cd19': {
            'url': 'http://cf.10xgenomics.com/samples/cell-exp/1.1.0/b_cells/b_cells_filtered_gene_bc_matrices.tar.gz',
            'data_path': 'filtered_matrices_mex/hg19/matrix.mtx',
            'compressed': True
        },
        'cd14': {
            'url': 'http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd34/cd34_filtered_gene_bc_matrices.tar.gz',
            'data_path': 'filtered_matrices_mex/hg19/matrix.mtx',
            'compressed': True
        },
        'cd4': {
            'url': 'http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd4_t_helper/cd4_t_helper_filtered_gene_bc_matrices.tar.gz',
            'data_path': 'filtered_matrices_mex/hg19/matrix.mtx',
            'compressed': True
        },
        'cd4-25': {
            'url': 'http://cf.10xgenomics.com/samples/cell-exp/1.1.0/regulatory_t/regulatory_t_filtered_gene_bc_matrices.tar.gz',
            'data_path': 'filtered_matrices_mex/hg19/matrix.mtx',
            'compressed': True
        },
        'cd4-45-25': {
            'url': 'http://cf.10xgenomics.com/samples/cell-exp/1.1.0/naive_t/naive_t_filtered_gene_bc_matrices.tar.gz',
            'data_path': 'filtered_matrices_mex/hg19/matrix.mtx',
            'compressed': True
        },
        'cd4-45': {
            'url': 'http://cf.10xgenomics.com/samples/cell-exp/1.1.0/memory_t/memory_t_filtered_gene_bc_matrices.tar.gz',
            'data_path': 'filtered_matrices_mex/hg19/matrix.mtx',
            'compressed': True
        },
        'cd56': {
            'url': 'http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd56_nk/cd56_nk_filtered_gene_bc_matrices.tar.gz',
            'data_path': 'filtered_matrices_mex/hg19/matrix.mtx',
            'compressed': True
        },
        'cd8': {
            'url': 'http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cytotoxic_t/cytotoxic_t_filtered_gene_bc_matrices.tar.gz',
            'data_path': 'filtered_matrices_mex/hg19/matrix.mtx',
            'compressed': True
        },
        'cd8-45': {
            'url': 'http://cf.10xgenomics.com/samples/cell-exp/1.1.0/naive_cytotoxic/naive_cytotoxic_filtered_gene_bc_matrices.tar.gz',
            'data_path': 'filtered_matrices_mex/hg19/matrix.mtx',
            'compressed': True
        },
    },
}

cortex_definition = {
    'name': 'Cortex',
    'labels': 'loaded',
    'loader': cortex_loader,
    'data' : {
        'cortex': {
            'url': 'https://storage.googleapis.com/linnarsson-lab-www-blobs/blobs/cortex/expression_mRNA_17-Aug-2014.txt',
            'data_path': 'data',
            'compressed': False
        }
    },
}

retina_definition = {
    'name': 'Retina',
    'labels': 'loaded',
    'loader': retina_loader,
    'data' : {
        'retina': {
            'url': 'https://github.com/YosefLab/scVI-data/raw/master/retina.loom',
            'data_path': 'data',
            'compressed': False
        }
    },
}

class PartialDataset() :
    def __init__(self, name, url, path, data_path, loader, compressed=True, small=False):
        self.name = name
        self.url = url
        self.path = path
        self.data_path = data_path
        self.loader = loader
        self.compressed = compressed
        self.small = small

    def download(self):
        if os.path.exists(self.path):
            print(f'Dataset {self.name} already downloaded')
            return
        print(f'Downloading dataset {self.name}...')
        r = requests.get(self.url)
        os.makedirs(self.path, exist_ok=True)
        path = self.path + '/data.tar.gz' if self.compressed else self.path + '/data'
        with open(path, 'wb') as f:
            f.write(r.content)
        print(f'Dataset {self.name} downloaded')

    def extract(self):
        if not self.compressed:
            return

        tar = tarfile.open(self.path + '/data.tar.gz')
        tar.extractall(self.path)
        tar.close()
        print(f'Dataset {self.name} extracted')

    def load(self):
        x = self.loader(self.path + '/' + self.data_path, self.small)
        if type(x) is tuple :
            self.data, self.labels = x
        else :
            self.data = x

    def __len__(self):
        return self.data.shape[0]


class GenomeDataset():
    def __init__(self, dataset_def, download=True, small=False):
        self.name = dataset_def['name']
        self.definition = dataset_def
        self.small = small
        self.partials = []

        self.load_definitions()

        if download == True:
            self.dl_dataset()

        self.load_dataset()

    def load_definitions(self):
        for key, value in self.definition['data'].items():
            self.partials.append(PartialDataset(key, value['url'], self.name + '/' + key, value['data_path'], self.definition['loader'], compressed=value['compressed'], small=self.small))

    def dl_dataset(self):
        print(f'Downloading dataset {self.name}...')
        for partial in self.partials:
            partial.download()
            partial.extract()

    def load_dataset(self):
        print(f'Loading dataset {self.name}...')

        if self.definition['labels'] == 'from_dataset':
            self.labels = []

            for i,partial in enumerate(self.partials):
                partial.load()
                self.labels += [i] * len(partial)

            if len(self.partials) > 1:
                self.data = vstack([partial.data for partial in self.partials])
            else:
                self.data = self.partials[0].data
        elif self.definition['labels'] == 'loaded':
            for partial in self.partials:
                partial.load()

            if len(self.partials) > 1:
                self.data = vstack([partial.data for partial in self.partials])
                self.labels = sum([partial.labels for partial in self.partials], [])
            else:
                self.data = self.partials[0].data
                self.labels = self.partials[0].labels

    def __len__(self) :
        return self.data.shape[0]
