import requests
import os
from scipy.sparse import vstack, coo_matrix
import tarfile
from scipy.io import mmread
import numpy as np

pbmc_definition = {
    'name': 'PBMC',
    'labels': 'from_dataset',
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

class PartialDataset() :
    def __init__(self, name, url, path, data_path, compressed=True, small=False):
        self.name = name
        self.url = url
        self.path = path
        self.data_path = data_path
        self.compressed = compressed
        self.small = small

    def download(self):
        if os.path.exists(self.path):
            print(f'Dataset {self.name} already downloaded')
            return
        print(f'Downloading dataset {self.name}...')
        r = requests.get(self.url)
        os.makedirs(self.path, exist_ok=True)
        with open(self.path + '/data.tar.gz', 'wb') as f:
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
        self.data = mmread(self.path + '/' + self.data_path).transpose()
        l,c = self.data.shape[0], self.data.shape[1]
        if self.small:
            self.data = coo_submatrix_pull(self.data, np.arange(0, int(l*0.1)), np.arange(0, c))

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
            self.partials.append(PartialDataset(key, value['url'], self.name + '/' + key, value['data_path'], compressed=value['compressed'], small=self.small))

    def dl_dataset(self):
        print(f'Downloading dataset {self.name}...')
        for partial in self.partials:
            partial.download()
            partial.extract()

    def load_dataset(self):
        print(f'Loading dataset {self.name}...')

        try:
            if self.definition['labels'] == 'from_dataset':
                self.labels = []

            for i,partial in enumerate(self.partials):
                partial.load()
                if self.definition['labels'] == 'from_dataset':
                    self.labels += [i] * len(partial)

            self.data = vstack([partial.data for partial in self.partials])
        except:
            print(f'Error, dataset {self.name} could not be loaded, please check the dataset definition and download state')

    def __len__(self) :
        return self.data.shape[0]
