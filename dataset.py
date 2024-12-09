import requests
import os
from scipy.sparse import vstack
import tarfile
from scipy.io import mmread

pbmc_definition = {
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
}


class PartialDataset() :
    def __init__(self, name, url, path, data_path, compressed=True):
        self.name = name
        self.url = url
        self.path = path
        self.data_path = data_path
        self.compressed = compressed

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

    def __len__(self):
        return self.data.shape[0]


class GenomeDataset():
    def __init__(self, name, dataset_def):
        self.name = name
        self.definition = dataset_def
        self.partials = []
        self.load_definition()

    def load_definition(self):
        for key, value in self.definition.items():
            self.partials.append(PartialDataset(key, value['url'], self.name + '/' + key, value['data_path'], compressed=value['compressed']))

    def dl_dataset(self):
        print(f'Downloading dataset {self.name}...')
        for partial in self.partials:
            partial.download()
            partial.extract()

    def load_dataset(self):
        print(f'Loading dataset {self.name}...')
        self.labels = []
        for i,partial in enumerate(self.partials):
            partial.load()
            self.labels += [i] * len(partial)
        self.data = vstack([partial.data for partial in self.partials])
