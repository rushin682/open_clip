import os
import json
import math

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import openslide
import anndata
from anndata import AnnData

import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

# from .mu_data import MuDataset
def load_json(json_file):

    with open(json_file) as f:
        gene_sets = json.load(f)
    
    return gene_sets

class Her2stDataSet:
    """
    Args:
        root (string): Tissue Cohort Root directory path
                       Make sure to give the entire path

    Attributes:
        slide_list (list): List of WSI object containers 
                            TissueContainer: [wsi_name, wsi, patch_coords, mask, stitch]
        patch_df (DataFrame): DataFrame of patch tuples: [(x, y), wsi_index]
    """

    def __init__(self, root, ids, transform=None, gene_range='all', gene_set_list=None, include_label=False):

        super(Her2stDataSet, self).__init__()
        
        self.cnt_dir = os.path.join(root, "data", "ST-cnts") # 'data/her2st/data/ST-cnts' 
        self.img_dir = os.path.join(root, "data", "ST-imgs") # 'data/her2st/data/ST-imgs' 
        self.pos_dir = os.path.join(root, "data", "ST-spotfiles") # 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = os.path.join(root, "data", "ST-pat/lbl") # 'data/her2st/data/ST-pat/lbl'
        self.patch_level = 0
        self.r = 224//2
        self.spot_diameter_fullres = 224 
        self.label_encoding = {'invasive cancer': 0,
                               'connective tissue': 1,
                               'undetermined': 2,
                               'immune infiltrate': 3,
                               'adipose tissue': 4,
                               'cancer in situ': 5,
                               'breast glands': 6}
        self.eval_metrics = ['acc', 'precision', 'recall', 'cm'] # measure accuracy of the data

        self.adata_file_path = os.path.join(root, "data", "ST-labeled-adatafiles") # 'data/her2st/data/ST-adatafiles'

        if gene_set_list:
            self.gene_set_list = gene_set_list
        else:
            self.gene_set_list = ['HALLMARK_KRAS_SIGNALING_UP', 
                                  'HALLMARK_TGF_BETA_SIGNALING', 
                                  'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 
                                  'HALLMARK_INTERFERON_GAMMA_RESPONSE', 
                                  'HALLMARK_ANGIOGENESIS', 
                                  'HALLMARK_INFLAMMATORY_RESPONSE', 
                                  'HALLMARK_HYPOXIA', 
                                  'HALLMARK_G2M_CHECKPOINT', 
                                  'HALLMARK_APOPTOSIS', 
                                  'HALLMARK_TNFA_SIGNALING_VIA_NFKB']
        self.gene_sets = load_json(os.path.join(root, "data", "hallmark_genesets.json"))
        self.gene_range = gene_range

        self.slides, self.slide_to_idx = self.find_slides(ids)

        self.patch_coords = self.make_dataset(self.slide_to_idx)
        self.num_of_spots = len(self.patch_coords)
        
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor()])
        self.include_label = include_label

    def img_loader(self, coordinates, slide_index):

        slide_name =  self.slide_to_idx[slide_index]
        pre = os.path.join(self.img_dir, slide_name[0], slide_name)
        wsi_name = os.listdir(pre)[0]
        wsi_path = os.path.join(pre, wsi_name)
        wsi = openslide.open_slide(wsi_path)
        patch = wsi.read_region(coordinates, self.patch_level, (self.spot_diameter_fullres, self.spot_diameter_fullres)).convert('RGB')

        return patch

    def gene_loader(self, coordinates, slide_index, gene_range='all'):

        slide_name = self.slide_to_idx[slide_index]
        file_path = os.path.join(self.adata_file_path, slide_name + "_filtered_feature_bc_matrix")
        if os.path.exists(file_path):
            # print("{} file exists".format(slide_name + "with AnnData object generated below"))
            adata = anndata.read_h5ad(file_path)

        spot_idx = adata.obsm['spatial'].tolist().index(list(coordinates))
        
        if gene_range == 'all':
            gene_embedding = adata[spot_idx, :].X[0][:77].tolist()
            # print("Yet to be tested. Gene embedding based on coordinates", gene_embedding)

        return gene_embedding

    def label_loader(self, coordinates, slide_index):
        slide_name = self.slide_to_idx[slide_index]
        file_path = os.path.join(self.adata_file_path, slide_name + "_filtered_feature_bc_matrix")
        if os.path.exists(file_path):
            # print("{} file exists".format(slide_name + "with AnnData object generated below"))
            adata = anndata.read_h5ad(file_path)

        spot_idx = adata.obsm['spatial'].tolist().index(list(coordinates))
        
        label = self.label_encoding[adata.obs['label'][spot_idx]]

        return label

    def find_slides(self, slides = None):
        """Find the slides to be used for dataset creation with structure as follows:

            directory/
            ├── data
                └── ST-imgs
                └── ST-cnts
                └── ST-spotfiles
                └── ST-pat
                    └── img
                    └── lbl
            ├── mudata_files
            ├── res
            ├── rsc
            └── scripts

        Args:
            directory(str): Root directory path, corresponding to ``self.root``
        Raises:
            FileNotFoundError: If ``dir`` has no processed_list_autogen.csv file or no files at all.
        Returns:
            (Tuple[List[str], Dict[int, str]]): List of all slides and dictionary mapping each slide to an index.
        """
        slide_to_idx = {}
        
        slide_to_idx = {i: slide_name.replace('\n', '') for i, slide_name in enumerate(slides)}
        
        return slides, slide_to_idx
    
    def make_dataset(self, slide_to_idx=None):

        """Generates a list of samples of a form (coords, slide).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            slide_to_idx (Dict[int, str]): Dictionary mapping slide name to slide index.
        Raises:
            FileNotFoundError: In case no valid file was found for any slide.
        Returns:
            List[Tuple[str, int]]: samples of a form (coord, slide)
        """

        # Should in general not occur
        if slide_to_idx is None:
            raise ValueError("'slide_to_idx' must have at least one entry to collect any samples.")

        instances = []
        available_slides = set()

        for (slide_index, slide_name) in slide_to_idx.items():

            file_path = os.path.join(self.adata_file_path, slide_name + "_filtered_feature_bc_matrix")
            if os.path.exists(file_path):
                print("{} file exists".format(slide_name+"_filtered_feature_bc_matrix"))
                adata = anndata.read_h5ad(file_path)
                
            else:
                os.makedirs(self.adata_file_path, exist_ok=True) 
                adata = self.generate_adata_file(slide_name, file_path)
                
            slide_coords = list(map(lambda coord: (tuple(coord), slide_index), adata.obsm['spatial']))
            instances.extend(slide_coords)

        return instances

    def generate_adata_file(self, slide_name, file_path):

            print('Loading metadata: ', slide_name)
            meta_dict, all_genes = self.get_meta(slide_name)

            exp_dict = meta_dict[all_genes].values
            center_dict = np.floor(meta_dict[['pixel_x','pixel_y']].values).astype(int)
            loc_dict = meta_dict[['x','y']].values
            in_tissue = meta_dict["selected"].values
            feature_types = ['Gene Expression'] * len(all_genes)
            spot_diameter_fullres = self.spot_diameter_fullres

            adata = AnnData(X = exp_dict, dtype=np.float32)

            adata.var_names_make_unique()
            adata.obs_names = [f"Spot_{i:d}" for i in range(adata.n_obs)]
            adata.var_names = all_genes
            adata.obs["in_tissue"] = in_tissue
            adata.obs["array_row"] = meta_dict['x'].values
            adata.obs["array_col"] = meta_dict['y'].values

            if slide_name in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                adata.obs["label"] = meta_dict['label'].values

            adata.var["gene_ids"] = all_genes
            adata.var["feature_types"] = feature_types
            adata.uns["spatial"] = {slide_name: {"scalefactors": {"spot_diameter_fullres": spot_diameter_fullres}}}
            adata.uns["library_id"] = slide_name
            adata.obsm["spatial"] = center_dict
            
            self.save_gene_set_exp(adata)
            adata.write(file_path)

            return adata

    def save_gene_set_exp(self, adata):
        
        for gs_name in self.gene_set_list:
            gene_list = self.gene_sets[gs_name]['geneSymbols']
            gene_list = self.get_overlap(adata, gene_list)
            # gene_symbols = [gene for gene in gene_symbols if gene in adata.var["gene_ids"]]
            adata.uns[gs_name] = gene_list

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv.gz'
        df = pd.read_csv(path,sep='\t',index_col=0, header=0)
        all_genes = list(df.columns)
        return df, all_genes

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        # df.set_index('id',inplace=True)
        return df


    def get_meta(self,name):
        cnt, all_genes = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        if name in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            lbl = self.get_lbl(name)
            meta = meta.join((lbl.set_index('id')))
        self.max_x = 0
        self.max_y = 0
        loc = meta[['x','y']].values
        self.max_x = max(self.max_x, loc[:,0].max())
        self.max_y = max(self.max_y, loc[:,1].max())
        return meta, all_genes

    def get_overlap(self, adata, gene_list):
        gene_set = set(gene_list)
        gene_set = gene_set&set(adata.var["gene_ids"])
        return list(gene_set)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: (sample) where sample is path to a WSI file
        """

        coordinates, slide_index = self.patch_coords[index]

        patch = self.img_loader(coordinates, slide_index)
        patch = self.transform(patch)

        gene_embedding = self.gene_loader(coordinates, slide_index, self.gene_range)
        gene_embedding = torch.as_tensor(gene_embedding, dtype=int)

        if self.include_label:
            label = self.label_loader(coordinates, slide_index)
            return (patch, gene_embedding, label)

        return (patch, gene_embedding)

    def __len__(self):
        return self.num_of_spots

        
