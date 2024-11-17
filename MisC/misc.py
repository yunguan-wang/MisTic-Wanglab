# Data IO
import h5py
import os
import pandas as pd
import scanpy as sc
import geopandas as gpd
# Data manipulation 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from geopandas import GeoDataFrame  
from shapely import Polygon
# Utility function 
from MisC.utility import import_data, calculate_mask_distance, binary_gumbel_softmax_sample, extract_layer_num, mask_eval
from MisC.transcript_reassign import propose_reassignment, test_proposed_reassignment, make_reassignment
# Typing 
from typing import Tuple, Union, Optional 



class misc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adata = None
        self.cell_coords = None
        self.tx_metadata = None
        self.current_layer = "counts_0"
        self.mask_distance = None
        
        self.reassign_coefficients = None 
        self.cell_type_coefficients = None 
        self.optimizer = None
        
        
    def import_data(self,
                    cell_by_gene_counts: Union[str, pd.DataFrame],
                    cell_metadata: Union[str, pd.DataFrame],
                    cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                    detected_transcripts: Union[str, pd.DataFrame],
                    **kwargs) -> None:
        self.adata, self.cell_coords, self.tx_metadata = import_data(cell_by_gene_counts=cell_by_gene_counts,
                                                                      cell_metadata=cell_metadata,
                                                                      cell_boundary_polygons=cell_boundary_polygons,
                                                                      detected_transcripts=detected_transcripts,
                                                                      **kwargs)
        self.mask_distance = calculate_mask_distance(adata=self.adata,
                                                      cell_coords=self.cell_coords)

        self.reassign_coefficients = nn.Linear(in_features=3, out_features=1, bias=True)
        self.cell_type_coefficients = nn.Linear(in_features=self.adata.uns['n_genes'], 
                                                out_features=self.adata.uns['counts_0_n_leiden'],
                                                bias=True)
        
    def encode(self, tx_features):
        reassign_logits = self.reassign_coefficients(tx_features) 
        return reassign_logits
        
    def decode(self, updated_cell_by_gene_counts):
        cell_type_logits = self.cell_type_coefficients(updated_cell_by_gene_counts)
        cell_type_probs = F.softmax(cell_type_logits, dim=-1)
        return cell_type_probs
    
    def forward(self,
                tx_features, 
                cell_by_gene_counts, 
                row_index_self,
                row_index_neighbor,
                col_index,
                temperature):
        reassign_logits = self.encode(tx_features=tx_features)
        reassign_probs = binary_gumbel_softmax_sample(logits=reassign_logits,
                                                        temperature=temperature)
        reassign_hard = torch.round(reassign_probs, deccimals=0)
        reassign_hard = (reassign_hard - reassign_probs).detach() + reassign_probs
        
        index_self = row_index_self * self.adata.uns['n_genes'] + col_index
        index_neighbor = row_index_neighbor * self.adata.uns['n_genes'] + col_index
        
        update_patch_self = torch.zeros(408*400, 1, dtype=torch.float32).scatter_add(0, index_self, reassign_hard)
        update_patch_neighbor = torch.zeros(408*400, 1, dtype=torch.float32).scatter_add(0, index_neighbor, reassign_hard)
    
        ordered_row_index = np.repead()
        ordered_col_index = np.tile()
        
        cell_by_gene_counts[ordered_row_index, ordered_col_index] -= update_patch_self.squeeze(1)
        cell_by_gene_counts[ordered_row_index, ordered_col_index] += update_patch_neighbor.squeeze(1)
        
        cell_type_probs = self.decode(updated_cell_by_gene_counts=cell_by_gene_counts)
        
        return reassign_hard, reassign_probs, cell_type_probs
    
    def train(self,
              n_epochs):
        self.train()
        true_labels = 0   
        reassign_hard, cell_type_probs = self()
        log_likelihood = F.cross_entropy(cell_type_probs, true_labels, reduction='sum')
        
        kl = 0
        
        loss = log_likelihood + kl
        
        
        

    
    def _reassign_tx(self) -> None:
        counts_to_subtract, counts_to_add, tx_to_reassign = propose_reassignment(
            adata=self.adata, 
            tx_metadata=self.tx_metadata,
            cell_coords=self.cell_coords,
            layer=self.current_layer,
            mask_distance=self.mask_distance)
        test_result = test_proposed_reassignment(
            adata=self.adata,
            layer=self.current_layer,
            counts_to_subtract=counts_to_subtract,
            counts_to_add=counts_to_add)
        self.adata, self.tx_metadata = make_reassignment(
            adata=self.adata,
            layer=self.current_layer,
            tx_metadata=self.tx_metadata,
            tx_to_reassign=tx_to_reassign,
            test_result=test_result)
    
    def reassign_tx(self, n_iter: int) -> None:
        for _ in range(n_iter):
            self._reassign_tx()
            layer_num = extract_layer_num(self.current_layer)
            self.current_layer = "counts_"+str(int(layer_num+1))
    
    ###########
    # Do not run just copied and pasted        
    @classmethod 
    def assemble_cell_coords(cls, 
                             input_path: str,
                             output_path: str) -> None:
        """Assemble the file containing the polygons of cell masks

        Parameters
        ----------
        input_path : str
            The input folder
        output_path : str
            The output folder
        """
        boundries_fn = os.listdir(input_path + '/cell_boundaries')
        for bfn in boundries_fn:
            cell_coords = pd.Series()
            # print(bfn)
            bfn = os.path.join(input_path, 'cell_boundaries', bfn)
            f = h5py.File(bfn,'r')
            diff_coords = []
            for cell in list(f['featuredata']):
                coords = []
                for i in range(7):
                    tmp = np.array((f['featuredata'][cell]['zIndex_'+str(i)]['p_0']['coordinates'][0]))
                    coords.append(tmp)
                non_unique, unique_v = mask_eval(coords)
                if non_unique:
                    print(cell)
                else: 
                    cell_coords[cell] = unique_v
            print(bfn)
            f.close()
            cell_coords = cell_coords.to_frame(name='coord')
            cell_coords['X'] = cell_coords.coord.apply(lambda x: '_'.join(x[0][:,0].round(2).astype(str)))
            cell_coords['Y'] = cell_coords.coord.apply(lambda x: '_'.join(x[0][:,1].round(2).astype(str)))
            if os.path.exists(os.path.join(input_path, 'cell_coords.csv')):
                cell_coords.iloc[:,1:].to_csv(input_path + '/cell_coords.csv', mode='a', header=False)
            else:
                    cell_coords.iloc[:,1:].to_csv(input_path + '/cell_coords.csv')
        
        cell_masks.index = ['cell_' + str(x+1) for x in cell_masks.index]
        cell_masks = cell_masks.loc[spacia_meta.index]
        cell_masks['n_polygon'] = cell_masks.X.apply(lambda x: len(x.split('_')))
        cell_masks.X = cell_masks.X.str.split('_')
        cell_masks.Y = cell_masks.Y.str.split('_')
        cell_masks['polygon'] = cell_masks.apply(
            lambda x: Polygon([(x.X[i], x.Y[i]) for i in range(x.n_polygon)]), axis=1)
        # Save as geopandas parquet file
        GeoDataFrame(
            cell_masks[['polygon']],geometry='polygon'
            ).to_parquet('cell_polygons.parquet', index=True)
        
    @classmethod 
    def foo(cls):
        pass 