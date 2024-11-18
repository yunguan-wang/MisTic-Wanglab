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
from torch import optim
import torch.nn.functional as F
from geopandas import GeoDataFrame  
from shapely import Polygon
# Utility function 
from MisC.utility import import_data, calculate_mask_distance, binary_gumbel_softmax_sample, extract_layer_num, mask_eval
from MisC.generate_tx_feature import generate_feature
from MisC.data_loader import generate_patch_coords, load_patch
from MisC.transcript_reassign import propose_reassignment, test_proposed_reassignment, make_reassignment
# Typing 
from typing import Tuple, Union, Optional 



class misc(nn.Module):
    def __init__(self,
                cell_centroid_x_col: str='center_x',
                cell_centroid_y_col: str='center_y',
                tx_x_col: str='global_x',
                tx_y_col: str='global_y',
                gene_col: str='gene',
                cell_col: str='cell_id',
                celltype_col: Optional[str]=None,
                leiden_res: float=1,
                preprocess: bool=True,
                max_centroid_dist: int=50,
                min_centroid_dist: int=0,
                mask_dist_cutoff: float=1,
                num_rep: int=3,
                method: str="split",
                n_cpus: int=8) -> None:
        super().__init__()
        
        # Record parameters 
        self.import_data_par = {
            cell_centroid_x_col: cell_centroid_x_col,
            cell_centroid_y_col: cell_centroid_y_col,
            tx_x_col: tx_x_col,
            tx_y_col: tx_y_col,
            gene_col: gene_col,
            cell_col: cell_col,
            celltype_col: celltype_col,
            leiden_res: leiden_res,
            preprocess: preprocess
        }
        self.calculate_mask_distance_par = {
            max_centroid_dist: max_centroid_dist,
            min_centroid_dist: min_centroid_dist
        }
        self.generate_feature_par = {
            mask_dist_cutoff: mask_dist_cutoff,
            num_rep: num_rep,
            method: method,
            n_cpus: n_cpus
        }
        
        # Create data 
        self.adata = None
        self.cell_coords = None
        self.tx_metadata = None
        self.current_layer = "counts_0"
        self.mask_distance = None
        self.intf_tx = None
        self.coord_list = None
        
        # Create parameters 
        self.model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.reassign_coefficients = None 
        self.cell_type_coefficients = None 
        self.optimizer = None
        
        
    def import_data(self,
                    cell_by_gene_counts: Union[str, pd.DataFrame],
                    cell_metadata: Union[str, pd.DataFrame],
                    cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                    detected_transcripts: Union[str, pd.DataFrame, gpd.GeoDataFrame]) -> None:
        self.adata, self.cell_coords, self.tx_metadata = import_data(cell_by_gene_counts=cell_by_gene_counts,
                                                                      cell_metadata=cell_metadata,
                                                                      cell_boundary_polygons=cell_boundary_polygons,
                                                                      detected_transcripts=detected_transcripts,
                                                                      **self.import_data_par)
        
        self.mask_distance = calculate_mask_distance(adata=self.adata,
                                                      cell_coords=self.cell_coords,
                                                      **self.calculate_mask_distance_par)

        self.intf_tx = generate_feature(adata=self.adata,
                                        layer=self.current_layer,
                                        tx_metadata=self.tx_metadata,
                                        cell_coords=self.cell_coords,
                                        mask_distance=self.mask_distance,
                                        **self.generate_feature_par)
        
        self.reassign_coefficients = nn.Linear(in_features=3, out_features=1, bias=True)
        self.cell_type_coefficients = nn.Linear(in_features=self.adata.uns['n_genes'], 
                                                out_features=self.adata.uns['counts_0_n_leiden'],
                                                bias=True)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
    def patchfy_data(self,
                    half_patch_size_x,
                    half_patch_size_y,
                    n_patches_x,
                    n_patches_y) -> None:
        self.coord_list = generate_patch_coords(adata=self.adata,
                                                half_patch_size_x=half_patch_size_x,
                                                half_patch_size_y=half_patch_size_y,
                                                n_patches_x=n_patches_x,
                                                n_patches_y=n_patches_y) 
    
    def encode(self, 
               tx_features: torch.tensor,
               temperature: float):
        reassign_logits = self.reassign_coefficients(tx_features) 
        reassign_probs = binary_gumbel_softmax_sample(logits=reassign_logits,
                                                        temperature=temperature)
        reassign_hard = torch.round(reassign_probs, deccimals=0)
        reassign_hard = (reassign_hard - reassign_probs).detach() + reassign_probs
        
        return reassign_hard, reassign_probs
        
    def decode(self, 
               updated_cell_by_gene_counts):
        cell_type_logits = self.cell_type_coefficients(updated_cell_by_gene_counts)
        return cell_type_logits
    
    def forward(self,
                tx_features, 
                cell_by_gene_counts, 
                row_index_self,
                row_index_neighbor,
                col_index,
                temperature):
        
        reassign_hard, reassign_probs = self.encode(tx_features=tx_features,
                                                    temperature=temperature)
        
        index_self = row_index_self * self.adata.uns['n_genes'] + col_index
        index_neighbor = row_index_neighbor * self.adata.uns['n_genes'] + col_index
        
        update_patch_self = torch.zeros((cell_by_gene_counts.shape[0])*(self.adata.uns['n_genes']), 
                                        1, dtype=torch.float32).scatter_add(0, index_self, reassign_hard)
        update_patch_neighbor = torch.zeros((cell_by_gene_counts.shape[0])*(self.adata.uns['n_genes']), 
                                            1, dtype=torch.float32).scatter_add(0, index_neighbor, reassign_hard)
    
        ordered_row_index = np.repeat(np.arange(cell_by_gene_counts.shape[0]), self.adata.uns['n_genes'])
        ordered_col_index = np.tile(np.arange(self.adata.uns['n_genes']), cell_by_gene_counts.shape[0])
        
        cell_by_gene_counts[ordered_row_index, ordered_col_index] -= update_patch_self.squeeze(1)
        cell_by_gene_counts[ordered_row_index, ordered_col_index] += update_patch_neighbor.squeeze(1)
        
        cell_type_logits = self.decode(updated_cell_by_gene_counts=cell_by_gene_counts)
        
        return reassign_probs, cell_type_logits
    
    def loss_function(self,
                      cell_type_logits, 
                      cell_type_labels,
                      reassign_probs):
        # Cross entropy loss (despite its name it's a loss)
        CEl = F.cross_entropy(cell_type_logits,
                             cell_type_labels, reduction='sum') 
        # Assume a priori equal probability
        log_ratio = torch.log(reassign_probs*2+1e-20)
        KLD = torch.sum(reassign_probs * log_ratio, dim=-1).sum()
        return CEl + KLD    
    
    def train(self,
              n_epochs):
        self.train()
        temperature = 1.0
        temp_min = 0.5
        ANNEAL_RATE = 0.00003
        log_interval = 10
        for epoch in range(n_epochs):
            np.random.shuffle(self.coord_list)
            temp = temperature
            train_loss = 0.0
            for minibatch_ind, coord in enumerate(self.coord_list):
                cell_by_gene_counts, tx_features, cell_type_labels, row_index_self, row_index_neighbor, col_index = load_patch(adata=self.adata,
                                                                                                                                intf_tx=self.intf_tx,
                                                                                                                                coord_tuple=coord,
                                                                                                                                layer=self.current_layer)
                self.optimizer.zero_grad()
                reassign_probs, cell_type_logits = self(tx_features=tx_features, 
                                                        cell_by_gene_counts=cell_by_gene_counts, 
                                                        row_index_self=row_index_self,
                                                        row_index_neighbor=row_index_neighbor,
                                                        col_index=col_index,
                                                        temperature=temp)
                loss = self.loss_function(cell_type_logits=cell_type_logits,
                                          cell_type_labels=cell_type_labels,
                                          reassign_probs=reassign_probs)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                if minibatch_ind % 100 == 1:
                    temp = np.maximum(temp * np.exp(-ANNEAL_RATE * minibatch_ind), temp_min) 

                if minibatch_ind % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                            epoch, minibatch_ind, len(self.coord_list),
                                100. * minibatch_ind / len(self.coord_list),
                                loss.item()))
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, train_loss / len(self.coord_list)))



    
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