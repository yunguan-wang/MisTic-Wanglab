# Data IO
import os
import json 
import pandas as pd
import scanpy as sc
import geopandas as gpd
from geopandas import read_parquet
# Data manipulation 
import numpy as np
import torch 
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F
from torch.distributions import Beta
import torch.nn.utils.parametrize as parametrize
# Utility function 
from MisC.utility import import_data, calculate_mask_distance,\
    binary_gumbel_softmax_sample, extract_layer_num,\
        make_reassignment_adata, make_reassignment_tx_metadata,\
            Positive, JSONEncoder, even_split
from MisC.generate_tx_feature import generate_feature
from MisC.data_loader import generate_patch_coords, load_patch
# Typing 
from typing import Union, Optional 


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
        """_summary_

        Parameters
        ----------
        cell_centroid_x_col : str, optional
            The column containing the x coordinates of cell centroids in cell metadata file, by default 'center_x'
        cell_centroid_y_col : str, optional
            The column containing the y coordinates of cell centroids in cell metadata file, by default 'center_y'
        tx_x_col : str, optional
            The column containing the x coordinates of transcript in detected transcript file, by default 'global_x'
        tx_y_col : str, optional
            The column containing the y coordinates of transcript in detected transcript file, by default 'global_y'
        gene_col : str, optional
            The column containing the gene information of transcript in detected transcript file, by default 'gene'
        cell_col : str, optional
            The column containing the cell id of transcript in detected transcript file, by default 'cell_id'
        celltype_col : Optional[str], optional
            The column containing the cell type information in cell metadata file, by default None
        leiden_res : float, optional
            The resolution for leiden clustering, by default 1
        preprocess : bool, optional
            Whether or not to process the data. The processed data is only used for visualization, by default True
        max_centroid_dist : int, optional
            The threshold on cell-cell centroid distances beyond which we do not consider two cells being neighbors, by default 50
        min_centroid_dist : int, optional
            The threshold on cell-cell centroid distances under which we do not consider two cells being neighbors, by default 0
        mask_dist_cutoff : float, optional
            The threshold of cell-cell distances beyond which a cell is no longer considered a neighbor, by default 1
        num_rep : int, optional
            Number of pseudo bulks, by default 3
        method : str, optional
            Method to generate pseudo bulks. Can be split or bootstrap, by default "split"
        n_cpus : int, optional
            Number of cpus to be used in deseq2 differential analysis, by default 8
        """
        super().__init__()
        
        # Record parameters 
        self.import_data_par = {
            'cell_centroid_x_col': cell_centroid_x_col,
            'cell_centroid_y_col': cell_centroid_y_col,
            'tx_x_col': tx_x_col,
            'tx_y_col': tx_y_col,
            'gene_col': gene_col,
            'cell_col': cell_col,
            'celltype_col': celltype_col,
            'leiden_res': leiden_res,
            'preprocess': preprocess
        }
        self.calculate_mask_distance_par = {
            'max_centroid_dist': max_centroid_dist,
            'min_centroid_dist': min_centroid_dist
        }
        self.generate_feature_par = {
            'mask_dist_cutoff': mask_dist_cutoff,
            'num_rep': num_rep,
            'method': method,
            'n_cpus': n_cpus
        }
        
        # Create data 
        self.adata = None
        self.cell_coords = None
        self.tx_metadata = None
        self.current_layer = "counts_0"
        self.mask_distance = None
        self.intf_tx = None
        self.coord_list = None
        self.tx_to_reassign_dict = {}
        
        # Parameters for training 
        self.init_temperature = 1.0
        self.min_temperature = 0.5
        self.current_temperature = self.init_temperature
        self.ANNEAL_RATE = 0.00003
        self.log_interval = 10
        self.penalty_weight = 0.1
        
        # Create parameters 
        self.model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.reassign_coefficients = None 
        self.cell_type_coefficients = None 
        self.distance_prior = None
        self.optimizer = None
        
        
    def import_data(self,
                    cell_by_gene_counts: Union[str, pd.DataFrame],
                    cell_metadata: Union[str, pd.DataFrame],
                    cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                    detected_transcripts: Union[str, pd.DataFrame, gpd.GeoDataFrame]) -> None:
        """Read in the four pieces information needed for subsequent analysis: 
        cell-by-gene counts, cell metadata, cell boundary information, and transcript information. Some 
        basic visualization and cell clustering will be performed. 

        Parameters
        ----------
        cell_by_gene_counts : Union[str, pd.DataFrame]
            Either the path to the csv file or a pandas dataframe containing the cell-by-gene count matrix whose first 
            column is assumed to contain the ID for each cell while the rest of the columns should be the transcript counts
            for each cell. 
        cell_metadata : Union[str, pd.DataFrame]
            Either the path to the csv file or a pandas dataframe containing the metadata for each cell whose 
            first column is assumed to contain the ID for each cell. The information in the file/object should 
            at least contain the xy coordinates of the centers of cells named center_x, and center_y, respectively.
            If the users have already performed cell typing which is not required and will only be used for visualization,
            the information can be stored here as a separate column: cell_type.
        cell_boundary_polygons : Union[str, gpd.GeoDataFrame]
            Either the path to the parquet file or the geopandas GeoDataFrame containing the vertex information 
            for each cell. The first column is assumed to be the IDs for cells. It should contain one column 'Geometry'
            that records the coordinates of the vertices.
        detected_transcripts : Union[str, pd.DataFrame]
            Either the path to the csv file or a pandas dataframe containing the information of the detected transcripts.
            The first column is assumed to be some index. The information should contain the ID for a transcripte/molecule, 
            the ID of the cell it belongs to, its xy coordinate named global_x, global_y respectively, and its gene information. 
        """
        print("Importing data... This could take a while. But you already know what we are dealing with...")
        self.adata, self.cell_coords, self.tx_metadata = import_data(cell_by_gene_counts=cell_by_gene_counts,
                                                                      cell_metadata=cell_metadata,
                                                                      cell_boundary_polygons=cell_boundary_polygons,
                                                                      detected_transcripts=detected_transcripts,
                                                                      **self.import_data_par)
        print("Computing mask distances... Keep in mind, patience is a virtue...")
        self.mask_distance = calculate_mask_distance(adata=self.adata,
                                                      cell_coords=self.cell_coords,
                                                      **self.calculate_mask_distance_par)
        print("Generating features for detected transcripts... Whatever, are we there yet...")
        self.intf_tx = generate_feature(adata=self.adata,
                                        layer=self.current_layer,
                                        tx_metadata=self.tx_metadata,
                                        cell_coords=self.cell_coords,
                                        mask_distance=self.mask_distance,
                                        **self.generate_feature_par)
        
    def patchfy_data(self,
                    half_patch_size_x: float,
                    half_patch_size_y: float,
                    n_patches_x: int,
                    n_patches_y: int) -> None:
        self.coord_list = generate_patch_coords(adata=self.adata,
                                                half_patch_size_x=half_patch_size_x,
                                                half_patch_size_y=half_patch_size_y,
                                                n_patches_x=n_patches_x,
                                                n_patches_y=n_patches_y) 
    
    def initiate_parameters(self) -> None:
        
        # The reassign_coefficients determines the logit of the reassigning probability of a 
        # transcript based on computed features.
        self.reassign_coefficients = nn.Linear(in_features=3, out_features=1, bias=True)
        # Due to the nature of the crafted features, we put a positivity constraint on the coefficients
        parametrize.register_parametrization(self.reassign_coefficients, "weight", Positive())
        
        # The cell type coefficients takes the gene counts and outputs the logits for all the cell types 
        self.cell_type_coefficients = nn.Linear(in_features=self.adata.uns['n_genes'], 
                                                out_features=self.adata.uns['counts_0_n_leiden'],
                                                bias=True)
        # 
        self.distance_prior = Beta(1,10)
        # Adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
    
    def encode(self, 
               tx_features: torch.tensor,
               temperature: float):
        reassign_logits = self.reassign_coefficients(tx_features) 
        reassign_probs = binary_gumbel_softmax_sample(logits=reassign_logits,
                                                        temperature=temperature,
                                                        model_device=self.model_device)
        reassign_hard = torch.round(reassign_probs, decimals=0)
        reassign_hard = (reassign_hard - reassign_probs).detach() + reassign_probs
        
        return reassign_hard, reassign_probs
        
    def decode(self, 
               updated_cell_by_gene_counts: torch.tensor):
        cell_type_logits = self.cell_type_coefficients(updated_cell_by_gene_counts)
        return cell_type_logits
    
    def forward(self,
                tx_features: torch.tensor, 
                cell_by_gene_counts: torch.tensor, 
                row_index_self: torch.tensor,
                row_index_neighbor: torch.tensor,
                col_index: torch.tensor,
                temperature: float):
        
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
        
        return reassign_hard, reassign_probs, cell_type_logits
    
    def loss_function(self,
                      cell_type_logits: torch.tensor, 
                      cell_type_labels: torch.tensor,
                      reassign_hard: torch.tensor,
                      reassign_probs: torch.tensor,
                      neighbor_mask_distance_rank: torch.tensor,
                      verbose: bool=False):
        # Cross entropy loss (despite its name it's a loss)
        CEl = F.cross_entropy(cell_type_logits,
                              cell_type_labels, reduction='mean') 
        # Assume a priori equal probability
        log_ratio_1 = torch.log(reassign_probs*2+1e-20)
        log_ratio_0 = torch.log((1-reassign_probs)*2+1e-20)
        KLD_reassign = torch.sum(reassign_probs * log_ratio_1 + (1-reassign_probs) * log_ratio_0, dim=-1).mean()
        # Can add an entropy term but let's first sit on this idea for a while 
        # entropy = 0.0
        # penalty = torch.sum(reassign_hard * tx_mask_distance, dim=-1).mean()
        reassign_ind = torch.nonzero(reassign_hard, as_tuple=True)
        KLD_dist = -self.distance_prior.log_prob(neighbor_mask_distance_rank[reassign_ind]).mean()
        
        l2_regularization = torch.sum(torch.square(self.reassign_coefficients.weight))
        
        if verbose:
            print("="*30)
            print("The cross entropy loss is {} and KLD_reassign is {}".format(CEl.detach().numpy(), 
                                                                      KLD_reassign.detach().numpy()))
            print("="*30) 
        return CEl + KLD_reassign + KLD_dist + l2_regularization
    
    def training_loop(self,
                      n_epochs: int,
                      verbose: bool=False):
        self.train()
        for epoch in range(n_epochs):
            np.random.shuffle(self.coord_list)
            self.current_temperature = self.init_temperature
            train_loss = 0.0
            for minibatch_ind, coord in enumerate(self.coord_list):
                cell_by_gene_counts, tx_features, cell_type_labels, row_index_self, row_index_neighbor, col_index, neighbor_mask_distance_rank = load_patch(adata=self.adata,
                                                                                                                                                            intf_tx=self.intf_tx,
                                                                                                                                                            coord_tuple=coord,
                                                                                                                                                            layer=self.current_layer,
                                                                                                                                                            model_device=self.model_device)
                reassign_hard, reassign_probs, cell_type_logits = self(tx_features=tx_features, 
                                                                        cell_by_gene_counts=cell_by_gene_counts, 
                                                                        row_index_self=row_index_self,
                                                                        row_index_neighbor=row_index_neighbor,
                                                                        col_index=col_index,
                                                                        temperature=self.current_temperature)
                self.optimizer.zero_grad()
                loss = self.loss_function(cell_type_logits=cell_type_logits,
                                          cell_type_labels=cell_type_labels,
                                          reassign_hard=reassign_hard,
                                          reassign_probs=reassign_probs,
                                          neighbor_mask_distance_rank=neighbor_mask_distance_rank,
                                          verbose=verbose)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                if minibatch_ind % 100 == 1:
                    self.current_temperature = np.maximum(self.current_temperature * np.exp(-self.ANNEAL_RATE * minibatch_ind), self.min_temperature) 

                if minibatch_ind % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                            epoch, minibatch_ind, len(self.coord_list),
                                100. * minibatch_ind / len(self.coord_list),
                                loss.item()))
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, train_loss / len(self.coord_list)))

    def trial_reassign_tx(self,
                          criteria: dict) -> None:
        with torch.no_grad():
            self.eval()
            tx_features_chunks = even_split(array=self.intf_tx[['distance_feature', 
                                            "neighbor_self_exp_feature", 
                                            "rest_self_exp_feature"]].values,
                                            chunk_size=1000)
            reassign_hard = np.array([], dtype=float).reshape(0,1)
            reassign_probs = np.array([], dtype=float).reshape(0,1)
            for tx_features_chunk in tx_features_chunks:
                tx_features = torch.tensor(tx_features_chunk, 
                                           dtype=torch.float32, 
                                           device=self.model_device)
                reassign_hard_chunk, reassign_probs_chunk = self.encode(tx_features=tx_features,
                                                                        temperature=self.min_temperature)
                reassign_hard = np.vstack([reassign_hard, reassign_hard_chunk.numpy()])
                reassign_probs = np.vstack([reassign_probs, reassign_probs_chunk.numpy()])
            
            for criterion_name in criteria: 
                criterion = criteria[criterion_name]
                if isinstance(criterion, str):
                    self.intf_tx['reassign'] = (reassign_hard.squeeze(1)) 
                else:
                    self.intf_tx['reassign'] = (reassign_probs.squeeze(1)>criterion)  
                tx_to_reassign = self.intf_tx.loc[self.intf_tx['reassign']==1, 
                                                    ['molecule_id', 'cell_id', 'neighbor_cell_id', "gene"]]
                trial_layer = self.current_layer+"_"+criterion_name
                self.tx_to_reassign_dict[trial_layer] = tx_to_reassign.copy()
                self.adata = make_reassignment_adata(adata=self.adata,
                                                     layer=self.current_layer,
                                                     tx_to_reassign=tx_to_reassign,
                                                     trial_layer=trial_layer)
    
    def final_reassign_tx(self,
                          selected_criterion: str) -> None:
        
        tx_to_reassign = self.tx_to_reassign_dict[self.current_layer+"_"+selected_criterion].copy()
        self.adata = make_reassignment_adata(adata=self.adata, 
                                             layer=self.current_layer,
                                             tx_to_reassign=tx_to_reassign)
        self.tx_metadata = make_reassignment_tx_metadata(tx_to_reassign=tx_to_reassign,
                                                         tx_metadata=self.tx_metadata)
        layer_num = extract_layer_num(self.current_layer)
        self.current_layer = "counts_"+str(int(layer_num+1))

    def save_model(self,
                   path: str,
                   save_data: bool=True) -> None:
        dir_name = os.path.dirname(path)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, path)
        if save_data:
            self.adata.write_h5ad(os.path.join(dir_name, "adata.h5ad"))
            self.tx_metadata.to_parquet(os.path.join(dir_name, "tx_metadata.parquet"),
                                        index=True)
            self.cell_coords.to_parquet(os.path.join(dir_name, "cell_coords.parquet"),
                                        index=True)
            self.mask_distance.to_parquet(os.path.join(dir_name, "mask_distance.parquet"),
                                        index=True)
            self.intf_tx.to_parquet(os.path.join(dir_name, "intf_tx.parquet"),
                                    index=True)
            with open(os.path.join(dir_name, "tx_to_reassign_dict.json"), "w") as f:
                json.dump(self.tx_to_reassign_dict, f, cls=JSONEncoder)
        
    def load_model(self,
                   path: str,
                   load_data: bool=True) -> None:
        dir_name = os.path.dirname(path)
        checkpoint = torch.load(path) 
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if load_data:
            self.adata = sc.read_h5ad(os.path.join(dir_name, "adata.h5ad"))
            self.tx_metadata = read_parquet(os.path.join(dir_name, "tx_metadata.parquet"))
            self.cell_coords = read_parquet(os.path.join(dir_name, "cell_coords.parquet"))
            self.mask_distance = read_parquet(os.path.join(dir_name, "mask_distance.parquet"))
            self.intf_tx = read_parquet(os.path.join(dir_name, "intf_tx.parquet"))
            self.tx_to_reassign_dict = json.load(open(dir_name, "tx_to_reassign_dict.json"))
            # Then convert into dataframes 
            
    