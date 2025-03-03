# Data IO
import os
import json 
import pandas as pd
import scanpy as sc
import geopandas as gpd
from geopandas import read_parquet
import polars as pl 
# Data manipulation 
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy, ks_2samp
import torch 
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F
from torch.distributions import Beta
import torch.nn.utils.parametrize as parametrize
# Utility function 
from MisC.utility import import_data, calculate_mask_distance, binary_gumbel_softmax_sample,\
        make_reassignment_adata, Positive, JSONEncoder, even_split, process_time_ram
from MisC.generate_tx_feature import generate_feature
from MisC.data_loader import generate_patch_coords, load_patch
# User entertainment
from tqdm.auto import tqdm
# Typing 
from typing import Union, Optional, Tuple 


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
                max_centroid_dist: float=50,
                mask_dist_cutoff: float=5,
                nearest: int=1,
                prior_50_reassign_prob: float=0.01,
                prior_5_reassign_prob: float=0.5,
                reparametrize: bool=False,
                seed: int=42,
                model_device: Optional[Union[str, torch.device]] = None) -> None:
        """Instantiate a misc object 

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
        mask_dist_cutoff : float, optional
            The threshold of cell-cell distances beyond which a cell is no longer considered a neighbor, by default 1
        prior_50_reassign_prob : float, optional
            The prior probability of reassigning a transcript that's ranked 50% based on the distance, by default 0.01
        prior_5_reassign_prob : float, optional
            The prior probability of reassigning a transcript that's ranked 5% based on the distance, by default 0.5
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
            'preprocess': True
        }
        self.calculate_mask_distance_par = {
            'max_centroid_dist': max_centroid_dist,
        }
        self.generate_feature_par = {
            'mask_dist_cutoff': mask_dist_cutoff,
            'nearest': nearest,
            'seed': seed
        }
        
        # Create data 
        self.adata = None
        self.cell_coords = None
        self.tx_metadata = None
        self.current_layer = "counts_0"
        self.mask_distance = None
        self.intf_tx_dict = {"intf_tx{}".format(i): None for i in range(nearest)}
        self.intf_tx = None
        self.coord_list = {"intf_tx{}".format(i): [] for i in range(nearest)}
        self.tx_to_reassign_dict = {}
        
        # Parameters for training 
        self.init_temperature = 1.0
        self.min_temperature = 0.5
        self.current_temperature = self.init_temperature
        self.ANNEAL_RATE = 0.00003
        self.log_interval = 10
        
        # Create parameters 
        # Set model device
        if model_device is None:
            self.model_device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(model_device, str):
            self.model_device = torch.device(model_device)
        else:
            self.model_device = model_device
        self.n_genes = None
        self.n_leiden = None
        self.prior_50_reassign_prob = prior_50_reassign_prob
        self.prior_5_reassign_prob = prior_5_reassign_prob
        self.reparametrize = reparametrize
        
        self.distance_reassign_coefficients = nn.ModuleDict({})
        self.self_expression_reassign_coefficients = nn.ModuleDict({})
        self.neighbor_expression_reassign_coefficients = nn.ModuleDict({})
        self.prior_distance_reassign_coefficients = nn.ModuleDict({})
        self.prior_self_expression_reassign_coefficients = nn.ModuleDict({})
        self.prior_neighbor_expression_reassign_coefficients = nn.ModuleDict({})
        
        self.cell_type_coefficients = nn.ModuleDict({})  
        self.optimizer = {"intf_tx{}".format(i): None for i in range(nearest)}
        # For training 
        self.CEl_list = {"intf_tx{}".format(i): [] for i in range(nearest)}
        self.KLD_list = {"intf_tx{}".format(i): {"distance": [],
                                                 "self_expression": [],
                                                 "neighbor_expression": []} for i in range(nearest)}

    def import_data(self,
                    cell_metadata: Union[str, pd.DataFrame],
                    cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                    detected_transcripts: Union[str, pd.DataFrame, gpd.GeoDataFrame],
                    cell_by_gene_counts: Optional[Union[str, pd.DataFrame]]=None) -> None:
        """Read in the four pieces information needed for subsequent analysis: 
        cell-by-gene counts, cell metadata, cell boundary information, and transcript information. Some 
        basic visualization and cell clustering will be performed. 

        Parameters
        ----------
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
        cell_by_gene_counts : Optional[Union[str, pd.DataFrame]]
            Either the path to the csv file or a pandas dataframe containing the cell-by-gene count matrix whose first 
            column is assumed to contain the ID for each cell while the rest of the columns should be the transcript counts
            for each cell. 
        
        """
        print("Importing data... This could take a while. But you already know what we are dealing with...")
        self.adata, self.cell_coords, self.tx_metadata = import_data(cell_metadata=cell_metadata,
                                                                      cell_boundary_polygons=cell_boundary_polygons,
                                                                      detected_transcripts=detected_transcripts,
                                                                      cell_by_gene_counts=cell_by_gene_counts,
                                                                      **self.import_data_par)
        self.n_genes = self.adata.uns['n_genes']
        self.n_leiden = self.adata.uns['n_leiden']
        print("Computing mask distances... Keep in mind, patience is a virtue...")
        self.mask_distance = calculate_mask_distance(adata=self.adata,
                                                      cell_coords=self.cell_coords,
                                                      **self.calculate_mask_distance_par)
        print("Generating features for detected transcripts... Whatever, are we there yet...")
        self.intf_tx_dict = generate_feature(adata=self.adata,
                                            layer=self.current_layer,
                                            tx_metadata=self.tx_metadata,
                                            cell_coords=self.cell_coords,
                                            mask_distance=self.mask_distance,
                                            **self.generate_feature_par)
        
    def patchfy_data(self,
                     percent_cell_per_patch: float=0.1,
                     num_overlap: int=7) -> None:
        """Generate patches represented by their coordinates 

        Parameters
        ----------
        percent_cell_per_patch : float, optional
            _description_, by default 0.1
        num_overlap : int, optional
            _description_, by default 7
        """
        for intf_tx_name in self.intf_tx_dict:
            self.coord_list[intf_tx_name] = generate_patch_coords(adata=self.adata, 
                                                                intf_tx=self.intf_tx_dict[intf_tx_name],
                                                                percent_cell_per_patch=percent_cell_per_patch,
                                                                num_overlap=num_overlap) 
            if len(self.coord_list[intf_tx_name]) == 0:
                self.generate_feature_par['nearest'] -= 1
                del self.intf_tx_dict[intf_tx_name]
                del self.coord_list[intf_tx_name]
                del self.optimizer[intf_tx_name]
                del self.CEl_list[intf_tx_name]
                del self.KLD_list[intf_tx_name]
    
    def initialize_parameters(self) -> None:
        """Initialize model parameters 

        """
        for intf_tx_name in self.intf_tx_dict:
            self.distance_reassign_coefficients.update({intf_tx_name: nn.Linear(in_features=1, out_features=1, bias=True)})
            self.self_expression_reassign_coefficients.update({intf_tx_name: nn.Linear(in_features=1, out_features=1, bias=True)})
            self.neighbor_expression_reassign_coefficients.update({intf_tx_name: nn.Linear(in_features=1, out_features=1, bias=True)})
            if self.reparametrize:
                parametrize.register_parametrization(self.distance_reassign_coefficients[intf_tx_name], "weight", Positive())
                parametrize.register_parametrization(self.self_expression_reassign_coefficients[intf_tx_name], "weight", Positive())
                parametrize.register_parametrization(self.neighbor_expression_reassign_coefficients[intf_tx_name], "weight", Positive())
            
            alpha_0 = -np.log(1/self.prior_50_reassign_prob-1+1e-20)
            temp = -np.log(0.05/0.95)
            alpha_1 = (-np.log(1/self.prior_5_reassign_prob-1+1e-20) - alpha_0)/temp
            
            self.prior_distance_reassign_coefficients.update({intf_tx_name: nn.Linear(in_features=1, out_features=1, bias=True)})
            self.prior_distance_reassign_coefficients[intf_tx_name].bias = nn.Parameter(torch.tensor(alpha_0, dtype=torch.float32).reshape_as(self.prior_distance_reassign_coefficients[intf_tx_name].bias))
            self.prior_distance_reassign_coefficients[intf_tx_name].weight = nn.Parameter(torch.tensor(alpha_1, dtype=torch.float32).reshape_as(self.prior_distance_reassign_coefficients[intf_tx_name].weight))
            for param in self.prior_distance_reassign_coefficients[intf_tx_name].parameters():
                param.requires_grad = False
            
            self.prior_self_expression_reassign_coefficients.update({intf_tx_name: nn.Linear(in_features=1, out_features=1, bias=True)})
            self.prior_self_expression_reassign_coefficients[intf_tx_name].bias = nn.Parameter(torch.tensor(alpha_0, dtype=torch.float32).reshape_as(self.prior_self_expression_reassign_coefficients[intf_tx_name].bias))
            self.prior_self_expression_reassign_coefficients[intf_tx_name].weight = nn.Parameter(torch.tensor(alpha_1, dtype=torch.float32).reshape_as(self.prior_self_expression_reassign_coefficients[intf_tx_name].weight))
            for param in self.prior_self_expression_reassign_coefficients[intf_tx_name].parameters():
                param.requires_grad = False
                
            self.prior_neighbor_expression_reassign_coefficients.update({intf_tx_name: nn.Linear(in_features=1, out_features=1, bias=True)})
            self.prior_neighbor_expression_reassign_coefficients[intf_tx_name].bias = nn.Parameter(torch.tensor(alpha_0, dtype=torch.float32).reshape_as(self.prior_neighbor_expression_reassign_coefficients[intf_tx_name].bias))
            self.prior_neighbor_expression_reassign_coefficients[intf_tx_name].weight = nn.Parameter(torch.tensor(alpha_1, dtype=torch.float32).reshape_as(self.prior_neighbor_expression_reassign_coefficients[intf_tx_name].weight))
            for param in self.prior_neighbor_expression_reassign_coefficients[intf_tx_name].parameters():
                param.requires_grad = False
            
            # The cell type coefficients takes the gene counts and outputs the logits for all the cell types 
            self.cell_type_coefficients.update({intf_tx_name: nn.Linear(in_features=self.n_genes, 
                                                out_features=self.n_leiden,
                                                bias=True)}) 
            # Adam optimizer
            self.optimizer[intf_tx_name] = optim.Adam([{'params': self.distance_reassign_coefficients[intf_tx_name].parameters()},
                                                       {'params': self.self_expression_reassign_coefficients[intf_tx_name].parameters()},
                                                       {'params': self.neighbor_expression_reassign_coefficients[intf_tx_name].parameters()},
                                                       {'params': self.prior_distance_reassign_coefficients[intf_tx_name].parameters()},
                                                       {'params': self.prior_self_expression_reassign_coefficients[intf_tx_name].parameters()},
                                                       {'params': self.prior_neighbor_expression_reassign_coefficients[intf_tx_name].parameters()},
                                                       {'params': self.cell_type_coefficients[intf_tx_name].parameters()}], lr=1e-3)
        # Send to correct device 
        self.to(self.model_device)
    
    def encode(self, 
               tx_features: dict,
               temperature: float,
               intf_tx_name: str) -> Tuple[torch.tensor, torch.tensor]:
        """Compute posterior probability given the features 

        Parameters
        ----------
        tx_features : torch.tensor
            The transcript features after seeing the cell type information 
        temperature : float
            The temperature parameter for sampling from the Gumbel softmax distribution 

        Returns
        -------
        Tuple[torch.tensor, torch.tensor]
            Random sample from the gumbel softmax and the associated probabilities 
        """
        distance_reassign_logits = self.distance_reassign_coefficients[intf_tx_name](tx_features['distance_feature'])
        distance_reassign_probs = binary_gumbel_softmax_sample(logits=distance_reassign_logits,
                                                        temperature=temperature,
                                                        model_device=self.model_device)
        
        self_expression_reassign_logits = self.self_expression_reassign_coefficients[intf_tx_name](tx_features['exp_feature'])
        self_expression_reassign_probs = binary_gumbel_softmax_sample(logits=self_expression_reassign_logits,
                                                                    temperature=temperature,
                                                                    model_device=self.model_device)        
        
        neighbor_expression_reassign_logits = self.neighbor_expression_reassign_coefficients[intf_tx_name](tx_features['neighbor_exp_feature'])
        neighbor_expression_reassign_probs = binary_gumbel_softmax_sample(logits=neighbor_expression_reassign_logits,
                                                                        temperature=temperature,
                                                                        model_device=self.model_device)
        
        reassign_probs = distance_reassign_probs * self_expression_reassign_probs * neighbor_expression_reassign_probs
        reassign_hard = torch.round(reassign_probs, decimals=0)
        reassign_hard = (reassign_hard - reassign_probs).detach() + reassign_probs
        
        return reassign_hard,\
            distance_reassign_probs, self_expression_reassign_probs, neighbor_expression_reassign_probs
        
    def decode(self, 
               updated_cell_by_gene_counts: torch.tensor,
               tx_prior_features: dict,
               intf_tx_name: str) -> Tuple[torch.tensor, torch.tensor]:
        """Compute the likelihood and the prior 

        Parameters
        ----------
        updated_cell_by_gene_counts : torch.tensor
            Cell by gene counts matrix (after transcript reassignment)
        prior_distance_features : torch.tensor
            The transcript distance features before seeing the cell type information  

        Returns
        -------
        Tuple[torch.tensor, torch.tensor]
            Logits for different cell types and the prior reassigning probabilities 
        """
        cell_type_logits = self.cell_type_coefficients[intf_tx_name](updated_cell_by_gene_counts)
        prior_distance_reassign_logits = self.prior_distance_reassign_coefficients[intf_tx_name](tx_prior_features["prior_distance_feature"])
        prior_distance_reassign_probs = torch.sigmoid(prior_distance_reassign_logits)
        
        prior_self_expression_reassign_logits = self.prior_self_expression_reassign_coefficients[intf_tx_name](tx_prior_features["prior_exp_feature"])
        prior_self_expression_reassign_probs = torch.sigmoid(prior_self_expression_reassign_logits)
        
        prior_neighbor_expression_reassign_logits = self.prior_neighbor_expression_reassign_coefficients[intf_tx_name](tx_prior_features["prior_neighbor_exp_feature"])
        prior_neighbor_expression_reassign_probs = torch.sigmoid(prior_neighbor_expression_reassign_logits)
        
        return cell_type_logits,\
            prior_distance_reassign_probs, prior_self_expression_reassign_probs, prior_neighbor_expression_reassign_probs
    
    def forward(self,
                tx_features: dict, 
                tx_prior_features: dict,
                cell_by_gene_counts: torch.tensor, 
                row_index_self: torch.tensor,
                row_index_neighbor: torch.tensor,
                col_index: torch.tensor,
                temperature: float,
                intf_tx_name: str) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """The forward pass 

        Parameters
        ----------
        tx_features : torch.tensor
            The transcript features after seeing the cell type information 
        tx_prior_features : torch.tensor
            The transcript distance features before seeing the cell type information
        cell_by_gene_counts : torch.tensor
            Cell by gene counts matrix
        row_index_self : torch.tensor
            The row indices for cells to which the transcripts are currently assigned
        row_index_neighbor : torch.tensor
            The row indices for cells that are neighbors of the cells to which the transcripts are currently assigned
        col_index : torch.tensor
            The gene index 
        temperature : float
            The temperature parameter for sampling from the Gumbel softmax distribution 

        Returns
        -------
        Tuple[torch.tensor, torch.tensor, torch.tensor]
            Posterior reassignment probabilities, logits for different cell types, and the prior reassigning probabilities 
        """
        # Sample from the posterior 
        reassign_hard, distance_reassign_probs, self_expression_reassign_probs, neighbor_expression_reassign_probs = self.encode(tx_features=tx_features,
                                                                                                                                temperature=temperature,
                                                                                                                                intf_tx_name=intf_tx_name)
        # As methods like pivot, and groupby do not exist for tensors 
        # what we are doing here is vectorizing the indices 
        index_self = row_index_self * self.adata.uns['n_genes'] + col_index
        index_neighbor = row_index_neighbor * self.adata.uns['n_genes'] + col_index
        # And perform cumulative sum
        update_patch_self = torch.zeros((cell_by_gene_counts.shape[0])*(self.adata.uns['n_genes']), 
                                        1, dtype=torch.float32, device=self.model_device).scatter_add(0, index_self, reassign_hard)
        update_patch_neighbor = torch.zeros((cell_by_gene_counts.shape[0])*(self.adata.uns['n_genes']), 
                                            1, dtype=torch.float32, device=self.model_device).scatter_add(0, index_neighbor, reassign_hard)
        # Then we unvectorize the indices 
        # row would be like 0 0 1 1
        # col would be like 0 1 0 1
        ordered_row_index = np.repeat(np.arange(cell_by_gene_counts.shape[0]), self.adata.uns['n_genes'])
        ordered_col_index = np.tile(np.arange(self.adata.uns['n_genes']), cell_by_gene_counts.shape[0])
        # Now we update the count matrix 
        cell_by_gene_counts[ordered_row_index, ordered_col_index] -= update_patch_self.squeeze(1)
        cell_by_gene_counts[ordered_row_index, ordered_col_index] += update_patch_neighbor.squeeze(1)
        # Compute the likelihood and prior 
        cell_type_logits, prior_distance_reassign_probs, prior_self_expression_reassign_probs, prior_neighbor_expression_reassign_probs = self.decode(updated_cell_by_gene_counts=cell_by_gene_counts,
                                                                                                                                                    tx_prior_features=tx_prior_features,
                                                                                                                                                    intf_tx_name=intf_tx_name)
        
        return distance_reassign_probs, self_expression_reassign_probs, neighbor_expression_reassign_probs,\
            cell_type_logits, prior_distance_reassign_probs, prior_self_expression_reassign_probs, prior_neighbor_expression_reassign_probs
    
    def loss_function(self,
                      cell_type_logits: torch.tensor, 
                      cell_type_labels: torch.tensor,
                      distance_reassign_probs: torch.tensor,
                      self_expression_reassign_probs: torch.tensor,
                      neighbor_expression_reassign_probs: torch.tensor,
                      prior_distance_reassign_probs: torch.tensor,
                      prior_self_expression_reassign_probs: torch.tensor,
                      prior_neighbor_expression_reassign_probs: torch.tensor) -> torch.tensor:
        """The loss function 

        Parameters
        ----------
        cell_type_logits : torch.tensor
            Logits for different cell types
        cell_type_labels : torch.tensor
            The actual labels 
        reassign_probs : torch.tensor
            Posterior reassignment probabilities
        prior_reassign_probs : torch.tensor
            Prior reassignment probabilities

        Returns
        -------
        torch.tensor
            The loss 
        """
        # Cross entropy loss (despite its name it's a loss)
        CEl = F.cross_entropy(cell_type_logits,
                                cell_type_labels, reduction='mean')
        # KL divergence for distance 
        distance_log_ratio_1 = torch.log(distance_reassign_probs/prior_distance_reassign_probs+1e-20)
        distance_log_ratio_0 = torch.log((1-distance_reassign_probs)/(1-prior_distance_reassign_probs)+1e-20)
        distance_KLD_reassign = torch.sum(distance_reassign_probs * distance_log_ratio_1 + (1-distance_reassign_probs) * distance_log_ratio_0, dim=-1).mean()
        # KL divergence for self expression 
        self_expression_log_ratio_1 = torch.log(self_expression_reassign_probs/prior_self_expression_reassign_probs+1e-20)
        self_expression_log_ratio_0 = torch.log((1-self_expression_reassign_probs)/(1-prior_self_expression_reassign_probs)+1e-20)
        self_expression_KLD_reassign = torch.sum(self_expression_reassign_probs * self_expression_log_ratio_1 + (1-self_expression_reassign_probs) * self_expression_log_ratio_0, dim=-1).mean()
        # KL divergence for neigh expression 
        neighbor_expression_log_ratio_1 = torch.log(neighbor_expression_reassign_probs/prior_neighbor_expression_reassign_probs+1e-20)
        neighbor_expression_log_ratio_0 = torch.log((1-neighbor_expression_reassign_probs)/(1-prior_neighbor_expression_reassign_probs)+1e-20)
        neighbor_expression_KLD_reassign = torch.sum(neighbor_expression_reassign_probs * neighbor_expression_log_ratio_1 + (1-neighbor_expression_reassign_probs) * neighbor_expression_log_ratio_0, dim=-1).mean()
        
        return CEl + distance_KLD_reassign + self_expression_KLD_reassign + neighbor_expression_KLD_reassign,\
            CEl.detach().cpu().numpy().item(),\
            distance_KLD_reassign.detach().cpu().numpy().item(),\
            self_expression_KLD_reassign.detach().cpu().numpy().item(),\
            neighbor_expression_KLD_reassign.detach().cpu().numpy().item()
    
    def training_loop(self,
                      n_epochs: int,
                      early_stop_pval: float=0.5) -> None:
        """The training loop 

        Parameters
        ----------
        n_epochs : int
            Number of epochs 
        """
        for intf_tx_name in self.intf_tx_dict:
            # Preconstruct the information to save time 
            adata_w_leiden_xy = self.adata.to_df(self.current_layer).merge(self.adata.obs[['leiden','x','y']],
                                                                            how='left',
                                                                            left_index=True,
                                                                            right_index=True)
            adata_w_leiden_xy = pl.from_pandas(adata_w_leiden_xy, include_index=True)
            adata_var = pl.from_pandas(self.adata.var, include_index=True)
            
            self.train()
            for epoch in range(n_epochs):
                if epoch >= 2:
                    cel_previous_2 = np.array(self.CEl_list[intf_tx_name][epoch-2])
                    cel_previous_1 = np.array(self.CEl_list[intf_tx_name][epoch-1])
                    p_val = ks_2samp(cel_previous_2, cel_previous_1).pvalue
                    if p_val > early_stop_pval:
                        print("="*30)
                        print("No improvement in the classification task detected. Stop early at epoch {}".format(epoch-1))
                        print("="*30)
                        break
                np.random.shuffle(self.coord_list[intf_tx_name])
                self.current_temperature = self.init_temperature
                train_loss = 0.0
                CEl_epoch_list = []
                KLD_epoch_list = {"distance": [],
                                  "self_expression": [],
                                  "neighbor_expression": []}
                for minibatch_ind, coord in tqdm(enumerate(self.coord_list[intf_tx_name]),
                                                total=len(self.coord_list[intf_tx_name]),
                                                desc=intf_tx_name):
                    # Load data 
                    cell_by_gene_counts, tx_features, tx_prior_features, cell_type_labels, row_index_self, row_index_neighbor, col_index = load_patch(adata_w_leiden_xy=adata_w_leiden_xy,
                                                                                                                                                    adata_var=adata_var,
                                                                                                                                                    intf_tx=self.intf_tx_dict[intf_tx_name],
                                                                                                                                                    coord_tuple=coord,
                                                                                                                                                    model_device=self.model_device)
                    # Compute likelihood, prior, and posterior 
                    distance_reassign_probs, self_expression_reassign_probs, neighbor_expression_reassign_probs,\
                        cell_type_logits, prior_distance_reassign_probs, prior_self_expression_reassign_probs, prior_neighbor_expression_reassign_probs = self(tx_features=tx_features, 
                                                                                                                                                            tx_prior_features=tx_prior_features,                
                                                                                                                                                            cell_by_gene_counts=cell_by_gene_counts, 
                                                                                                                                                            row_index_self=row_index_self,
                                                                                                                                                            row_index_neighbor=row_index_neighbor,
                                                                                                                                                            col_index=col_index,
                                                                                                                                                            temperature=self.current_temperature,
                                                                                                                                                            intf_tx_name=intf_tx_name)
                    # Loss and gradient 
                    self.optimizer[intf_tx_name].zero_grad()
                    loss, CEl, distance_KLD, self_expression_KLD, neighbor_expression_KLD = self.loss_function(cell_type_logits=cell_type_logits,
                                                                                                    cell_type_labels=cell_type_labels,
                                                                                                    distance_reassign_probs=distance_reassign_probs,
                                                                                                    self_expression_reassign_probs=self_expression_reassign_probs,
                                                                                                    neighbor_expression_reassign_probs=neighbor_expression_reassign_probs,
                                                                                                    prior_distance_reassign_probs=prior_distance_reassign_probs,
                                                                                                    prior_self_expression_reassign_probs=prior_self_expression_reassign_probs,
                                                                                                    prior_neighbor_expression_reassign_probs=prior_neighbor_expression_reassign_probs)
                    loss.backward()
                    CEl_epoch_list.append(CEl)
                    KLD_epoch_list['distance'].append(distance_KLD)
                    KLD_epoch_list['self_expression'].append(self_expression_KLD)
                    KLD_epoch_list["neighbor_expression"].append(neighbor_expression_KLD)
                    train_loss += loss.item()
                    self.optimizer[intf_tx_name].step()
                    # We gradually decrease the temperature so that the posterior would approach a bernoulli 
                    if minibatch_ind % 100 == 1:
                        self.current_temperature = np.maximum(self.current_temperature * np.exp(-self.ANNEAL_RATE * minibatch_ind), self.min_temperature) 
                    # Print training information 
                    if minibatch_ind % self.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                                epoch, minibatch_ind, len(self.coord_list[intf_tx_name]),
                                    100. * minibatch_ind / len(self.coord_list[intf_tx_name]),
                                    loss.item()))
                self.CEl_list[intf_tx_name].append(CEl_epoch_list)
                self.KLD_list[intf_tx_name]['distance'].append(KLD_epoch_list['distance'])
                self.KLD_list[intf_tx_name]['self_expression'].append(KLD_epoch_list['self_expression'])
                self.KLD_list[intf_tx_name]['neighbor_expression'].append(KLD_epoch_list['neighbor_expression'])
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                        epoch, train_loss / len(self.coord_list[intf_tx_name])))

    def compute_reassign_probs(self) -> None:
        """Compute reassignment probabilities
        """
        for intf_tx_name, intf_tx in self.intf_tx_dict.items():
            self.eval()
            with torch.no_grad():
                # Split the matrix into chunks in case the gpu memory is small
                tx_features_chunks = even_split(array=intf_tx[['distance_feature', 
                                                "exp_feature", 
                                                "neighbor_exp_feature"]].to_numpy(),
                                                chunk_size=np.ceil(intf_tx.shape[0]/100))
                reassign_probs = np.array([], dtype=float).reshape(0,1)
                for tx_features_chunk in tqdm(tx_features_chunks):
                    tx_features = {"distance_feature": torch.tensor(tx_features_chunk[:,[0]], 
                                                    dtype=torch.float32, device=self.model_device),
                                    "exp_feature": torch.tensor(tx_features_chunk[:,[1]], 
                                                                        dtype=torch.float32, device=self.model_device),
                                    "neighbor_exp_feature": torch.tensor(tx_features_chunk[:,[2]], 
                                                                        dtype=torch.float32, device=self.model_device)}
                    _, distance_reassign_probs_chunk, self_expression_reassign_probs_chunk, neighbor_expression_reassign_probs_chunk = self.encode(tx_features=tx_features,
                                                                                                                                                temperature=self.min_temperature,
                                                                                                                                                intf_tx_name=intf_tx_name)
                    reassign_probs_chunk = distance_reassign_probs_chunk.cpu().numpy() *\
                                        self_expression_reassign_probs_chunk.cpu().numpy() *\
                                        neighbor_expression_reassign_probs_chunk.cpu().numpy()
                    reassign_probs = np.vstack([reassign_probs, reassign_probs_chunk])
                # Store the computed probabilities 
                self.intf_tx_dict[intf_tx_name] = intf_tx.with_columns(pl.Series(name="reassign_probs", values=reassign_probs.squeeze(-1)))
        
        prob_cols = ["reassign_probs0"]
        neighbor_cols = ['neighbor_cell_id0']
        self.intf_tx = self.intf_tx_dict['intf_tx0'].select(['molecule_id', "cell_id", "neighbor_cell_id", "gene","reassign_probs"])
        self.intf_tx = self.intf_tx.rename({"neighbor_cell_id":"neighbor_cell_id0",
                                            "reassign_probs":"reassign_probs0"})
        if len(self.intf_tx_dict) > 1:
            for i in range(1, len(self.intf_tx_dict)):
                prob_cols.append("reassign_probs{}".format(i))
                neighbor_cols.append("neighbor_cell_id{}".format(i))
                self.intf_tx = self.intf_tx.join(self.intf_tx_dict['intf_tx{}'.format(i)].select(['molecule_id',"neighbor_cell_id", "reassign_probs"]),
                                                how='left', on='molecule_id')
                self.intf_tx = self.intf_tx.with_columns(pl.col("neighbor_cell_id").fill_null(""),
                                                        pl.col("reassign_probs").fill_null(0))
                self.intf_tx = self.intf_tx.rename({"neighbor_cell_id":"neighbor_cell_id{}".format(i),
                                                    "reassign_probs":"reassign_probs{}".format(i)})
        self.intf_tx = self.intf_tx.with_columns(pl.concat_list(prob_cols).list.arg_max().alias("max_index"))
        self.intf_tx = self.intf_tx.with_columns(pl.concat_list(neighbor_cols).list.get(pl.col("max_index")).alias("neighbor_cell_id"))
        self.intf_tx = self.intf_tx.with_columns(pl.concat_list(prob_cols).list.get(pl.col("max_index")).alias("reassign_probs"))
        self.intf_tx = self.intf_tx.drop(neighbor_cols+prob_cols)
        
    def reassign_tx(self,
                    criteria: dict={"threshold": 0.5}) -> None:
        """Generate transcript reassignment based on various criteria 

        Parameters
        ----------
        criteria : dict, optional
            A dictionary of criterion name-criterion pair. For random assignment, the criterion should be a string, by default {"threshold": 0.5, "random": "random"}
        """
        for criterion_name in tqdm(criteria): 
            criterion = criteria[criterion_name]
            if isinstance(criterion, str):
                reassign = np.random.binomial(n=1, p=self.intf_tx['reassign_probs'])
            else:
                reassign = (self.intf_tx['reassign_probs']>criterion).to_numpy().astype(int)
            
            self.intf_tx = self.intf_tx.with_columns(pl.Series(name='reassign', values=reassign))
            tx_to_reassign = self.intf_tx.filter(pl.col("reassign")==1).drop("reassign").to_pandas()
            
            self.intf_tx = self.intf_tx.drop(['reassign'])
            trial_layer = self.current_layer+"_"+criterion_name
            # For each criterion, we store the update 
            # And make assignment to the count matrix 
            # This will also compute UMAP by default
            # We do not actually reassign transcript at this stage 
            self.tx_to_reassign_dict[trial_layer] = tx_to_reassign.copy()
            self.adata = make_reassignment_adata(adata=self.adata,
                                                layer=self.current_layer,
                                                tx_to_reassign=tx_to_reassign,
                                                trial_layer=trial_layer,
                                                preprocess=self.import_data_par['preprocess'])
        
    def save_model(self,
                   dir_name: str,
                   model_name: str,
                   save_reassigning_result: bool=False,
                   selected_criterion: Optional[str]=None) -> None:
        """Save the model

        Parameters
        ----------
        path : str
            The path to the torch model. It should end with .pt 
            Other pieces of information will be saved in the same directory 
        """
        if save_reassigning_result:
            assert selected_criterion in self.tx_to_reassign_dict, "selected_criterion not found in tx_to_reassign_dict"
        
        torch.save({'model_state_dict': self.state_dict()} | \
            {'optimizer_state_dict_{}'.format(intf_tx_name): self.optimizer[intf_tx_name].state_dict() for intf_tx_name in self.intf_tx_dict}, 
            os.path.join(dir_name, model_name+".pt"))
        
        model_meta = {'import_data_par': self.import_data_par,
                      'calculate_mask_distance_par': self.calculate_mask_distance_par,
                      'generate_feature_par': self.generate_feature_par,
                      'n_genes': self.n_genes,
                      'n_leiden': self.n_leiden,
                      'prior_50_reassign_prob': self.prior_50_reassign_prob,
                      'prior_5_reassign_prob': self.prior_5_reassign_prob,
                      'reparametrize': self.reparametrize,
                      'current_layer': self.current_layer,
                      "save_reassigning_result": save_reassigning_result,
                      "selected_criterion": selected_criterion}
        
        with open(os.path.join(dir_name, model_name+"_meta.json"), "w") as f:
            json.dump(model_meta, f, cls=JSONEncoder)
            
        if save_reassigning_result:
            tx_to_reassign = self.tx_to_reassign_dict[self.current_layer+"_"+selected_criterion].copy()
            tx_to_reassign.rename(columns={"cell_id": "original_cell_id",
                                           "neighbor_cell_id": "reassigned_cell_id"},
                                  inplace=True)
            tx_to_reassign.to_parquet(os.path.join(dir_name, model_name+"_tx_to_reassign.parquet"))
            # with open(os.path.join(dir_name, model_name+"_tx_to_reassign_dict.json"), "w") as f:
            #     json.dump(self.tx_to_reassign_dict, f, cls=JSONEncoder)
        
    def load_model(self,
                   dir_name: str,
                   model_name: str) -> None:
        """Load the model

        Parameters
        ----------
        path : str
            The path to the torch model. It should end with .pt 
            Other pieces of information will be saved in the same directory 
        """
        
        model_meta = json.load(open(os.path.join(dir_name, model_name+"_meta.json")))
        self.import_data_par = model_meta['import_data_par']
        self.calculate_mask_distance_par = model_meta['calculate_mask_distance_par']
        self.generate_feature_par = model_meta['generate_feature_par']
        self.n_genes = model_meta['n_genes']
        self.n_leiden = model_meta['n_leiden']
        self.current_layer = model_meta['current_layer']
        self.prior_50_reassign_prob = model_meta['prior_50_reassign_prob']
        self.prior_5_reassign_prob = model_meta['prior_5_reassign_prob']
        self.reparametrize = model_meta['reparametrize']
        
        save_reassigning_result = model_meta['save_reassigning_result']
        selected_criterion = model_meta['selected_criterion']
        
        if save_reassigning_result:
            tx_to_reassign = pd.read_parquet(os.path.join(dir_name, model_name+"_tx_to_reassign.parquet"))
            tx_to_reassign.rename(columns={"original_cell_id": "cell_id",
                                           "reassigned_cell_id": "neighbor_cell_id"},
                                  inplace=True)
            self.tx_to_reassign_dict = {self.current_layer+"_"+selected_criterion: tx_to_reassign}
            # self.tx_to_reassign_dict = json.load(open(os.path.join(dir_name, model_name+"_tx_to_reassign_dict.json")))
            # for k in self.tx_to_reassign_dict:
            #     self.tx_to_reassign_dict[k] = pd.read_json(self.tx_to_reassign_dict[k])
        
        self.initialize_parameters()
        checkpoint = torch.load(os.path.join(dir_name, model_name+".pt")) 
        self.load_state_dict(checkpoint['model_state_dict'])
        for i in range(self.generate_feature_par['nearest']):
            intf_tx_name = "intf_tx{}".format(i)
            self.optimizer[intf_tx_name].load_state_dict(checkpoint['optimizer_state_dict_{}'.format(intf_tx_name)])
        
        
            
    