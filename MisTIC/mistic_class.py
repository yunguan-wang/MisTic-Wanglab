# Data IO
import os
import json 
import pandas as pd
import scanpy as sc
import geopandas as gpd
import polars as pl 
# Data manipulation 
import numpy as np
from scipy.special import softmax
from scipy.stats import ks_2samp, entropy
import torch 
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
# Utility function 
from MisTIC.utility import import_data, calculate_mask_distance, binary_gumbel_softmax_sample,\
        make_reassignment_adata, calibrate_threshold, compute_gene_threshold, generate_count_patches,\
            diagLinear, Positive, JSONEncoder, even_split
from MisTIC.generate_tx_feature import generate_feature
from MisTIC.data_loader import generate_patch_coords, load_patch
# User entertainment
from tqdm.auto import tqdm
# Typing 
from typing import Union, Optional, Tuple 


class mistic(nn.Module):
    def __init__(self,
                cell_centroid_x_col: str='center_x',
                cell_centroid_y_col: str='center_y',
                celltype_col: Optional[str]=None,
                tx_x_col: str='global_x',
                tx_y_col: str='global_y',
                gene_col: str='gene',
                cell_col: str='cell_id',
                leiden_res: float=1,
                dr_method: str='umap',
                max_centroid_dist: float=50,
                mask_dist_cutoff: float=5,
                nearest: int=3,
                prior_50_reassign_prob: float=0.01,
                prior_5_reassign_prob: float=0.99,
                seed: int=42,
                max_de_cells: int=100000,
                model_device: Optional[Union[str, torch.device]] = None) -> None:
        """Instantiate a mistic object 

        Parameters
        ----------
        cell_centroid_x_col : str, optional
            The column containing the x coordinates of cell centroids in cell metadata file, by default 'center_x'
        cell_centroid_y_col : str, optional
            The column containing the y coordinates of cell centroids in cell metadata file, by default 'center_y'
        celltype_col : Optional[str], optional
            The column containing the cell type information in cell metadata file, by default None
        tx_x_col : str, optional
            The column containing the x coordinates of transcript in detected transcript file, by default 'global_x'
        tx_y_col : str, optional
            The column containing the y coordinates of transcript in detected transcript file, by default 'global_y'
        gene_col : str, optional
            The column containing the gene information of transcript in detected transcript file, by default 'gene'
        cell_col : str, optional
            The column containing the cell id of transcript in detected transcript file, by default 'cell_id'
        leiden_res : float, optional
            The resolution for leiden clustering, by default 1
        dr_method : str, optional
            The dimension reduction method to be used. Either pca or umap, by default "umap"
        max_centroid_dist : int, optional
            The threshold on cell-cell centroid distances beyond which we do not consider two cells being neighbors, by default 50
        mask_dist_cutoff : float, optional
            The threshold of cell-cell distances beyond which a cell is no longer considered a neighbor, by default 5
        prior_50_reassign_prob : float, optional
            The prior probability of reassigning a transcript that's ranked 50%, by default 0.01
        prior_5_reassign_prob : float, optional
            The prior probability of reassigning a transcript that's ranked 5%, by default 0.99
        seed: int, optional 
            For reproducibitlity, by default 42
        model_device : Optional[Union[str, torch.device]], optional
            The device to use 
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
            'dr_method': dr_method
        }
        self.calculate_mask_distance_par = {
            'max_centroid_dist': max_centroid_dist,
        }
        self.generate_feature_par = {
            'mask_dist_cutoff': mask_dist_cutoff,
            'nearest': nearest,
            'seed': seed,
            'max_de_cells': max_de_cells
        }
        
        # Create data 
        self.adata = None
        self.cell_coords = None
        self.tx_metadata = None
        self.current_layer = "counts_0"
        self.mask_distance = None
        # intf_tx will contain info for all neighbors 
        self.intf_tx = None
        # tx_reassign_info aggregates all neighbor info
        self.tx_reassign_info = None
        # For each neighbor, we generate patches 
        self.coord_list = {"neighbor{}".format(i): [] for i in range(nearest)}
        self.tx_to_reassign = None
        self.tx_to_remove = None
        self.criterion_df = None
        
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
        self.calibrator = None
        
        self.reassign_coefficients = None
        self.prior_reassign_coefficients = None

        self.cell_type_coefficients = None
        self.optimizer = None
        # For training 
        self.CEl_list = []
        self.KLD_list = []

    def import_data(self,
                    cell_metadata: Union[str, pd.DataFrame],
                    cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                    detected_transcripts: Union[str, pd.DataFrame, gpd.GeoDataFrame],
                    cell_by_gene_counts: Optional[Union[str, pd.DataFrame]]=None) -> None:
        """Read in the three pieces information needed for subsequent analysis: 
        cell metadata, cell boundary information, transcript information, and 
        optionally cell-by-gene matrix. Some data curation will be performed 

        Parameters
        ----------
        cell_metadata : Union[str, pd.DataFrame]
            Either the path to the csv file or a pandas dataframe containing the metadata for each cell whose 
            first column is assumed to contain the ID for each cell. The information in the file/object should 
            at least contain the xy coordinates of the centers of cells.
            If the users have already performed cell typing which is not required,
            the information can be stored here as a separate column.
        cell_boundary_polygons : Union[str, gpd.GeoDataFrame]
            Either the path to the parquet file or the geopandas GeoDataFrame containing the vertex information 
            for each cell. The first column is assumed to be the IDs for cells. It should contain one column 
            that records the coordinates of the vertices.
        detected_transcripts : Union[str, pd.DataFrame, gpd.GeoDataFrame]
            Either the path to the csv file or a pandas dataframe or geopandas GeoDataFrame containing the information of the detected transcripts.
            The first column is assumed to be some index. The information should contain the ID for a transcripte/molecule, 
            the ID of the cell it belongs to, its xy coordinate, and its gene information. 
        cell_by_gene_counts : Optional[Union[str, pd.DataFrame]], optional
            Either the path to the csv file or a pandas dataframe containing the cell-by-gene count matrix whose first 
            column is assumed to contain the ID for each cell while the rest of the columns should be the transcript counts
            for each cell. If this is not provided, the cell-by-gene matrix will be constructed from the detected_transcripts.
            If provided, the users are responsible for ensuring that the counts correspond to what's recorded in detected_transcripts , by default None
            
        """
        # Import data 
        self.adata, self.cell_coords, self.tx_metadata = import_data(cell_metadata=cell_metadata,
                                                                      cell_boundary_polygons=cell_boundary_polygons,
                                                                      detected_transcripts=detected_transcripts,
                                                                      cell_by_gene_counts=cell_by_gene_counts,
                                                                      **self.import_data_par)
        self.n_genes = self.adata.uns['n_genes']
        self.n_leiden = self.adata.uns['n_leiden']
        # Compute cell-cell mask distances
        self.mask_distance = calculate_mask_distance(adata=self.adata,
                                                      cell_coords=self.cell_coords,
                                                      **self.calculate_mask_distance_par)
        # Generate features 
        self.intf_tx = generate_feature(adata=self.adata,
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
            Rough percentage of cell per patch, by default 0.1
        num_overlap : int, optional
            Akin to stride, by default 7
        """
        for neighbor_index in range(self.generate_feature_par['nearest']):
            key = "neighbor"+str(neighbor_index)
            self.coord_list[key] = generate_patch_coords(adata=self.adata, 
                                                        intf_tx=self.intf_tx,
                                                        percent_cell_per_patch=percent_cell_per_patch,
                                                        num_overlap=num_overlap,
                                                        neighbor_index=neighbor_index) 
            if len(self.coord_list[key]) == 0:
                self.generate_feature_par['nearest'] -= 1
                del self.coord_list[key]
    
    def initialize_parameters(self) -> None:
        """Initialize model parameters 

        """
        # Based on the prior info, we first compute the coefficients 
        alpha_0 = -np.log(1/self.prior_50_reassign_prob-1+1e-20)
        temp = -np.log(0.05/0.95) 
        alpha_1 = (-np.log(1/self.prior_5_reassign_prob-1+1e-20) - alpha_0)/temp
        # And the calibration function
        self.calibrator = calibrate_threshold(alpha_0=alpha_0,
                                              alpha_1=alpha_1)
        # For the prior, no gradients 
        self.prior_reassign_coefficients = diagLinear(features=3, bias=True)
        self.prior_reassign_coefficients.bias = nn.Parameter(torch.tensor([alpha_0]*3, dtype=torch.float32).reshape_as(self.prior_reassign_coefficients.bias))
        self.prior_reassign_coefficients.weight = nn.Parameter(torch.tensor([alpha_1]*3, dtype=torch.float32).reshape_as(self.prior_reassign_coefficients.weight))
        for param in self.prior_reassign_coefficients.parameters():
            param.requires_grad = False
        # The posterior, we reparametrize
        self.reassign_coefficients = diagLinear(features=3, bias=True,
                                                initial_weights=[np.log(np.exp(alpha_1)-1)]*3,
                                                initial_bias=[alpha_0]*3)
        parametrize.register_parametrization(self.reassign_coefficients, "weight", Positive())
        # The cell type coefficients takes the gene counts and outputs the logits for all the cell types 
        self.cell_type_coefficients = nn.Linear(in_features=self.n_genes, 
                                            out_features=self.n_leiden,
                                            bias=True)
        # Adam optimizer
        self.optimizer = optim.Adam([{'params': self.reassign_coefficients.parameters()},
                                    {'params': self.prior_reassign_coefficients.parameters()},
                                    {'params': self.cell_type_coefficients.parameters()}], lr=1e-3)
        # Send to correct device 
        self.to(self.model_device)
    
    def encode(self, 
               tx_features: torch.tensor,
               temperature: float) -> Tuple[torch.tensor, torch.tensor]:
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
        reassign_logits = self.reassign_coefficients(tx_features)
        reassign_probs = torch.sigmoid(reassign_logits)
        individual_reassign_hard = binary_gumbel_softmax_sample(logits=reassign_logits,
                                                        temperature=temperature,
                                                        model_device=self.model_device,
                                                        hard=True)
        # The final reassignment is a product of all three
        reassign_hard = torch.prod(individual_reassign_hard, dim=1, keepdim=True)
        
        return reassign_hard, reassign_probs
        
    def decode(self, 
               updated_cell_by_gene_counts: torch.tensor,
               tx_prior_features: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """Compute the likelihood and the prior 

        Parameters
        ----------
        updated_cell_by_gene_counts : torch.tensor
            Cell by gene counts matrix (after transcript reassignment)
        tx_prior_features : torch.tensor
            The transcript features before seeing the cell type information  

        Returns
        -------
        Tuple[torch.tensor, torch.tensor]
            Logits for different cell types and the prior reassigning probabilities 
        """
        # Likelihood 
        cell_type_logits = self.cell_type_coefficients(updated_cell_by_gene_counts)
        # Prior
        prior_reassign_logits = self.prior_reassign_coefficients(tx_prior_features)
        prior_reassign_probs = torch.sigmoid(prior_reassign_logits)
        
        return cell_type_logits, prior_reassign_probs
    
    def forward(self,
                tx_features: dict, 
                tx_prior_features: dict,
                cell_by_gene_counts: torch.tensor, 
                row_index_self: torch.tensor,
                row_index_neighbor: torch.tensor,
                col_index: torch.tensor,
                temperature: float) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
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
        reassign_hard, reassign_probs = self.encode(tx_features=tx_features,
                                                    temperature=temperature)
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
        cell_type_logits, prior_reassign_probs= self.decode(updated_cell_by_gene_counts=cell_by_gene_counts,
                                                            tx_prior_features=tx_prior_features)
        
        return reassign_probs, cell_type_logits, prior_reassign_probs
    
    def loss_function(self,
                      cell_type_logits: torch.tensor, 
                      cell_type_labels: torch.tensor,
                      reassign_probs: torch.tensor,
                      prior_reassign_probs: torch.tensor) -> Tuple[torch.tensor, np.array, np.array]:
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
            The loss, the cross entropy loss, the KLs 
        """
        # Cross entropy loss (despite its name it's a loss)
        CEl = F.cross_entropy(cell_type_logits,
                                cell_type_labels, reduction='mean')
        # KL divergence 
        log_ratio_1 = torch.log(reassign_probs/(prior_reassign_probs+1e-20)+1e-20)
        log_ratio_0 = torch.log((1-reassign_probs)/(1-prior_reassign_probs+1e-20)+1e-20)
        KLD_reassign = torch.mean(reassign_probs * log_ratio_1 + (1-reassign_probs) * log_ratio_0, dim=0)
        
        return CEl + KLD_reassign.sum(), CEl.detach().cpu().numpy().item(), KLD_reassign.detach().cpu().numpy()
    
    def training_loop(self,
                      n_epochs: int,
                      early_stop_pval: float=0.5) -> None:
        """The training loop 

        Parameters
        ----------
        n_epochs : int
            Number of epochs 
        early_stop_pval: float, optional 
            The p-value for ks test 
        """
        
        # Preconstruct the information to save time 
        adata_w_leiden_xy = self.adata.to_df(self.current_layer).merge(self.adata.obs[['leiden','x','y']],
                                                                        how='left',
                                                                        left_index=True,
                                                                        right_index=True)
        adata_w_leiden_xy = pl.from_pandas(adata_w_leiden_xy, include_index=True)
        adata_var = pl.from_pandas(self.adata.var, include_index=True)
        
        self.train()
        for epoch in range(n_epochs):
            self.current_temperature = self.init_temperature
            CEl_epoch_list = []
            KLD_epoch_list = []
            # Starting from the 3rd epoch, we test if convergence has been achieved 
            if epoch >= 2:
                cel_previous_2 = np.array(self.CEl_list[epoch-2])
                cel_previous_1 = np.array(self.CEl_list[epoch-1])
                p_val = ks_2samp(cel_previous_2, cel_previous_1).pvalue
                if p_val > early_stop_pval:
                    print("="*30)
                    print("No improvement in the classification task detected. Stop early at epoch {}".format(epoch-1))
                    print("="*30)
                    break
            # Shuffle the neighbor indices 
            neighbor_indices = list(range(self.generate_feature_par['nearest']))
            np.random.shuffle(neighbor_indices)
            for neighbor_index in neighbor_indices:
                key = "neighbor"+str(neighbor_index)
                # Shuffle the coordinates 
                np.random.shuffle(self.coord_list[key])
                sub_intf_tx = self.intf_tx.filter(pl.col("neighbor_index") == neighbor_index)
                train_loss = 0.0
                for minibatch_ind, coord in tqdm(enumerate(self.coord_list[key]),
                                                total=len(self.coord_list[key]),
                                                desc=key):
                    # Load data 
                    cell_by_gene_counts, tx_features, tx_prior_features, cell_type_labels, row_index_self, row_index_neighbor, col_index = load_patch(adata_w_leiden_xy=adata_w_leiden_xy,
                                                                                                                                                    adata_var=adata_var,
                                                                                                                                                    intf_tx=sub_intf_tx,
                                                                                                                                                    coord_tuple=coord,
                                                                                                                                                    model_device=self.model_device)
                    # Compute likelihood, prior, and posterior 
                    reassign_probs, cell_type_logits, prior_reassign_probs= self(tx_features=tx_features, 
                                                                                tx_prior_features=tx_prior_features,                
                                                                                cell_by_gene_counts=cell_by_gene_counts, 
                                                                                row_index_self=row_index_self,
                                                                                row_index_neighbor=row_index_neighbor,
                                                                                col_index=col_index,
                                                                                temperature=self.current_temperature)
                    # Loss and gradient 
                    self.optimizer.zero_grad()
                    loss, CEl, KLD = self.loss_function(cell_type_logits=cell_type_logits,
                                                        cell_type_labels=cell_type_labels,
                                                        reassign_probs=reassign_probs,
                                                        prior_reassign_probs=prior_reassign_probs)
                    loss.backward()
                    CEl_epoch_list.append(CEl)
                    KLD_epoch_list.append(KLD)
                    train_loss += loss.item()
                    self.optimizer.step()
                    # We gradually decrease the temperature so that the posterior would approach a bernoulli 
                    if minibatch_ind % 100 == 1:
                        self.current_temperature = np.maximum(self.current_temperature * np.exp(-self.ANNEAL_RATE * minibatch_ind), self.min_temperature) 
                    # Print training information 
                    if minibatch_ind % self.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                                epoch, minibatch_ind, len(self.coord_list[key]),
                                    100. * minibatch_ind / len(self.coord_list[key]),
                                    loss.item()))
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                        epoch, train_loss / len(self.coord_list[key])))
            self.CEl_list.append(np.array(CEl_epoch_list))
            self.KLD_list.append(np.array(KLD_epoch_list))
                
    def compute_reassign_probs(self) -> None:
        """Compute reassignment probabilities
        """
        
        self.eval()
        with torch.no_grad():
            # Split the matrix into chunks in case the gpu memory is small
            tx_features_chunks = even_split(array=self.intf_tx[['distance_feature', 
                                            "exp_feature", 
                                            "neighbor_exp_feature"]].to_numpy(),
                                            chunk_size=np.ceil(self.intf_tx.shape[0]/100))
            reassign_probs = np.array([], dtype=float).reshape(0,1)
            for tx_features_chunk in tqdm(tx_features_chunks):
                tx_features = torch.tensor(tx_features_chunk, dtype=torch.float32, device=self.model_device)
                _, reassign_probs_chunk = self.encode(tx_features=tx_features,
                                                        temperature=self.min_temperature)
                reassign_probs_chunk = torch.prod(reassign_probs_chunk, dim=1, keepdim=True)
                reassign_probs = np.vstack([reassign_probs, reassign_probs_chunk.cpu().numpy()])
            # Store the computed probabilities 
            self.intf_tx = self.intf_tx.with_columns(pl.Series(name="reassign_probs_raw", values=reassign_probs.squeeze(-1)))
            # Calibrate the probabilities 
            self.intf_tx = self.intf_tx.with_columns(pl.Series(name="reassign_probs", values=self.calibrator(reassign_probs.squeeze(-1))))
            # Pick the most likely neighbor
            self.tx_reassign_info = self.intf_tx.group_by("molecule_id").agg(pl.all().sort_by("reassign_probs", descending=False).last())

    def _correct_tx(self,
                    adata_obs: pl.DataFrame,
                    reassign_threshold: float,
                    remove_threshold: float) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Correct tx given thrsholds

        Parameters
        ----------
        adata_obs : pl.DataFrame
            Cell types from adata.obs
        reassign_threshold : float
            Threshold for reassigning tx
        remove_threshold : float
            Threshold for removing tx in percentage of reassign_threshold. 
            For example, if reassign_threshold=0.3 and remove_threshold=1/3, 
            the actual threshold for removal is 0.3 * (1-1/3)=0.2

        Returns
        -------
        Tuple[pl.DataFrame, pl.DataFrame]
            Reassigned transcripts, removed transcripts 
        """
        tx_to_change = self.tx_reassign_info.drop(['prior_distance_feature',
                                                    'prior_exp_feature',
                                                    'prior_neighbor_exp_feature'])
        # Add reassign threshold 
        tx_to_change = tx_to_change.with_columns(pl.lit(reassign_threshold).cast(pl.Float64).alias("reassign_threshold"))
        # Add removal threshold 
        tx_to_change = tx_to_change.with_columns((pl.col("reassign_threshold")*(1-remove_threshold)).alias("remove_threshold"))
        # Dichotomize results 
        tx_to_change = tx_to_change.with_columns(pl.when((pl.col('reassign_probs')>pl.col("reassign_threshold"))).then(1).otherwise(0).alias('reassign'))
        # For not reassigned tx, remove them per threshold 
        tx_to_change = tx_to_change.with_columns(pl.when((pl.col("reassign")==0) & (pl.col('reassign_probs')>pl.col("remove_threshold"))).then(1).otherwise(0).alias('remove'))
        # Filter reassigned 
        tx_to_reassign = tx_to_change.filter(pl.col("reassign")==1).drop(["reassign", "remove", "reassign_threshold", "remove_threshold"])
        tx_to_remove = tx_to_change.filter(pl.col("remove")==1).drop(["reassign", "remove", "reassign_threshold", "remove_threshold"])
        # Rename for readability
        tx_to_reassign = tx_to_reassign.join(adata_obs.rename({"cell_type": "from_cell_type"}),
                                                how='left', left_on='cell_id', right_on="cell_id")
        tx_to_reassign = tx_to_reassign.join(adata_obs.rename({"cell_type": "to_cell_type"}),
                                                how='left', left_on='neighbor_cell_id', right_on="cell_id")
        tx_to_reassign = tx_to_reassign.drop(["cell_type", "neighbor_celltype"])
        # Filter removed 
        tx_to_remove = tx_to_remove.join(adata_obs.rename({"cell_type": "from_cell_type"}),
                                                how='left', left_on='cell_id', right_on="cell_id")
        tx_to_remove = tx_to_remove.drop(["cell_type", "neighbor_celltype"])
        
        return tx_to_reassign, tx_to_remove
    
    def _find_criteria(self,
                    adata_obs: pl.DataFrame,
                    reassign_threshold_grid: Union[np.array, list],
                    remove_threshold_grid: Union[np.array, list]) -> None:
        """Grid search for criteria 

        Parameters
        ----------
        adata_obs : pl.DataFrame
            Cell types from adata.obs
        reassign_threshold_grid : Union[np.array, list]
            An array of reassign threshold to try 
        remove_threshold_grid : Union[np.array, list]
            An array of remove threshold to try 
        """
        criterion_df = []
        zero_counts = self.adata.to_df().copy()
        zero_counts.loc[:,:] = 0
        zero_counts = pl.from_pandas(zero_counts, include_index=True)
        for reassign_threshold in tqdm(reassign_threshold_grid, desc="Reassign grid"):
            for remove_threshold in tqdm(remove_threshold_grid, desc="Remove grid"):
                # Correct tx 
                tx_to_reassign, tx_to_remove = self._correct_tx(adata_obs=adata_obs,
                                                            reassign_threshold=reassign_threshold,
                                                            remove_threshold=remove_threshold)
                # Generate patches 
                counts_to_subtract, counts_to_add, rm_counts_to_subtract = generate_count_patches(adata=zero_counts,
                                                                                                tx_to_reassign=tx_to_reassign,
                                                                                                tx_to_remove=tx_to_remove)
                # Split into chunks 
                X_chunks = even_split(array=(self.adata.to_df("counts_0")+counts_to_add-counts_to_subtract-rm_counts_to_subtract).values,
                               chunk_size=np.ceil(self.adata.X.shape[0]/100))
                leiden_chunks = even_split(array=self.adata.obs["counts_0_leiden"].astype(int).values,
                                    chunk_size=np.ceil(self.adata.X.shape[0]/100))
                loss = 0
                with torch.no_grad():
                    for X, leiden in zip(X_chunks, leiden_chunks):
                        X = torch.tensor(X, dtype=torch.float32, device=self.model_device)
                        leiden = torch.tensor(leiden, dtype=torch.int64, device=self.model_device)
                        cell_type_logits = self.cell_type_coefficients(X)
                        loss += F.cross_entropy(cell_type_logits, leiden, reduction='sum').cpu().numpy().item()
                loss /= (self.adata.X.shape[0])
                
                criterion_df.append((reassign_threshold, remove_threshold, loss, tx_to_remove.shape[0]))
        self.criterion_df=pd.DataFrame(criterion_df, columns=['reassign_threshold', 'remove_threshold_percent', 'loss', "n_removed"])
        # The first would be the best criterion
        self.criterion_df.sort_values("loss", ascending=True, ignore_index=True, inplace=True)
        self.criterion_df.loc[:, "remove_threshold"] = self.criterion_df.loc[:, 'reassign_threshold'] * (1-self.criterion_df.loc[:, 'remove_threshold_percent'])
        self.criterion_df.loc[:, "loss_increase"] = self.criterion_df.loc[:, 'loss']/self.criterion_df.at[0, 'loss']-1
        
    def correct_tx(self,
                    reassign_threshold_grid: Union[int, float, np.array, list]=np.arange(start=0.1, stop=0.6, step=0.1),
                    remove_threshold_grid: Union[int, float, np.array, list]=np.arange(start=0, stop=0.4, step=0.1),
                    choice_type: str="conservative") -> None:
        """Generate transcript reassignment based on various criteria 

        Parameters
        ----------
        reassign_threshold_grid : Union[int, float, np.array, list], optional
            An array of reassign threshold to try , by default np.arange(start=0.1, stop=0.5, step=0.1)
        remove_threshold_grid : Union[int, float, np.array, list], optional
            An array of remove threshold in percentage of reassign threshold to try, by default np.linspace(start=0, stop=1, num=10)
        """
        adata_obs = pl.from_pandas(self.adata.obs["cell_type"], include_index=True)  
        # Make sure the grid is iterable 
        if isinstance(reassign_threshold_grid, int) or isinstance(reassign_threshold_grid, float):
            reassign_threshold_grid = [reassign_threshold_grid]
        if isinstance(remove_threshold_grid, int) or isinstance(remove_threshold_grid, float):
            remove_threshold_grid = [remove_threshold_grid]
        if (len(reassign_threshold_grid) > 1) or (len(remove_threshold_grid) > 1):
            # Generate cirteria df
            self._find_criteria(adata_obs=adata_obs,
                                reassign_threshold_grid=reassign_threshold_grid,
                                remove_threshold_grid=remove_threshold_grid)
            if choice_type == "best":
                # The first one is the best 
                ind = 0
            elif choice_type == "aggressive":
                ind = self.criterion_df.loc[self.criterion_df['loss_increase'] < 0.01, 'n_removed'].idxmax(axis=0)
            elif choice_type == "conservative":
                ind = self.criterion_df.loc[self.criterion_df['n_removed']==0, 'loss'].idxmin(axis=0)
            else: 
                raise ValueError('Invalid choice type')
            reassign_threshold = self.criterion_df.at[ind, "reassign_threshold"]
            remove_threshold = self.criterion_df.at[ind, "remove_threshold_percent"]
        else: 
            reassign_threshold = reassign_threshold_grid[0]
            remove_threshold = remove_threshold_grid[0]
        # Generate tx 
        tx_to_reassign, tx_to_remove = self._correct_tx(adata_obs=adata_obs,
                                                        reassign_threshold=reassign_threshold,
                                                        remove_threshold=remove_threshold)

        trial_layer = self.current_layer+"_corrected"
        # For each criterion, we store the update 
        # And make assignment to the count matrix 
        # This will also compute UMAP by default
        # We do not actually reassign transcript at this stage 
        self.adata = make_reassignment_adata(adata=self.adata,
                                            layer=self.current_layer,
                                            tx_to_reassign=tx_to_reassign,
                                            tx_to_remove=tx_to_remove,
                                            trial_layer=trial_layer,
                                            dr_method=self.import_data_par['dr_method'])
        # Rename for readibility
        tx_to_reassign = tx_to_reassign.rename({"cell_id": "from_cell_id",
                                                "neighbor_cell_id": "to_cell_id"})
        tx_to_remove = tx_to_remove.rename({"cell_id": "from_cell_id"})
        
        self.tx_to_reassign = tx_to_reassign.to_pandas().copy()
        self.tx_to_remove = tx_to_remove.to_pandas().copy()
    
    def recluster(self,
                temperature: float=0.0,
                top_k: Optional[int]=None,
                new_layer: Optional[str]=None,
                overwrite_previous_trials: bool=False,
                update_leiden: bool=False) -> None:
        """Regenerate the clusters 

        Parameters
        ----------
        temperature : float, optional
            Temperature in sampling multinomial, by default 0.0
        top_k : Optional[int], optional
            Only the top k candidates will be sampled, by default None
        new_layer : Optional[str], optional
            Layer upon which the logits will be computed, by default None
        overwrite_previous_trials : bool, optional 
            Whether or not to overwrite previous trials at the same layer. This is useful is the user
            changes to the reassign criteria, by default False
        update_leiden : bool, optional
            Whether or not to update the latest leiden information. This is useful when the user 
            is really confident about the reclassification results, by default False
        """
        if new_layer is None:
            new_layer = self.current_layer+"_corrected"
        self.eval()
        with torch.no_grad():
            cell_by_gene_counts_chunks = even_split(array=self.adata.layers[new_layer],
                                                    chunk_size=np.ceil(self.adata.X.shape[0]/100))
            logits = torch.empty((0, self.adata.uns["n_leiden"]), dtype=torch.float32)
            cell_type_predict = torch.empty((0, 1), dtype=torch.int64)
            for cell_by_gene_counts_chunk in tqdm(cell_by_gene_counts_chunks):
                logits_chunk = self.cell_type_coefficients(torch.tensor(cell_by_gene_counts_chunk,
                                                                dtype=torch.float32, 
                                                                device=self.model_device)).cpu()
                logits = torch.cat((logits, logits_chunk), dim=0)
                # For top k, the -inf mask trick is used 
                if top_k is not None:
                    top_logits, _ = torch.topk(logits_chunk, top_k)
                    min_val = top_logits[:, [-1]]
                    logits_chunk = torch.where(
                        logits_chunk < min_val,
                        torch.tensor(float('-inf')).to(logits.device),
                        logits_chunk
                    )
                # If temperature is not 0, we sample from multinomial 
                if temperature > 0.0:
                    logits_chunk = logits_chunk/temperature
                    probs = torch.softmax(logits_chunk, dim=-1)
                    cell_type_predict_chunk = torch.multinomial(probs, num_samples=1)
                else:
                    cell_type_predict_chunk = torch.argmax(logits_chunk, dim=-1, keepdim=True)
                cell_type_predict = torch.cat((cell_type_predict, cell_type_predict_chunk), dim=0)
        # Record the results 
        cell_type_predict = cell_type_predict.numpy()
        logits = logits.numpy()
        probs = softmax(logits, axis=1)
        # perplexity is also computed to see how uncertain the model is 
        perplexity = np.exp(entropy(probs, axis=1, keepdims=True))
        # If the user wants to overwrite previous trials
        # we detect previous trials of the same layer and drop the columns 
        if overwrite_previous_trials:
            previous_trials = []
            i=0
            while True:
                new_leiden_name = new_layer + "_leiden_" + str(i)
                if new_leiden_name in self.adata.obs.columns:
                    previous_trials.append(new_leiden_name)
                    previous_trials.append(new_layer + "_cell_type_" + str(i))
                    previous_trials.append(new_leiden_name+"_perplexity")
                else:
                    break
                i += 1
            self.adata.obs.drop(columns=previous_trials, inplace=True)
        # To allow the user to recluster multiple times 
        # by running the recluster method multiple times 
        # The first time the user runs recluster, the index will be 0
        # after that every time the user runs recluster, the index will 
        # increment by 1
        i=0
        while True:
            new_leiden_name = new_layer + "_leiden_" + str(i)
            new_cell_type_name = new_layer + "_cell_type_" + str(i)
            if new_leiden_name not in self.adata.obs.columns:
                break 
            i += 1
        self.adata.obs[new_leiden_name] = cell_type_predict
        self.adata.obs[new_leiden_name] = self.adata.obs[new_leiden_name].astype(str)
        
        temp_df = self.adata.obs[[new_leiden_name]].merge(self.adata.uns['cell_type_leiden_map'],
                                            how='left', left_on = new_leiden_name,
                                            right_on = "cell_type_index")
        self.adata.obs[new_cell_type_name] = temp_df['cell_type_name'].values.copy()
        
        self.adata.obs[new_leiden_name+"_perplexity"] = perplexity
        if update_leiden:
            self.adata.obs['leiden'] = self.adata.obs[new_leiden_name].copy()
    
    def save_model(self,
                   dir_name: str,
                   model_name: str,
                   save_correction_result: bool=True) -> None:
        """Save the model

        Parameters
        ----------
        dir_name : str
            The path to the directory where the model along with other info will be saved.
        model_name : str
            The name of the model 
        save_reassigning_result : bool, optional 
            Whether or not to save the reassignment result, by default True 
        """
        torch.save({'model_state_dict': self.state_dict()} | \
                {'optimizer_state_dict': self.optimizer.state_dict()}, 
                os.path.join(dir_name, model_name+".pt"))
        
        model_meta = {'import_data_par': self.import_data_par,
                      'calculate_mask_distance_par': self.calculate_mask_distance_par,
                      'generate_feature_par': self.generate_feature_par,
                      'n_genes': self.n_genes,
                      'n_leiden': self.n_leiden,
                      'prior_50_reassign_prob': self.prior_50_reassign_prob,
                      'prior_5_reassign_prob': self.prior_5_reassign_prob,
                      'current_layer': self.current_layer,
                      "save_correction_result": save_correction_result}
        
        with open(os.path.join(dir_name, model_name+"_meta.json"), "w") as f:
            json.dump(model_meta, f, cls=JSONEncoder)
        
        if save_correction_result:
            self.criterion_df.to_csv(os.path.join(dir_name, model_name+"_criteria_df.csv"),
                                     index=False)
            self.tx_to_reassign.to_parquet(os.path.join(dir_name, model_name+"_tx_to_reassign.parquet"))
            self.tx_to_remove.to_parquet(os.path.join(dir_name, model_name+"_tx_to_remove.parquet"))
            
    def load_model(self,
                   dir_name: str,
                   model_name: str) -> None:
        """Load the model

        Parameters
        ----------
        dir_name : str
            The path to the directory where the model along with other info will be saved.
        model_name : str
            The name of the model 
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
        save_correction_result = model_meta['save_correction_result']
        
        if save_correction_result:
            self.criterion_df = pd.read_csv(os.path.join(dir_name, model_name+"_criteria_df.csv"))
            self.tx_to_reassign = pd.read_parquet(os.path.join(dir_name, model_name+"_tx_to_reassign.parquet"))
            self.tx_to_remove = pd.read_parquet(os.path.join(dir_name, model_name+"_tx_to_remove.parquet"))
        
        self.initialize_parameters()
        checkpoint = torch.load(os.path.join(dir_name, model_name+".pt")) 
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        
            
    