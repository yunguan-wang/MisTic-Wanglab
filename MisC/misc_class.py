# Data IO
import os
import json 
import pandas as pd
import scanpy as sc
import geopandas as gpd
import polars as pl 
# Data manipulation 
import numpy as np
from scipy.stats import ks_2samp
import torch 
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
# Utility function 
from MisC.utility import import_data, calculate_mask_distance, binary_gumbel_softmax_sample,\
        make_reassignment_adata, diagLinear, Positive, JSONEncoder, even_split, process_time_ram
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
                dr_method: str='umap',
                max_centroid_dist: float=50,
                mask_dist_cutoff: float=5,
                nearest: int=3,
                prior_50_reassign_prob: float=0.01,
                prior_5_reassign_prob: float=0.99,
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
            'dr_method': dr_method
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
        self.intf_tx = None
        self.tx_reassign_info = None
        self.coord_list = {"neighbor{}".format(i): [] for i in range(nearest)}
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
            _description_, by default 0.1
        num_overlap : int, optional
            _description_, by default 7
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
        
        alpha_0 = -np.log(1/self.prior_50_reassign_prob-1+1e-20)
        temp = -np.log(0.05/0.95) 
        alpha_1 = (-np.log(1/self.prior_5_reassign_prob-1+1e-20) - alpha_0)/temp
        
        self.prior_reassign_coefficients = diagLinear(features=3, bias=True)
        self.prior_reassign_coefficients.bias = nn.Parameter(torch.tensor([alpha_0]*3, dtype=torch.float32).reshape_as(self.prior_reassign_coefficients.bias))
        self.prior_reassign_coefficients.weight = nn.Parameter(torch.tensor([alpha_1]*3, dtype=torch.float32).reshape_as(self.prior_reassign_coefficients.weight))
        for param in self.prior_reassign_coefficients.parameters():
            param.requires_grad = False
        
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
        prior_distance_features : torch.tensor
            The transcript distance features before seeing the cell type information  

        Returns
        -------
        Tuple[torch.tensor, torch.tensor]
            Logits for different cell types and the prior reassigning probabilities 
        """
        cell_type_logits = self.cell_type_coefficients(updated_cell_by_gene_counts)
        
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
                      prior_reassign_probs: torch.tensor) -> torch.tensor:
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
            if epoch >= 2:
                cel_previous_2 = np.array(self.CEl_list[epoch-2])
                cel_previous_1 = np.array(self.CEl_list[epoch-1])
                p_val = ks_2samp(cel_previous_2, cel_previous_1).pvalue
                if p_val > early_stop_pval:
                    print("="*30)
                    print("No improvement in the classification task detected. Stop early at epoch {}".format(epoch-1))
                    print("="*30)
                    break
            neighbor_indices = list(range(self.generate_feature_par['nearest']))
            np.random.shuffle(neighbor_indices)
            for neighbor_index in neighbor_indices:
                key = "neighbor"+str(neighbor_index)
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
            self.intf_tx = self.intf_tx.with_columns(pl.Series(name="reassign_probs", values=reassign_probs.squeeze(-1)))
            self.tx_reassign_info = self.intf_tx.group_by("molecule_id").agg(pl.all().sort_by("reassign_probs", descending=False).last())
        
    def reassign_tx(self,
                    criteria: dict={"threshold": 0.5}) -> None:
        """Generate transcript reassignment based on various criteria 

        Parameters
        ----------
        criteria : dict, optional
            A dictionary of criterion name-criterion pair. For random assignment, the criterion should be a string, by default {"threshold": 0.5, "random": "random"}
        """
        adata_obs = pl.from_pandas(self.adata.obs["cell_type"], include_index=True)
        for criterion_name in tqdm(criteria): 
            tx_to_reassign = self.tx_reassign_info.drop(['prior_distance_feature',
                                                        'prior_exp_feature',
                                                        'prior_neighbor_exp_feature'])
            criterion = criteria[criterion_name]
            if isinstance(criterion, str):
                reassign = np.random.binomial(n=1, p=tx_to_reassign['reassign_probs'])
            else:
                reassign = (tx_to_reassign['reassign_probs']>criterion).to_numpy().astype(int)
            tx_to_reassign = tx_to_reassign.with_columns(pl.Series(name='reassign', values=reassign))
            tx_to_reassign = tx_to_reassign.filter(pl.col("reassign")==1).drop("reassign")
            
            tx_to_reassign = tx_to_reassign.join(adata_obs[['cell_type']].rename({"cell_type": "from_cell_type"}),
                                                 how='left', left_on='cell_id', right_index=True)
            tx_to_reassign = tx_to_reassign.join(adata_obs[['cell_type']].rename({"cell_type": "to_cell_type"}),
                                                 how='left', left_on='neighbor_cell_id', right_index=True)
            tx_to_reassign = tx_to_reassign.drop(["cell_type", "neighbor_celltype"])
            
            trial_layer = self.current_layer+"_"+criterion_name
            # For each criterion, we store the update 
            # And make assignment to the count matrix 
            # This will also compute UMAP by default
            # We do not actually reassign transcript at this stage 
            self.adata = make_reassignment_adata(adata=self.adata,
                                                layer=self.current_layer,
                                                tx_to_reassign=tx_to_reassign,
                                                trial_layer=trial_layer,
                                                dr_method=self.import_data_par['dr_method'])
            tx_to_reassign = tx_to_reassign.rename({"cell_id": "from_cell_id",
                                                    "neighbor_cell_id": "to_cell_id"})
            self.tx_to_reassign_dict[trial_layer] = tx_to_reassign.to_pandas().copy()
        
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
            assert self.current_layer+"_"+selected_criterion in self.tx_to_reassign_dict, "selected_criterion not found in tx_to_reassign_dict"
        
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
                      "save_reassigning_result": save_reassigning_result,
                      "selected_criterion": selected_criterion}
        
        with open(os.path.join(dir_name, model_name+"_meta.json"), "w") as f:
            json.dump(model_meta, f, cls=JSONEncoder)
            
        if save_reassigning_result:
            tx_to_reassign = self.tx_to_reassign_dict[self.current_layer+"_"+selected_criterion].copy()
            tx_to_reassign.to_parquet(os.path.join(dir_name, model_name+"_tx_to_reassign.parquet"))
        
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
        
        save_reassigning_result = model_meta['save_reassigning_result']
        selected_criterion = model_meta['selected_criterion']
        
        if save_reassigning_result:
            tx_to_reassign = pd.read_parquet(os.path.join(dir_name, model_name+"_tx_to_reassign.parquet"))
            self.tx_to_reassign_dict = {self.current_layer+"_"+selected_criterion: tx_to_reassign}
        
        self.initialize_parameters()
        checkpoint = torch.load(os.path.join(dir_name, model_name+".pt")) 
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        
            
    