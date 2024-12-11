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
                preprocess: bool=True,
                max_centroid_dist: float=50,
                min_centroid_dist: float=0,
                mask_dist_cutoff: float=1,
                num_rep: int=3,
                method: str="split",
                reparametrize: bool=True) -> None:
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
        min_centroid_dist : int, optional
            The threshold on cell-cell centroid distances under which we do not consider two cells being neighbors, by default 0
        mask_dist_cutoff : float, optional
            The threshold of cell-cell distances beyond which a cell is no longer considered a neighbor, by default 1
        num_rep : int, optional
            Number of pseudo bulks, by default 3
        method : str, optional
            Method to generate pseudo bulks. Can be split or bootstrap, by default "split"
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
            'method': method
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
        
        # Create parameters 
        self.model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.reparametrize = reparametrize
        self.reassign_coefficients = None 
        self.prior_reassign_coefficients = None
        self.cell_type_coefficients = None 
        self.optimizer = None
        
        
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
        
    def patchfy_data(self) -> None:
        self.coord_list = generate_patch_coords(adata=self.adata, intf_tx=self.intf_tx) 
    
    def initiate_parameters(self,
                            prior_50_reassign_prob: float=0.01,
                            prior_5_reassign_prob: float=0.5) -> None:
        
        # The reassign_coefficients determines the logit of the reassigning probability of a 
        # transcript based on computed features.
        self.reassign_coefficients = nn.Linear(in_features=3, out_features=1, bias=True)
        # Due to the nature of the crafted features, we put a positivity constraint on the coefficients
        if self.reparametrize:
            parametrize.register_parametrization(self.reassign_coefficients, "weight", Positive())
        
        self.prior_reassign_coefficients = nn.Linear(in_features=1, out_features=1, bias=True)
        alpha_0 = -np.log(1/prior_50_reassign_prob-1+1e-20)
        temp = -np.log(0.05/0.95)
        alpha_1 = (-np.log(1/prior_5_reassign_prob-1+1e-20) - alpha_0)/temp
        self.prior_reassign_coefficients.bias = nn.Parameter(torch.tensor(alpha_0, dtype=torch.float32).reshape_as(self.prior_reassign_coefficients.bias))
        self.prior_reassign_coefficients.weight = nn.Parameter(torch.tensor(alpha_1, dtype=torch.float32).reshape_as(self.prior_reassign_coefficients.weight))
        for param in self.prior_reassign_coefficients.parameters():
            param.requires_grad = False
        # The cell type coefficients takes the gene counts and outputs the logits for all the cell types 
        self.cell_type_coefficients = nn.Linear(in_features=self.adata.uns['n_genes'], 
                                                out_features=self.adata.uns['counts_0_n_leiden'],
                                                bias=True)
        # Adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
    
    def encode(self, 
               tx_features: torch.tensor,
               temperature: float) -> Tuple[torch.tensor, torch.tensor]:
        reassign_logits = self.reassign_coefficients(tx_features) 
        reassign_probs = binary_gumbel_softmax_sample(logits=reassign_logits,
                                                        temperature=temperature,
                                                        model_device=self.model_device)
        reassign_hard = torch.round(reassign_probs, decimals=0)
        reassign_hard = (reassign_hard - reassign_probs).detach() + reassign_probs
        
        return reassign_hard, reassign_probs
        
    def decode(self, 
               updated_cell_by_gene_counts: torch.tensor,
               prior_distance_features: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        cell_type_logits = self.cell_type_coefficients(updated_cell_by_gene_counts)
        prior_reassign_logits = self.prior_reassign_coefficients(prior_distance_features)
        prior_reassign_probs = torch.sigmoid(prior_reassign_logits)
        return cell_type_logits, prior_reassign_probs
    
    def forward(self,
                tx_features: torch.tensor, 
                tx_prior_features: torch.tensor,
                cell_by_gene_counts: torch.tensor, 
                row_index_self: torch.tensor,
                row_index_neighbor: torch.tensor,
                col_index: torch.tensor,
                temperature: float) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        
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
        
        cell_type_logits, prior_reassign_probs = self.decode(updated_cell_by_gene_counts=cell_by_gene_counts,
                                                            prior_distance_features=tx_prior_features)
        
        return reassign_probs, cell_type_logits, prior_reassign_probs
    
    def loss_function(self,
                      cell_type_logits: torch.tensor, 
                      cell_type_labels: torch.tensor,
                      reassign_probs: torch.tensor,
                      prior_reassign_probs: torch.tensor, 
                      verbose: bool=False) -> torch.tensor:
        # Cross entropy loss (despite its name it's a loss)
        CEl = F.cross_entropy(cell_type_logits,
                              cell_type_labels, reduction='mean') 
        
        log_ratio_1 = torch.log(reassign_probs/prior_reassign_probs+1e-20)
        log_ratio_0 = torch.log((1-reassign_probs)/(1-prior_reassign_probs)+1e-20)
        KLD_reassign = torch.sum(reassign_probs * log_ratio_1 + (1-reassign_probs) * log_ratio_0, dim=-1).mean()
        
        if verbose:
            print("="*30)
            print("The cross entropy loss is {} and KLD_reassign is {}".format(CEl.detach().numpy(), 
                                                                      KLD_reassign.detach().numpy()))
            print("="*30) 
        return CEl + KLD_reassign
    
    def training_loop(self,
                      n_epochs: int,
                      verbose: bool=False) -> None:
        self.train()
        for epoch in range(n_epochs):
            np.random.shuffle(self.coord_list)
            self.current_temperature = self.init_temperature
            train_loss = 0.0
            for minibatch_ind, coord in enumerate(self.coord_list):
                cell_by_gene_counts, tx_features, tx_prior_features, cell_type_labels, row_index_self, row_index_neighbor, col_index = load_patch(adata=self.adata,
                                                                                                                                intf_tx=self.intf_tx,
                                                                                                                                coord_tuple=coord,
                                                                                                                                layer=self.current_layer,
                                                                                                                                model_device=self.model_device)
                reassign_probs, cell_type_logits, prior_reassign_probs = self(tx_features=tx_features, 
                                                                            tx_prior_features=tx_prior_features,                
                                                                            cell_by_gene_counts=cell_by_gene_counts, 
                                                                            row_index_self=row_index_self,
                                                                            row_index_neighbor=row_index_neighbor,
                                                                            col_index=col_index,
                                                                            temperature=self.current_temperature)
                self.optimizer.zero_grad()
                loss = self.loss_function(cell_type_logits=cell_type_logits,
                                          cell_type_labels=cell_type_labels,
                                          reassign_probs=reassign_probs,
                                          prior_reassign_probs=prior_reassign_probs,
                                          verbose=verbose)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                if minibatch_ind % 100 == 1:
                    self.current_temperature = np.maximum(self.current_temperature * np.exp(-self.ANNEAL_RATE * minibatch_ind), self.min_temperature) 

                if minibatch_ind % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f} Training Loss: {:.6f}'.format(
                            epoch, minibatch_ind, len(self.coord_list),
                                100. * minibatch_ind / len(self.coord_list),
                                loss.item(), train_loss))
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, train_loss / len(self.coord_list)))

    def trial_reassign_tx(self,
                          criteria: dict) -> None:
        self.eval()
        with torch.no_grad():
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
            
            self.intf_tx['reassign_probs'] = reassign_probs.squeeze(-1)
            for criterion_name in criteria: 
                criterion = criteria[criterion_name]
                if isinstance(criterion, str):
                    self.intf_tx['reassign'] = (reassign_hard.squeeze(-1)) 
                else:
                    self.intf_tx['reassign'] = (reassign_probs.squeeze(-1)>criterion)  
                tx_to_reassign = self.intf_tx.loc[self.intf_tx['reassign']==1, 
                                                    ['molecule_id', 'cell_id', 'neighbor_cell_id', "gene"]]
                self.intf_tx.drop(columns=['reassign'], inplace=True)
                trial_layer = self.current_layer+"_"+criterion_name
                self.tx_to_reassign_dict[trial_layer] = tx_to_reassign.copy()
                self.adata = make_reassignment_adata(adata=self.adata,
                                                     layer=self.current_layer,
                                                     tx_to_reassign=tx_to_reassign,
                                                     trial_layer=trial_layer,
                                                     preprocess=self.import_data_par['preprocess'])
    
    def final_reassign_tx(self,
                          selected_criterion: str) -> None:
        
        tx_to_reassign = self.tx_to_reassign_dict[self.current_layer+"_"+selected_criterion].copy()
        self.adata = make_reassignment_adata(adata=self.adata, 
                                             layer=self.current_layer,
                                             tx_to_reassign=tx_to_reassign,
                                             preprocess=self.import_data_par['preprocess'])
        self.tx_metadata = make_reassignment_tx_metadata(tx_to_reassign=tx_to_reassign,
                                                         tx_metadata=self.tx_metadata)
        layer_num = extract_layer_num(self.current_layer)
        self.current_layer = "counts_"+str(int(layer_num+1))

    def recluster(self,
                  temperature=0.0,
                  top_k=None,
                  new_layer: Optional[str]=None):
        if new_layer is None:
            new_layer = self.current_layer
        self.eval()
        with torch.no_grad():
            cell_by_gene_counts_chunks = even_split(array=self.adata.layers[new_layer].values,
                                                    chunk_size=1000)
            logits = torch.empty((0, self.adata.uns["counts_0_n_leiden"]), dtype=torch.float32)
            cell_type_predict = torch.empty((0, 1), dtype=torch.int64)
            for cell_by_gene_counts_chunk in cell_by_gene_counts_chunks:
                logits_chunk = self.cell_type_coefficients(torch.tensor(cell_by_gene_counts_chunk,
                                                                dtype=torch.float32, 
                                                                device=self.model_device))
                logits = torch.cat((logits, logits_chunk), dim=0)
                
                if top_k is not None:
                    top_logits, _ = torch.topk(logits_chunk, top_k)
                    min_val = top_logits[:, -1]
                    logits_chunk = torch.where(
                        logits_chunk < min_val,
                        torch.tensor(float('-inf')).to(logits.device),
                        logits_chunk
                    )
                if temperature > 0.0:
                    logits_chunk = logits_chunk/temperature
                    probs = torch.softmax(logits_chunk, dim=-1)
                    cell_type_predict_chunk = torch.multinomial(probs, num_samples=1)
                else:
                    cell_type_predict_chunk = torch.argmax(logits_chunk, dim=-1, keepdim=True)
                cell_type_predict = torch.cat((cell_type_predict, cell_type_predict_chunk), dim=0)
                
        return cell_type_predict.numpy(), logits.numpy()
    
    def save_model(self,
                   path: str) -> None:
        dir_name = os.path.dirname(path)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, path)
        # Since h5ad does not support saving geopandas 
        self.adata.obs = pd.DataFrame(self.adata.obs)
        self.adata.obs.drop(columns=['cell_centroid_geom'], inplace=True)
        self.adata.write_h5ad(os.path.join(dir_name, "adata.h5ad"))
        self.adata.obs = gpd.GeoDataFrame(self.adata.obs,
                                geometry=gpd.points_from_xy(self.adata.obs['x'], self.adata.obs['y']))
        self.adata.obs.rename_geometry("cell_centroid_geom", inplace=True)
        
        # Then other files 
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
                   path: str) -> None:
        dir_name = os.path.dirname(path)
        self.adata = sc.read_h5ad(os.path.join(dir_name, "adata.h5ad"))
        # Reconstruct the geodataframe 
        self.adata.obs = gpd.GeoDataFrame(self.adata.obs,
                                geometry=gpd.points_from_xy(self.adata.obs['x'], self.adata.obs['y']))
        self.adata.obs.rename_geometry("cell_centroid_geom", inplace=True)
        
        self.initiate_parameters()
        checkpoint = torch.load(path) 
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Other files 
        self.tx_metadata = read_parquet(os.path.join(dir_name, "tx_metadata.parquet"))
        self.cell_coords = read_parquet(os.path.join(dir_name, "cell_coords.parquet"))
        self.mask_distance = read_parquet(os.path.join(dir_name, "mask_distance.parquet"))
        self.intf_tx = read_parquet(os.path.join(dir_name, "intf_tx.parquet"))
        self.tx_to_reassign_dict = json.load(open(os.path.join(dir_name, "tx_to_reassign_dict.json")))
        # Then convert into dataframes 
        for k in self.tx_to_reassign_dict:
            self.tx_to_reassign_dict[k] = pd.read_json(self.tx_to_reassign_dict[k])
            
    