# Data manipulation
import numpy as np 
import scanpy as sc 
import geopandas as gpd 
import torch 
from scipy.signal import oaconvolve
from collections import Counter
# Typing
from typing import Tuple


def trial_patch_coords(adata: sc.AnnData,
                    intf_tx: gpd.GeoDataFrame,
                    percent_cell_per_patch: float=0.01) -> Tuple[list, np.array]:
    """Generate patch coordinates based on a rough percentage of cells 

    Parameters
    ----------
    adata : sc.AnnData
        An Anndata containing cell centroid information 
    intf_tx : gpd.GeoDataFrame
        Transcripts potentially to be reassigned 
    percent_cell_per_patch : float, optional
        Rough percentage of cells to be included in one patch, by default 0.01

    Returns
    -------
    Tuple[list, np.array]
        A list of patches represented by their coordinates in the order of left, right, bottom, and top.
        An array of numbers of found cells within each patch 
    """
    # According to the percentage, we shrink the total lengths of the two axies 
    # These will be the width and length of a patch 
    dx = (adata.uns['centroid_x_max']-adata.uns['centroid_x_min'])*np.sqrt(percent_cell_per_patch)
    dy = (adata.uns['centroid_y_max']-adata.uns['centroid_y_min'])*np.sqrt(percent_cell_per_patch)
    # Each patch will be visited 10 times 
    n_rep = 10
    # The increment steps of x and y
    inc_x = dx/n_rep
    inc_y = dy/n_rep
    # Define intervals of the form 
    # [x_min-inc_x + n*inc_x, x_min-inc_x + (n+1)*inc_x))
    # n starts from 0
    left_x_start = adata.uns['centroid_x_min']-inc_x
    bottom_y_start = adata.uns['centroid_y_min']-inc_y
    # Left x /bottom y end are only approximate left ends of the final intervals
    left_x_end = adata.uns['centroid_x_max']
    bottom_y_end = adata.uns['centroid_y_max']
    # Number of increments 
    n_inc_x = int(np.floor((left_x_end - left_x_start)/inc_x)+1)
    n_inc_y = int(np.floor((bottom_y_end - bottom_y_start)/inc_y)+1)
    # Discretize coordinates 
    coord_ind_matrix = np.zeros((n_inc_y, n_inc_x))
    # Find which interval each point belongs to
    x_ind = np.floor((adata.obs['x'] - left_x_start)/inc_x).values.astype(int)
    y_ind = np.floor((adata.obs['y'] - bottom_y_start)/inc_y).values.astype(int)
    # Make sure all cells are also in the intf_tx 
    intf_ind = adata.obs.index.isin(intf_tx['cell_id'])
    x_ind = x_ind[intf_ind]
    y_ind = y_ind[intf_ind]
    # Find for each cell of the matrix, how many data points we have 
    freq_table = Counter(zip(y_ind, x_ind))
    coord_ind_matrix[tuple(zip(*freq_table.keys()))] = list(freq_table.values())
    # Then convolve with a 10*10 ones 
    # This is basically perform overlapping summation 
    patch = np.ones((n_rep, n_rep))
    n_cells_per_patch = oaconvolve(coord_ind_matrix, patch, mode='full')
    # Filter out positions with low cell counts 
    bottom_y_ind_array, left_x_ind_array = np.where(n_cells_per_patch > 10)
    # Correct for extra rows and columns resulting from the convolution 
    offset_x = n_cells_per_patch.shape[0] - coord_ind_matrix.shape[0] 
    offset_y = n_cells_per_patch.shape[1] - coord_ind_matrix.shape[1] 
    non_edge = np.where((bottom_y_ind_array>=offset_x) &\
                        (left_x_ind_array>=offset_y))[0]
    # Map indices back to coordinates 
    left_x_array = left_x_start + (left_x_ind_array[non_edge]-offset_x) * inc_x
    bottom_y_array = bottom_y_start + (bottom_y_ind_array[non_edge]-offset_y) * inc_y

    right_x_array = left_x_array + dx
    top_y_array = bottom_y_array + dy

    coord_list = list(zip(left_x_array, right_x_array, bottom_y_array, top_y_array))
    return coord_list, n_cells_per_patch 


def generate_patch_coords(adata: sc.AnnData,
                          intf_tx: gpd.GeoDataFrame) -> list:
    """Generate patches represented by their coordinates 

    Parameters
    ----------
    adata : sc.AnnData
        An Anndata containing cell centroid information 
    intf_tx : gpd.GeoDataFrame
        Transcripts potentially to be reassigned 

    Returns
    -------
    list
        A list of patches represented by their coordinates in the order of left, right, bottom, and top.
    """
    
    n_cells = adata.X.shape[0]
    n_genes = adata.uns['n_genes']
    # The maximum number of transcripts detected within a cell 
    max_tx = adata.layers['counts_0'].sum(axis=1).max()
    # As the model parameters are almost negligible we focus on the data 
    # We assume 16GB of either RAM or GPU memory 
    # (this is satisfied by most modern day laptop and GPUs >= p100)
    # The actual computation will also consume memories, therefore, we confine the data 
    # to maximally 8GB 
    # The following formula generally underestimates by assuming all cells possess the maximal number of tx 
    max_n_cells_per_patch_can_have = 8*(1024**3)/(4*(4*max_tx + n_genes))
    # Want to make sure roughly each patch will have at least 100 cells 
    # But at the same time, we have enough patches 
    if n_cells <= max_n_cells_per_patch_can_have:
        percent_cell_per_patch = np.clip(np.max([100/n_cells, 0.01]), 
                                         a_min=None, a_max=1)
    else:
        percent_cell_per_patch = np.clip(np.max([100/n_cells, max_n_cells_per_patch_can_have/n_cells/100]),
                                         a_min=None, a_max=1)
    # Make sure the actual maximal number does not exceed the limit
    # Otherwise, shrink the percentage a little bit and try again 
    accept = False
    while not accept: 
        coord_list, n_cells_per_patch = trial_patch_coords(adata=adata,
                                                            intf_tx=intf_tx,
                                                            percent_cell_per_patch=percent_cell_per_patch) 
        max_n_cells_per_patch = np.max(n_cells_per_patch) 
        if max_n_cells_per_patch <= max_n_cells_per_patch_can_have:
            accept = True
        else:
            percent_cell_per_patch *= 0.9
        
    return coord_list


def load_patch(adata: sc.AnnData, 
                intf_tx: gpd.GeoDataFrame,
                coord_tuple: tuple,
                layer: str,
                model_device: torch.device) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """Given a tuple of coordinates, extract the gene counts, tx features, and auxiliary information  

    Parameters
    ----------
    adata : sc.AnnData
        An Anndata containing cell centroid information 
    intf_tx : gpd.GeoDataFrame
        Transcripts potentially to be reassigned 
    coord_tuple : tuple
        A tuple of the four coordinates of a patch 
    layer : str
        The name of the layer from which gene counts are extracted 
    model_device : torch.device
        The torch device 

    Returns
    -------
    Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]
        A tensor of gene counts used for cell type classification 
        A tensor of features used for computing transcript reassignment probability after seeing the cell type information 
        A tensor of features used for computing transcript reassignment probability without the cell type information 
        A tensor of cell type labels 
        A tensor of the row indices of cells themselves 
        A tensor of the row indices of cells' neighbors 
        A tensor of gene indices 
    """
    left_x, right_x, bottom_y, top_y = coord_tuple
    # Extract gene counts from half closed half open intervals 
    cell_patch = adata.to_df(layer)[(adata.obs['x']>=left_x) & (adata.obs['x']<right_x) & \
                            (adata.obs['y']>=bottom_y) & (adata.obs['y']<top_y)].copy()  
    
    cell_patch['row_index'] = [i for i in range(cell_patch.shape[0])]
    # Add cell type information 
    cell_patch = cell_patch.merge(adata.obs[['leiden']], how='left',
                                  left_index=True, right_index=True)
    # Make sure all cells as well as their neighbors are within the patch 
    tx_patch = intf_tx.loc[(intf_tx['cell_id'].isin(cell_patch.index)) & \
                            (intf_tx['neighbor_cell_id'].isin(cell_patch.index)), :].copy()
    # Generate three indices to be used for adjusting gene counts 
    tx_patch = tx_patch.merge(adata.var[['col_index']], 
               how="left", left_on="gene", right_index=True)
    tx_patch = tx_patch.merge(cell_patch[['row_index']], 
               how='left', left_on="cell_id", right_index=True).rename(columns={"row_index": "row_index_self"})
    tx_patch = tx_patch.merge(cell_patch[['row_index']], 
               how='left', left_on="neighbor_cell_id", right_index=True).rename(columns={"row_index": "row_index_neighbor"})
    
    cell_type_labels = torch.tensor(cell_patch['leiden'].astype(int).values, dtype=torch.int64)
    # cell_type_labels = torch.tensor(cell_patch['leiden'].astype(int).values, dtype=torch.int64).unsqueeze(1)
    # cell_type_labels = torch.zeros(cell_patch.shape[0], 40, dtype=torch.float32).scatter(0, cell_type_labels, 1).to(model_device)
    # Drop irrelevant columns and convert dataframes to tensors 
    cell_patch.drop(columns=['row_index', "leiden"], inplace=True)
    cell_by_gene_counts = torch.tensor(cell_patch.values, dtype=torch.float32, device=model_device)
    tx_features = torch.tensor(tx_patch[['distance_feature', 
                                         "neighbor_self_exp_feature", 
                                         "rest_self_exp_feature"]].values, 
                               dtype=torch.float32, device=model_device)
    tx_prior_features = torch.tensor(tx_patch[['prior_distance_feature']].values,
                                     dtype=torch.float32, device=model_device)
    row_index_self = torch.LongTensor(tx_patch[['row_index_self']].values, device=model_device)
    row_index_neighbor = torch.LongTensor(tx_patch[['row_index_neighbor']].values, device=model_device)
    col_index = torch.LongTensor(tx_patch[['col_index']].values, device=model_device)
    
    return cell_by_gene_counts, tx_features, tx_prior_features, cell_type_labels, row_index_self, row_index_neighbor, col_index
    
    



