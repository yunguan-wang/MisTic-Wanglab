import numpy as np 
import pandas as pd 
import torch 

# Can also construct based on RAM
# Set an upper limit  we can decide on the percentage 

def generate_patch_coords(adata,
                          percent_cell_per_patch: float=0.01):
    coord_list = []
    dx = (adata.uns['centroid_x_max']-adata.uns['centroid_x_min'])*np.sqrt(percent_cell_per_patch)
    dy = (adata.uns['centroid_y_max']-adata.uns['centroid_y_min'])*np.sqrt(percent_cell_per_patch)
    left_x_array = np.arange(start=adata.uns['centroid_x_min']-dx/10,
                            stop=adata.uns['centroid_x_max']+dx/10,
                            step=dx/10)
    bottom_y_array = np.arange(start=adata.uns['centroid_y_min']-dy/10,
                            stop=adata.uns['centroid_y_max']+dy/10,
                            step=dy/10)
    for left_x in left_x_array:
        for bottom_y in bottom_y_array:
            right_x = left_x + dx
            top_y = bottom_y + dy
            ind = (adata.obs['x']>left_x) & (adata.obs['x']<right_x) & (adata.obs['y']>bottom_y) & (adata.obs['y']<top_y)
            if ind.sum() > 10:
                coord_list.append((left_x, right_x, bottom_y, top_y))
        
    return coord_list


def load_patch(adata, 
                intf_tx,
                coord_tuple,
                layer,
                model_device: torch.device):
    
    left_x, right_x, bottom_y, top_y = coord_tuple
    
    cell_patch = adata.to_df(layer)[(adata.obs['x']>left_x) & (adata.obs['x']<right_x) & \
                            (adata.obs['y']>bottom_y) & (adata.obs['y']<top_y)].copy()  
    
    cell_patch['row_index'] = [i for i in range(cell_patch.shape[0])]
    cell_patch = cell_patch.merge(adata.obs[['leiden']], how='left',
                                  left_index=True, right_index=True)
    
    tx_patch = intf_tx.loc[(intf_tx['cell_id'].isin(cell_patch.index)) & \
                            (intf_tx['neighbor_cell_id'].isin(cell_patch.index)), :].copy()
    
    tx_patch = tx_patch.merge(adata.var[['col_index']], 
               how="left", left_on="gene", right_index=True)
    tx_patch = tx_patch.merge(cell_patch[['row_index']], 
               how='left', left_on="cell_id", right_index=True).rename(columns={"row_index": "row_index_self"})
    tx_patch = tx_patch.merge(cell_patch[['row_index']], 
               how='left', left_on="neighbor_cell_id", right_index=True).rename(columns={"row_index": "row_index_neighbor"})
    
    cell_type_labels = torch.tensor(cell_patch['leiden'].astype(int).values, dtype=torch.int64)
    # cell_type_labels = torch.tensor(cell_patch['leiden'].astype(int).values, dtype=torch.int64).unsqueeze(1)
    # cell_type_labels = torch.zeros(cell_patch.shape[0], 40, dtype=torch.float32).scatter(0, cell_type_labels, 1).to(model_device)
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
    # neighbor_mask_distance_rank = torch.tensor(tx_patch[['neighbor_mask_distance_rank']].values,
    #                                             dtype=torch.float32, device=model_device)
    
    return cell_by_gene_counts, tx_features, tx_prior_features, cell_type_labels, row_index_self, row_index_neighbor, col_index
    
    



