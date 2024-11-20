import numpy as np 
import pandas as pd 
import torch 


def generate_patch_coords(adata,
                          half_patch_size_x,
                          half_patch_size_y,
                          n_patches_x,
                          n_patches_y):
    coord_list = []
    center_x_array = np.linspace(start=adata.uns['centroid_x_min']-10,
                                 stop=adata.uns['centroid_x_max']+10,
                                 num=n_patches_x)
    dx = center_x_array[1] - center_x_array[0]
    half_patch_size_x = np.max([dx/2, half_patch_size_x])
    center_y_array = np.linspace(start=adata.uns['centroid_y_min']-10,
                                 stop=adata.uns['centroid_y_max']+10,
                                 num=n_patches_y)
    dy = center_x_array[1] - center_x_array[0]
    half_patch_size_y = np.max([dy/2, half_patch_size_y])
        
    for center_x in center_x_array:
        for center_y in center_y_array:
            left_x = center_x - half_patch_size_x
            right_x = center_x + half_patch_size_x
            bottom_y = center_y - half_patch_size_y
            top_y = center_y + half_patch_size_y
            
            ind = (adata.obs['x']>left_x) & (adata.obs['x']<right_x) & (adata.obs['y']>bottom_y) & (adata.obs['y']<top_y)
            if ind.sum() > 100:
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
    tx_features = torch.tensor(tx_patch[['distance_ratio', 
                                         "neighbor_self_exp_feature", 
                                         "rest_self_exp_feature"]].values, 
                               dtype=torch.float32, device=model_device)
    row_index_self = torch.LongTensor(tx_patch[['row_index_self']].values, device=model_device)
    row_index_neighbor = torch.LongTensor(tx_patch[['row_index_neighbor']].values, device=model_device)
    col_index = torch.LongTensor(tx_patch[['col_index']].values, device=model_device)
    tx_mask_distance = torch.tensor(tx_patch[['tx_mask_distance']].values,
                                    dtype=torch.float32, device=model_device)
    
    return cell_by_gene_counts, tx_features, cell_type_labels, row_index_self, row_index_neighbor, col_index, tx_mask_distance
    
    



