import numpy as np 
import pandas as pd 
import torch 


def load_random_patch(adata, 
                      tx_metadata,
                      half_patch_size_x,
                      half_patch_size_y,
                      layer):
    accept = False 
    while(not accept):
        center_x = np.random.uniform(low=adata.uns['centroid_x_min']-10, 
                                    high=adata.uns['centroid_x_max']+10)
        center_y = np.random.uniform(low=adata.uns['centroid_y_min']-10, 
                                    high=adata.uns['centroid_y_max']+10)    
        left_x = center_x - half_patch_size_x
        right_x = center_x + half_patch_size_x
        bottom_y = center_y - half_patch_size_y
        top_y = center_y + half_patch_size_y
        
        cell_patch = adata.to_df(layer)[(adata.obs['x']>left_x) & (adata.obs['x']<right_x) & \
                            (adata.obs['y']>bottom_y) & (adata.obs['y']<top_y)].copy()  
        if cell_patch.shape[0] > 100:
            accept = True
        
    cell_patch['row_index'] = [i for i in range(cell_patch.shape[0])]
    cell_patch = cell_patch.merge(adata.obs[['leiden']], how='left',
                                  left_index=True, right_index=True)
    
    tx_patch = tx_metadata.loc[(tx_metadata['cell_id'].isin(cell_patch.index)) & \
                                (tx_metadata['neighbor_by_centroid'].isin(cell_patch.index)), :].copy()
    
    tx_patch = tx_patch.merge(adata.var[['col_index']], 
               how="left", left_on="gene", right_index=True)
    tx_patch = tx_patch.merge(cell_patch[['row_index']], 
               how='left', left_on="cell_id", right_index=True).rename(columns={"row_index": "row_index_self"})
    tx_patch = tx_patch.merge(cell_patch[['row_index']], 
               how='left', left_on="neighbor_by_centroid", right_index=True).rename(columns={"row_index": "row_index_neighbor"})
    
    
    true_labels = torch.tensor(cell_patch['leiden'].astype(int).values, dtype=torch.int64).unsqueeze(1)
    true_labels = torch.zeros(cell_patch.shape[0], 40, dtype=torch.float32).scatter(0, true_labels, 1)
    cell_patch.drop(columns=['row_index', "leiden"], inplace=True)
    
    cell_by_gene_counts = torch.tensor(cell_patch.values, dtype=torch.float32)
    
    tx_features = torch.tensor(tx_patch[['distance_ratio', "one_v_one", "one_v_rest"]].values, dtype=torch.float32)
    row_index_self = torch.LongTensor(tx_patch[['row_index_self']].values)
    row_index_neighbor = torch.LongTensor(tx_patch[['row_index_neighbor']].values)
    col_index = torch.LongTensor(tx_patch[['col_index']].values)
    
    return cell_by_gene_counts, tx_features, true_labels, row_index_self, row_index_neighbor, col_index
    
    



