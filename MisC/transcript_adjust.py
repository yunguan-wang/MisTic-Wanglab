import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
from copy import deepcopy
from utility import calculate_mask_distance, annotate_tx_mask_distance, remove_tx


def propose_adjustment(adata,
                       cell_coords,
                       tx_metadata,
                       mask_dist_cutoff=1):
    
    # It is now evident masks can still be very close even when centroid distance is above 40
    mask_distance  = calculate_mask_distance(
        cell_dist, cell_coords, max_centroid_dist=75, re_cal_centroid_dist=True)
    interface = mask_distance[mask_distance.mask_distance<=mask_dist_cutoff]
    intf_tx = annotate_tx_mask_distance(
        interface, tx_metadata, cell_coords)
    
    current_counts = counts.copy()
    current_adata = deepcopy(adata)

    current_cell_coords = cell_coords.copy()

    current_cell_coords.drop(columns=['x','y','leiden'], inplace=True)
    current_cell_coords = current_cell_coords.merge(
        current_adata.obs[['x','y','leiden']], right_index=True, left_index=True, how='left')
    interface = mask_distance[mask_distance.mask_distance<=mask_dist_cutoff]
    current_intf_tx = annotate_tx_mask_distance(
        interface, tx_metadata, current_cell_coords)

    counts_to_subtract, counts_to_add, current_intf_tx = remove_tx(current_counts, current_cell_coords, current_intf_tx,
                                                                   hard_threshold=0.25)
    
    proposed_counts = current_counts.copy()
    proposed_counts.loc[current_counts_update.index,current_counts_update.columns] = current_counts_update.values
    proposed_adata = sc.AnnData(proposed_counts)
    proposed_adata.obs = current_adata.obs.loc[proposed_adata.obs_names]
    proposed_adata.obs['celltype'] = current_adata.obs.loc[proposed_adata.obs_names, 'leiden']
    sc.pp.normalize_total(proposed_adata, target_sum=1000)

    sc.pp.log1p(proposed_adata)
    # np.nan_to_num(proposed_adata.X, nan=0, copy=False)
    proposed_adata.raw = proposed_adata.copy()

    return 0


def test_proposed_adjustment():
    pass 


def adjust_counts():
    pass 











    




