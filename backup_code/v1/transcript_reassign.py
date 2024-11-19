# Data IO 
from tqdm.auto import tqdm
import pandas as pd
import scanpy as sc
import geopandas as gpd
# Data manipulation 
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy
# Utility functions 
from MisC.generate_tx_feature import generate_feature
from MisC.utility import  extract_layer_num, generate_count_patches
# Typing 
from typing import Tuple, Optional 


# def propose_reassignment(adata: sc.AnnData, 
#                          tx_metadata: gpd.GeoDataFrame,
#                          cell_coords: gpd.GeoDataFrame,
#                          layer: str, 
#                          mask_distance: pd.DataFrame, 
#                          mask_dist_cutoff: float=1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """Generate updates to the cell-by-gene counts matrix and the transcript matrix 

#     Parameters
#     ----------
#     adata : sc.AnnData
#         The AnnData object containing cell metainformation
#     tx_metadata : gpd.GeoDataFrame
#         The detected transcripts
#     cell_coords : gpd.GeoDataFrame
#         The geodataframe recording the vertices of all the cells 
#     layer : str
#         The layer upon which the update is computed 
#     mask_distance: pd.DataFrame
#         A dataframe where each row is a pair of neighboring cells as well as their distance information 
#     mask_dist_cutoff : float, optional
#         The threshold of cell-cell distances beyond which a cell is no longer considered a neighbor, by default 1

#     Returns
#     -------
#     Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
#         Updates to counts and transcript matrices. The first two dataframes are "patches" to the current 
#         count matrix. The first one is the number of transcripts that should be subtracted from the count while 
#         the second one is the number of transcripts to be added to the count. 
#         The last two dataframes are "patches" for actual transcript reassignment. The first one shows which 
#         transcript should be reassigned to which cell type while the second one records the original assignment.  
#     """
#     #TODO: Add a progress bar
#     # print('Identifying transcripts that need to be reassigned...')
    
    
    
    
#     # intf_tx['reassign_logit'] = intf_tx['distance_ratio'] + intf_tx['log2FoldChange'] * intf_tx['padj']
    
#     # intf_tx["reassign_prob"] = 1/(1+np.exp(-intf_tx['reassign_logit']))
#     # intf_tx['reassign'] = (intf_tx['reassign_prob']>0.9)
    
#     # tx_to_reassign = intf_tx.loc[intf_tx['reassign']==1, ['molecule_id', 'cell_id', 'neighbor_by_centroid', "gene"]]
    
#     # counts_to_subtract, counts_to_add = generate_count_patches(adata=adata,
#     #                                                            tx_to_reassign=tx_to_reassign)
    

#     # return counts_to_subtract, counts_to_add, tx_to_reassign
#     pass         


def test_proposed_reassignment(adata: sc.AnnData,
                               layer: str,
                               counts_to_subtract: pd.DataFrame,
                               counts_to_add: pd.DataFrame) -> dict:
    """Given the two "patches", this function tests which portion of the patches should be accepted 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell metainformation
    layer : str
        The layer upon which the update is computed 
    counts_to_subtract : pd.DataFrame
        The number of transcripts that should be subtracted from the count
    counts_to_add : pd.DataFrame
        The number of transcripts that should be subtracted from the count

    Returns
    -------
    dict
        A dictionary recording for each cell type the ids of cells that become purer and those that become contaminated 
    """
    # We first tentatively accept the changes and record which cells have updates and which have not 
    adata.layers[layer+"_proposed_update"] = adata.layers[layer]+counts_to_add-counts_to_subtract 
    index_w_updates = (counts_to_add.sum(axis=1)>0) | (counts_to_subtract.sum(axis=1)>0)
    index_wo_updates = (counts_to_add.sum(axis=1)==0) & (counts_to_subtract.sum(axis=1)==0)
    
    cell_types = np.unique(adata.obs['leiden'])
    test_result = {cell_type: {} for cell_type in cell_types}
    print('Testing transcript reassignment...')
    for cell_type in tqdm(cell_types):
        # For each cell type, we will perform a one-vs-other binary classification 
        # in cells without updates 
        index_other_type = (adata.obs['leiden'] != cell_type) & index_wo_updates
        index_other_type = index_other_type[index_other_type].index
        # Extract the counts and the cell labels and rename types that are not cell_type to other 
        x_train = adata.to_df(layer).loc[index_wo_updates, :]
        y_train = adata.obs.loc[index_wo_updates, ['leiden']].astype(str)
        y_train.loc[index_other_type, "leiden"] = "other"
        classifier = LogisticRegression(max_iter=10000, class_weight='balanced')
        classifier.fit(x_train, y_train['leiden'].values)
        # Use the fitted model to compute the probability of cells of the same cell type 
        # but have updates 
        index_cell_type = (adata.obs['leiden'] == cell_type) & index_w_updates
        x_test_original = adata.to_df(layer).loc[index_cell_type, :]
        x_test_update = adata.to_df(layer+"_proposed_update").loc[index_cell_type, :]
        
        prediction_original = classifier.predict_proba(x_test_original)
        prediction_udpate = classifier.predict_proba(x_test_update)
        # We use entropy to measure the purity 
        entropy_original = pd.DataFrame(
            entropy(prediction_original, axis=1),
            columns=['entropy'],
            index=x_test_original.index
        )
        entropy_update = pd.DataFrame(
            entropy(prediction_udpate, axis=1),
            columns=['entropy'],
            index=x_test_update.index
        )
        # Those with decreased entropies are purer otherwise they are contaminated 
        purer_cell_ids = x_test_original.index[entropy_update['entropy'] < entropy_original['entropy']]
        contaminated_cell_ids = x_test_original.index[entropy_update['entropy'] >= entropy_original['entropy']]
        
        test_result[cell_type] = {"purer_cell_ids": purer_cell_ids,
                                  "contaminated_cell_ids": contaminated_cell_ids}

    return test_result


def make_reassignment(adata: sc.AnnData,
                      layer: str,
                      tx_metadata: gpd.GeoDataFrame,
                      tx_to_reassign: pd.DataFrame,
                      test_result: Optional[dict]=None) -> Tuple[sc.AnnData, gpd.GeoDataFrame]:
    """Given the testing results, this function makes the actual reassignment and adjustment 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell metainformation
    layer : str
        The layer upon which the update is computed 
    tx_metadata : gpd.GeoDataFrame
        The detected transcripts
    tx_assignment_addition : pd.DataFrame
        Transcripts that should be reassigned to which cell type
    tx_assignment_removal : pd.DataFrame
        The original assignment
    test_result : dict
        A dictionary of the testing results 

    Returns
    -------
    Tuple[sc.AnnData, gpd.GeoDataFrame]
        The adjusted adata and tx_metadata 
    """
    
    tx_to_reassign['accept'] = True

    print('Finalize transcript reassignment...')
    if test_result is not None:
        for cell_type in tqdm(test_result):
            tx_to_reassign.loc[tx_to_reassign.cell_id.isin(test_result[cell_type]['contaminated_cell_ids']),
                                    "accept"] = False
    
    tx_to_reassign = tx_to_reassign[tx_to_reassign['accept']]
    tx_to_reassign.drop(columns=['accept'], inplace=True)

    counts_to_subtract, counts_to_add = generate_count_patches(adata=adata,
                                                               tx_to_reassign=tx_to_reassign)
    
    layer_num = extract_layer_num(layer)
    # Update adata
    adata.layers["counts_"+str(int(layer_num+1))] = adata.layers[layer]+counts_to_add-counts_to_subtract 
    # Updata transcripts 
    tx_to_reassign.loc[:, ['cell_id', 'neighbor_cell_id']] = tx_to_reassign.loc[:, ['neighbor_cell_id', 'cell_id']]
    tx_to_reassign.index = tx_to_reassign['molecule_id']
    tx_metadata["cell_id_"+str(layer_num)] = tx_metadata['cell_id']
    tx_metadata.update(tx_to_reassign)
    # No need to update boundary or metadata 
    
    return adata, tx_metadata




