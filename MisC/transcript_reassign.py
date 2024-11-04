# Data IO 
from tqdm import tqdm
import pandas as pd
import scanpy as sc
import geopandas as gpd
# Data manipulation 
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy
# Utility functions 
from utility import annotate_tx_mask_distance, mix_norm_cdf, extract_layer_num
# Typing 
from typing import Tuple, Optional 


def propose_reassignment(adata: sc.AnnData, 
                         tx_metadata: gpd.GeoDataFrame,
                         cell_coords: gpd.GeoDataFrame,
                         layer: str, 
                         mask_distance: pd.DataFrame, 
                         mask_dist_cutoff: float=1, 
                         tx_mask_d_max: float = 2.0,
                         hard_threshold: Optional[float]=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate updates to the cell-by-gene counts matrix and the transcript matrix 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell metainformation
    tx_metadata : gpd.GeoDataFrame
        The detected transcripts
    cell_coords : gpd.GeoDataFrame
        The geodataframe recording the vertices of all the cells 
    layer : str
        The layer upon which the update is computed 
    mask_distance: pd.DataFrame
        A dataframe where each row is a pair of neighboring cells as well as their distance information 
    mask_dist_cutoff : float, optional
        The threshold of cell-cell distances beyond which a cell is no longer considered a neighbor, by default 1
    tx_mask_d_max : float, optional
        The threshold beyond which we do not consider a certain transcript as a membrane transcript, by default 2.0
    hard_threshold : Optional[float], optional
        The threshold on the difference of percentages, by default None

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Updates to counts and transcript matrices. The first two dataframes are "patches" to the current 
        count matrix. The first one is the number of transcripts that should be subtracted from the count while 
        the second one is the number of transcripts to be added to the count. 
        The last two dataframes are "patches" for actual transcript reassignment. The first one shows which 
        transcript should be reassigned to which cell type while the second one records the original assignment.  
    """
    #TODO: Add a progress bar
    print('Identifying transcripts that need to be reassigned...')
    intf_tx = annotate_tx_mask_distance(adata=adata,
                                        tx_metadata=tx_metadata,
                                        cell_coords=cell_coords,
                                        mask_distance=mask_distance, 
                                        mask_dist_cutoff=mask_dist_cutoff)
    # The basic logic for reassigning transcript is based on the difference of the 
    # expressed transcripts between two types of cells 
    # If a type of transcript is lowly expressed in a cell but some transcripts of that type are 
    # present in that cell and the neighboring cell highly express that transcript, we reassign that transcripts 
    # We first compute the percentages of positively expressed genes in each cell type
    s_total = adata.to_df(layer).groupby(adata.obs.leiden, observed=True).count()
    s_above_0 = (adata.to_df(layer)>0).groupby(adata.obs.leiden, observed=True).sum()
    percent_pos = s_above_0 / s_total
    
    # We only consider membrane transcripts 
    intf_tx = intf_tx[intf_tx.tx_mask_distance<tx_mask_d_max].sort_values('tx_mask_distance')
    # Then we compute the differences in the percentages 
    intf_tx['pct_exp_celltype'] = intf_tx.apply(
        lambda x: percent_pos.loc[x['cell_type'],x['gene']], axis=1).values
    intf_tx['pct_exp_nearby'] = intf_tx.apply(
        lambda x: percent_pos.loc[x['neighbor_celltype'],x['gene']], axis=1).values
    intf_tx['pct_diff'] = intf_tx.pct_exp_nearby - intf_tx.pct_exp_celltype
    
    if hard_threshold is None:
        # Do not use 
        # Not completed 
        gmm_dict = {}
        for cell_type0 in percent_pos.index:
            for cell_type1 in percent_pos.index:
                if cell_type0 == cell_type1:
                    continue
                # 0 vs other (if > 0 then gene expressed)
                # only consider > 0
                fold_change = np.log2(percent_pos.loc[cell_type0,:]+1) - np.log2(percent_pos.loc[cell_type1, :]+1)
                model = GMM(2)
                model.fit(fold_change.values.reshape((-1,1)))
                gmm_dict["{}-{}".format(cell_type0, cell_type1)] = model
        intf_tx['remove_prob'] = intf_tx.apply(lambda x: mix_norm_cdf(x['pct_diff'],
                                                gmm_dict["{}-{}".format(x['cell_type'], x['neighbor_celltype'])]), axis=1).values

        intf_tx['reassign'] = np.random.binomial(1, intf_tx['remove_prob'])
    else:
        # If its greater than the threshold, we reassign the transcript 
        intf_tx['reassign'] = (intf_tx['pct_diff']>=hard_threshold).astype(int)
    
    tx_to_reassign = intf_tx[intf_tx['reassign']==1]
    # We only keep the cell that is closest to the transcript 
    tx_to_reassign = tx_to_reassign.groupby(tx_to_reassign.index).first()
    
    # Generate two patches for the count matrix 
    counts_to_subtract = pd.DataFrame(0, index=adata.obs_names, columns=adata.var_names)
    counts_to_add = pd.DataFrame(0, index=adata.obs_names, columns=adata.var_names)
    # For removal, we for each cell count how many genes occured 
    cell_to_remove = tx_to_reassign.groupby(by=['cell_id', "gene"], as_index=False).size()
    # For addition, we for each cell in the neighbor count how many genes occured 
    cell_to_add = tx_to_reassign.groupby(by=['neighbor_by_centroid', "gene"], as_index=False).size()
    cell_to_add.rename(columns={"neighbor_by_centroid": "cell_id"}, inplace=True)
    # Transform the dataframe from long to wide 
    subtract_patch = pd.pivot(cell_to_remove, values="size", columns="gene", index='cell_id').fillna(0)
    add_patch = pd.pivot(cell_to_add, values="size", columns="gene", index='cell_id').fillna(0)
    # The update the find matching rows and columns 
    counts_to_subtract.update(subtract_patch)
    counts_to_add.update(add_patch)
    
    # Finally, record which transcript should be assigned to which cell as well as its original assignment
    tx_assignment_addition = tx_to_reassign[['neighbor_by_centroid', "gene"]].rename(columns={"neighbor_by_centroid": "cell_id"})
    tx_assignment_removal = tx_to_reassign[['cell_id', "gene"]]

    return counts_to_subtract, counts_to_add, tx_assignment_addition, tx_assignment_removal


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
                      tx_assignment_addition: pd.DataFrame, 
                      tx_assignment_removal: pd.DataFrame,
                      test_result: dict) -> Tuple[sc.AnnData, gpd.GeoDataFrame]:
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
    tx_assignment_addition['accept'] = True
    tx_assignment_removal['accept'] = True
    print('Finalize transcript reassignment...')
    for cell_type in tqdm(test_result):
        # Removal is the one to be subtracted from   
        tx_assignment_removal.loc[tx_assignment_removal.cell_id.isin(test_result[cell_type]['contaminated_cell_ids']),
                                "accept"] = False

        # Addition is the one to be added on 
        tx_assignment_addition.loc[tx_assignment_addition.cell_id.isin(test_result[cell_type]['contaminated_cell_ids']),
                                "accept"] = False
    
    tx_assignment_addition = tx_assignment_addition[tx_assignment_addition['accept']]
    tx_assignment_removal = tx_assignment_removal[tx_assignment_removal['accept']]
    
    tx_assignment_addition.drop(columns=['accept'], inplace=True)
    tx_assignment_removal.drop(columns=['accept'], inplace=True)
    
    # Generate two patches for the count matrix 
    counts_to_subtract = pd.DataFrame(0, index=adata.obs_names, columns=adata.var_names)
    counts_to_add = pd.DataFrame(0, index=adata.obs_names, columns=adata.var_names)
    # For removal, we for each cell count how many genes occured 
    cell_to_remove = tx_assignment_removal.groupby(by=['cell_id', "gene"], as_index=False).size()
    # For addition, we for each cell in the neighbor count how many genes occured 
    cell_to_add = tx_assignment_addition.groupby(by=['cell_id', "gene"], as_index=False).size()
    # Transform the dataframe from long to wide 
    subtract_patch = pd.pivot(cell_to_remove, values="size", columns="gene", index='cell_id').fillna(0)
    add_patch = pd.pivot(cell_to_add, values="size", columns="gene", index='cell_id').fillna(0)
    # The update the find matching rows and columns 
    counts_to_subtract.update(subtract_patch)
    counts_to_add.update(add_patch)
    
    layer_num = extract_layer_num(layer)
    # Update adata
    adata.layers["counts_"+str(int(layer_num+1))] = adata.layers[layer]+counts_to_add-counts_to_subtract 
    # Updata transcripts 
    tx_metadata["cell_id_"+str(layer_num)] = tx_metadata['cell_id']
    tx_metadata.update(tx_assignment_addition)
    # No need to update boundary or metadata 
    
    return adata, tx_metadata




