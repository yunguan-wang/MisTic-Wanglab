# Data IO
import scanpy as sc
import pandas as pd
import numpy as np 
import geopandas as gpd
# Data manipulation 
from itertools import combinations
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from shapely import distance
# User entertainment
from tqdm.auto import tqdm
# Typing 
from typing import Tuple


def expression_feature(adata: sc.AnnData,
                        layer: str,
                        num_rep: int=3,
                        method: str="split",
                        n_cpus: int=8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use deseq2 with pseudo bulks to perform differential analysis among different cell types and cell type vs all other cell types 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell metainformation
    layer : str
        Name of the layer. It should be like counts_0, counts_1_proposed_update, etc
    num_rep : int, optional
        Number of repetitions in case of method='bootstrap' or number of chuns in case of method='split', by default 3
    method : str, optional
        The method to generate pseudo bulks, by default "split"
    n_cpus : int, optional
        Number of CPUs to be used for deseq2, by default 8

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Differential analysis results 
    """
    assert method in ['split', 'bootstrap'], "method has to be either split or bootstrap"
    
    # To prepare for one-vs-rest comparison, we first construct an augmented dataframe 
    # where the trailing samples are just the original counts that belong to 
    # all the cells except for one type 
    aug_sample = adata.to_df(layer).copy()
    aug_sample['leiden'] = adata.obs['leiden'].copy()
    aug_sample.reset_index(drop=False, names=['cell_id'],inplace=True)
    for l in tqdm(adata.uns[layer+"_leiden"], desc="Augmenting sample"):
        temp = adata.to_df(layer).loc[adata.obs['leiden']!=l, :].copy()
        temp['leiden'] = "cell_type_m_"+str(l)
        temp.reset_index(drop=False, names=['cell_id'],inplace=True)
        aug_sample = pd.concat([aug_sample, temp], axis=0, join='inner', ignore_index=True)
        
    counts_list = []
    # If the method is split, we will split per each cell type into num_rep nonoverlaping chunks 
    if method == "split":
        # Shuffle the data 
        rep_sample = aug_sample.groupby(['leiden'],
                                        observed=True,
                                        as_index=True).apply(lambda x: x.sample(frac=1, replace=False),
                                                             include_groups=False)
        # The dataframe should only have counts and leiden 
        rep_sample.reset_index(drop=False, names=['leiden', 'id_to_drop'], inplace=True)
        rep_sample.drop(columns=['cell_id','id_to_drop'], inplace=True)
        rep_counts = rep_sample.groupby("leiden", as_index=True).apply(lambda x: np.array_split(x, num_rep), include_groups=False)
        # Construct pseudo bulk by summing up the counts 
        for l in rep_counts.index:
            for rep in range(num_rep):
                temp = rep_counts[l][rep].sum(axis=0).to_frame().T
                temp['leiden'] = l
                counts_list.append(temp)
    else:   
        # If the method is bootstrap, we simply resample the data with replacement num_rep times 
        for _ in range(num_rep):
            rep_sample = aug_sample.groupby(['leiden'],
                                        observed=True,
                                        as_index=True).apply(lambda x: x.sample(frac=1, replace=True),
                                                             include_groups=False)
            rep_sample.reset_index(drop=False, names=['leiden', 'id_to_drop'], inplace=True)
            rep_sample.drop(columns=['cell_id','id_to_drop'], inplace=True)
            # Construct pseudo bulks 
            rep_counts = rep_sample.groupby(by=['leiden'], observed=True, as_index=False).sum()
            counts_list.append(rep_counts)
    # Concatnate list to a dataframe and use the index as "sample" id
    counts_df = pd.concat(counts_list, axis=0).reset_index(drop=True)    
    counts_df["sample_id"] = "Sample"+counts_df.index.astype(str)
    # Treat leiden as different "conditions"
    metadata = counts_df[["sample_id", "leiden"]].set_index("sample_id", inplace=False)
    counts_df.drop(columns=['leiden'], inplace=True)
    counts_df.set_index('sample_id', inplace=True)
    # Deseq2
    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors="leiden",
        refit_cooks=True,
        inference=inference,
    )
    dds.deseq2()
    print("Performing one-vs-one differential analysis...")
    exp_1v1_list = []
    for neighbor_celltype, celltype in tqdm(combinations(adata.uns[layer+"_leiden"], 2), 
                                            total=(adata.uns[layer+"_n_leiden"] * (adata.uns[layer+"_n_leiden"]+1))/2):
        # Since deseq2 will replace _ with -
        n_ct = neighbor_celltype.replace("_", "-")
        ct = celltype.replace("_", "-")
        # Use contrast to estimate log2FC and padj
        stat_res = DeseqStats(dds, 
                              inference=inference, 
                              contrast=['leiden', n_ct, ct],
                              quiet=True)
        stat_res.summary() 
        if np.any(np.isnan(stat_res.results_df['pvalue'])) or np.any(np.isnan(stat_res.results_df['padj'])):
            print("When comparing {} to {}, some pvalues/padjs are NaN. We will fill it with 1.".format(neighbor_celltype, celltype))
            stat_res.results_df['pvalue'].fillna(1.0, inplace=True)
            stat_res.results_df['padj'].fillna(1.0, inplace=True)
        exp_1v1_list.append([celltype, neighbor_celltype, 
                         list(stat_res.results_df.index), # Genes
                         list(stat_res.results_df['log2FoldChange']),
                         list(stat_res.results_df['padj'])])
    # Convert to dataframe and compute -log10(p)
    exp_1v1_df0 = pd.DataFrame(exp_1v1_list, columns=["cell_type", 'neighbor_celltype',
                                            'gene', 'log2FoldChange', 'padj']).reset_index(drop=True)
    exp_1v1_df0 = exp_1v1_df0.explode(column=['gene', 'log2FoldChange', 'padj'])
    exp_1v1_df0['padj'] = -np.log10(exp_1v1_df0['padj'].astype(float)+1e-7)
    exp_1v1_df0['log2FoldChange'] = exp_1v1_df0['log2FoldChange'].astype(float)
    # Want both neighbor - self and self - neighbor 
    exp_1v1_df1 = exp_1v1_df0.copy()
    exp_1v1_df1['cell_type'], exp_1v1_df1['neighbor_celltype'] = exp_1v1_df1['neighbor_celltype'], exp_1v1_df1['cell_type']
    exp_1v1_df1['log2FoldChange'] *= -1
    # Concatenate dataframes 
    exp_1v1_df = pd.concat([exp_1v1_df0, exp_1v1_df1], axis=0).reset_index(drop=True) 
    
    print("Performing one-vs-rest differential analysis...")
    exp_1vR_list = []
    for celltype in tqdm(adata.uns[layer+"_leiden"]):
        # Since deseq2 will replace _ with -
        ct = celltype.replace("_", "-")
        stat_res = DeseqStats(dds, 
                              inference=inference, 
                              contrast=['leiden', "cell-type-m-"+ct, ct],
                              quiet=True)
        stat_res.summary() 
        if np.any(np.isnan(stat_res.results_df['pvalue'])) or np.any(np.isnan(stat_res.results_df['padj'])):
            print("When comparing rest to {}, some pvalues/padjs are NaN. We will fill it with 1.".format(celltype))
            stat_res.results_df['pvalue'].fillna(1.0, inplace=True)
            stat_res.results_df['padj'].fillna(1.0, inplace=True)
        exp_1vR_list.append([celltype, 
                                list(stat_res.results_df.index), 
                                list(stat_res.results_df['log2FoldChange']),
                                list(stat_res.results_df['padj'])]) 
    # Convert to dataframe and compute -log10(p)
    exp_1vR_df = pd.DataFrame(exp_1vR_list, columns=["cell_type", 'gene', 'log2FoldChange', 'padj']).reset_index(drop=True) 
    exp_1vR_df = exp_1vR_df.explode(column=['gene', 'log2FoldChange', 'padj'])
    exp_1vR_df['padj'] = -np.log10(exp_1vR_df['padj'].astype(float)+1e-7)
    exp_1vR_df['log2FoldChange'] = exp_1vR_df['log2FoldChange'].astype(float)

    return exp_1v1_df, exp_1vR_df


def distance_feature(adata: sc.AnnData,
                    tx_metadata: gpd.GeoDataFrame,
                    cell_coords: gpd.GeoDataFrame,
                    mask_distance: pd.DataFrame, 
                    mask_dist_cutoff: float=1) -> gpd.GeoDataFrame:
    """Depending on the cell-cell distance computed based on cell masks, this function computes 
    the distances of all the transcripts of a cell to the nearest neighboring cell and then computes 
    the distance ratio 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell metainformation
    tx_metadata : gpd.GeoDataFrame
        The detected transcripts
    cell_coords : gpd.GeoDataFrame
        The geodataframe recording the vertices of all the cells 
    mask_distance : pd.DataFrame
        The cell-cell distance based on cell masks 
    mask_dist_cutoff : float, optional
        The threshold of cell-cell distances beyond which a cell is no longer considered a neighbor, by default 1

    Returns
    -------
    gpd.GeoDataFrame
        A dataframe containing the distances of all the transcripts to all the neighboring cells 
    """
    # Based on the cell-cell distance computed via cell masks 
    # we find cells (interface cells) that are close to their neighbors (identified via cell centroids)
    # Note that in computing the mask distance, we only included pairs 
    # of cells that of different types 
    interface = mask_distance[mask_distance.mask_distance<=mask_dist_cutoff]
    valid_cells = interface["cell_id"].unique()
    # We extract transcripts of those interface cells 
    intf_tx = tx_metadata[tx_metadata["cell_id"].isin(valid_cells)]
    # Then we compute the distance between transcript and cell mask of neighboring cell
    print("Computing distances among transcripts and neighboring cells...")
    intf_tx = intf_tx.merge(
        interface[["cell_id", 'neighbor_cell_id','mask_distance']], on="cell_id", how='left')
    intf_tx['mask_geom'] = cell_coords.loc[intf_tx.neighbor_cell_id.values, "cell_boundary_geom"].values
    intf_tx['tx_mask_distance'] = distance(intf_tx["tx_geom"], intf_tx['mask_geom'])
    # Get the nearest neighbor 
    intf_tx.sort_values('tx_mask_distance', inplace=True)
    intf_tx = intf_tx.groupby(['molecule_id'], as_index=False).nth(0).reset_index(drop=True)
    
    intf_tx = intf_tx.merge(adata.obs[["leiden","cell_centroid_geom"]], how='left',
                            left_on='cell_id', right_index=True).rename(columns={"leiden": "cell_type",
                                                                                "cell_centroid_geom": "self_centroid_geom"},
                                                                    inplace=False)
    intf_tx = intf_tx.merge(adata.obs[["leiden","cell_centroid_geom"]], how='left',
                            left_on='neighbor_cell_id', right_index=True).rename(columns={"leiden": "neighbor_celltype",
                                                                                         "cell_centroid_geom": "neighbor_centroid_geom"},
                                                                    inplace=False)
    # Compute distance ratio
    intf_tx['self_centroid_distance'] = distance(intf_tx['tx_geom'], intf_tx['self_centroid_geom'])
    intf_tx['neighbor_centroid_distance'] = distance(intf_tx['tx_geom'], intf_tx['neighbor_centroid_geom'])
    intf_tx['distance_ratio'] = intf_tx['self_centroid_distance']/intf_tx['neighbor_centroid_distance']
    intf_tx['distance_ratio'] = np.log2(intf_tx['distance_ratio']) 
    
    return intf_tx



def generate_feature(adata: sc.AnnData,
                    layer: str,
                    tx_metadata: gpd.GeoDataFrame,
                    cell_coords: gpd.GeoDataFrame,
                    mask_distance: pd.DataFrame, 
                    mask_dist_cutoff: float=1,
                    num_rep: int=3,
                    method: str="split",
                    n_cpus: int=8) -> gpd.GeoDataFrame:
    """A wrap up function for expression_feature and distance_feature 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell metainformation
    layer : str
        Name of the layer. It should be like counts_0, counts_1_proposed_update, etc
    tx_metadata : gpd.GeoDataFrame
        The detected transcripts
    cell_coords : gpd.GeoDataFrame
        The geodataframe recording the vertices of all the cells 
    mask_distance : pd.DataFrame
        The cell-cell distance based on cell masks 
    mask_dist_cutoff : float, optional
        The threshold of cell-cell distances beyond which a cell is no longer considered a neighbor, by default 1
    num_rep : int, optional
        Number of repetitions in case of method='bootstrap' or number of chuns in case of method='split', by default 3
    method : str, optional
        The method to generate pseudo bulks, by default "split"
    n_cpus : int, optional
        Number of CPUs to be used for deseq2, by default 8
    Returns
    -------
    gpd.GeoDataFrame
        A dataframe containing information on the transcripts 
    """
    # Features based on distance 
    intf_tx = distance_feature(adata=adata,
                               tx_metadata=tx_metadata,
                               cell_coords=cell_coords,
                               mask_distance=mask_distance,
                               mask_dist_cutoff=mask_dist_cutoff)
    # Features based on differential analysis 
    exp_1v1_df, exp_1vR_df = expression_feature(adata=adata,
                                                layer=layer,
                                                num_rep=num_rep,
                                                method=method,
                                                n_cpus=n_cpus)
    # Combine the two piceces of information 
    intf_tx = intf_tx.merge(exp_1v1_df, how='left', 
                            on=['cell_type', "neighbor_celltype", "gene"])
    intf_tx.rename(columns={"log2FoldChange": "neighbor_self_log2FoldChange",
                            "padj": "neighbor_self_padj"}, inplace=True)
    intf_tx = intf_tx.merge(exp_1vR_df, how='left',
                            on=['cell_type', "gene"])
    intf_tx.rename(columns={"log2FoldChange": "rest_self_log2FoldChange",
                            "padj": "rest_self_padj"}, inplace=True)
    # Combine fc and p so that if fc is large and -log10(p) is large it's more likely to be reassigned 
    intf_tx['neighbor_self_exp_feature'] = intf_tx['neighbor_self_log2FoldChange'] * intf_tx['neighbor_self_padj']
    intf_tx['rest_self_exp_feature'] = intf_tx['rest_self_log2FoldChange'] * intf_tx['rest_self_padj']
    
    return intf_tx 
    