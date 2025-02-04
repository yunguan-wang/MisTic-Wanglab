# Data IO
import scanpy as sc
import pandas as pd
import polars as pl
import numpy as np 
import geopandas as gpd
# Data manipulation 
from itertools import combinations
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import get_num_processes
from shapely import distance
from scipy.spatial import KDTree
# User entertainment
from MisC.utility import process_time_ram
from tqdm.auto import tqdm
# Typing 
from typing import Tuple


def expression_feature(adata: sc.AnnData,
                        layer: str,
                        num_rep: int=3,
                        method: str="split",
                        seed: int=42) -> Tuple[pl.DataFrame, pl.DataFrame]:
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

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Differential analysis results 
    """
    n_cpus = get_num_processes(n_cpus=None)
    np.random.seed(seed)
    # To prepare for one-vs-rest comparison, we first construct an augmented dataframe 
    # where the trailing samples are just the original counts that belong to 
    # all the cells except for one type 
    with process_time_ram("Prepare for DE") as ctm:
        aug_sample = adata.to_df(layer).copy()
        aug_sample['leiden'] = adata.obs['leiden'].copy()
        aug_sample.reset_index(drop=False, names=['cell_id'],inplace=True)
        for l in tqdm(adata.uns["unique_leiden"], desc="Augmenting sample"):
            temp_n = np.min([(adata.obs['leiden']==l).sum(),
                             (adata.obs['leiden']!=l).sum()])
            temp = adata.to_df(layer).loc[adata.obs['leiden']!=l, :].sample(n=temp_n).copy()
            temp['leiden'] = "cell_type_m_"+str(l)
            temp.reset_index(drop=False, names=['cell_id'],inplace=True)
            aug_sample = pd.concat([aug_sample, temp], axis=0, join='inner', ignore_index=True)
    
    with process_time_ram("Generate psdudo-bulk") as ctm:
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
    with process_time_ram("Performing one-vs-one differential analysis...") as ctm:
        exp_1v1_list = []
        for neighbor_celltype, celltype in tqdm(combinations(adata.uns["unique_leiden"], 2), 
                                                total=(adata.uns["n_leiden"] * (adata.uns["n_leiden"]-1))/2):
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
    
    with process_time_ram("Performing one-vs-rest differential analysis...") as ctm:
        exp_1vR_list = []
        for celltype in tqdm(adata.uns["unique_leiden"]):
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
    
    with process_time_ram("Generate DE features") as ctm:
        # We multiply the two quantities 
        exp_1v1_df['neighbor_self_exp_feature'] = exp_1v1_df['log2FoldChange'] * exp_1v1_df['padj']
        exp_1vR_df['rest_self_exp_feature'] = exp_1vR_df['log2FoldChange'] * exp_1vR_df['padj']
        # To avoid extreme values, we rank-transform the data and compute the log "odds"
        exp_1v1_df['neighbor_self_exp_feature'] = exp_1v1_df['neighbor_self_exp_feature'].rank(pct=True)*0.999
        exp_1v1_df['neighbor_self_exp_feature'] = np.log(exp_1v1_df['neighbor_self_exp_feature']/(1-exp_1v1_df['neighbor_self_exp_feature']))
        
        exp_1vR_df['rest_self_exp_feature'] = exp_1vR_df['rest_self_exp_feature'].rank(pct=True)*0.999
        exp_1vR_df['rest_self_exp_feature'] = np.log(exp_1vR_df['rest_self_exp_feature']/(1-exp_1vR_df['rest_self_exp_feature']))
        
        exp_1v1_df.drop(columns=['log2FoldChange', 'padj'], inplace=True)
        exp_1vR_df.drop(columns=['log2FoldChange', 'padj'], inplace=True)
    
    exp_1v1_df = pl.from_pandas(exp_1v1_df)
    exp_1vR_df = pl.from_pandas(exp_1vR_df)
    
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
    with process_time_ram("Finding cells in the interface") as ctm:
        interface = mask_distance.loc[mask_distance.mask_distance<=mask_dist_cutoff, ['cell_id', 'neighbor_cell_id']]
        interface = interface.merge(adata.obs[["leiden"]], how='left',
                        left_on='cell_id', right_index=True).rename(columns={"leiden": "cell_type"},
                                                                        inplace=False)
        interface = interface.merge(adata.obs[["leiden"]], how='left',
                                left_on='neighbor_cell_id', right_index=True).rename(columns={"leiden": "neighbor_celltype"},
                                                                        inplace=False)
        interface = pl.from_pandas(interface)
        valid_cells = interface["cell_id"].unique()
    ########################################################
    ########################################################
    with process_time_ram("Extracting transcripts") as ctm:
        # Directly indexing a huge pandas df would take a long time 
        # We switch to polars to speed it up
        intf_tx = pl.from_pandas(tx_metadata[['molecule_id', 'cell_id', "gene", "global_x", "global_y"]])
        intf_tx = intf_tx.filter(pl.col("cell_id").is_in(valid_cells))
    ########################################################
    ########################################################
    with process_time_ram("Find transcripts' NNs based on cell centroids") as ctm:
        # Again we use KDTree to query to >3NNs
        # The logic is that point to mask distance computation takes a long time 
        # and since we are using ranks, centroid distances would be a okish substitute 
        # The choice of >3 is that 1 of them would be itself 
        # the other two, one might be of the same type the other might be of a different type 
        kn = 5
        centroid_tree = KDTree(adata.obs[['x', 'y']])
        dd, ii = centroid_tree.query(intf_tx[["global_x", "global_y"]], k=kn, workers=-1)
    ########################################################
    ########################################################
    with process_time_ram("Add NN info") as ctm:
        intf_tx = intf_tx.drop(["global_x", "global_y"])
        intf_tx = intf_tx.select(pl.all().repeat_by(kn).flatten())
        
        intf_tx = intf_tx.with_columns(pl.Series(name = "neighbor_cell_id", values=adata.obs_names[ii.ravel()].values))
        intf_tx = intf_tx.with_columns(pl.Series(name = "neighbor_distance", values=dd.ravel()))
        
        intf_tx = intf_tx.filter(pl.col("cell_id")!=pl.col("neighbor_cell_id"))
        
        intf_tx = intf_tx.join(interface, how='left', on=["cell_id", "neighbor_cell_id"])
        intf_tx = intf_tx.drop_nulls()
    ########################################################
    ########################################################
    with process_time_ram("Computing overall min") as ctm:
        min_distance = intf_tx.group_by(['molecule_id']).agg(pl.min("neighbor_distance"))
        min_distance = min_distance.rename({"neighbor_distance": "min_neighbor_distance"})
    ########################################################
    ########################################################
    with process_time_ram("Computing min with different cell type") as ctm:
        intf_tx = intf_tx.filter(pl.col("cell_type")!=pl.col("neighbor_celltype"))
        
        intf_tx = intf_tx.sort("neighbor_distance")
        
        intf_tx = intf_tx.group_by(["molecule_id"], maintain_order=True).first()
        intf_tx = intf_tx.join(min_distance, how='left', on=['molecule_id'])
    ########################################################
    ########################################################
    with process_time_ram("Computing percentile ranks by cell") as ctm:
        intf_tx = intf_tx.with_columns(
            (pl.col("neighbor_distance").rank(descending=False)/pl.col("neighbor_distance").count()*0.999)
            .over("cell_id").alias("neighbor_distance_rank"))
        
        intf_tx = intf_tx.with_columns(
            (pl.col("min_neighbor_distance").rank(descending=False)/pl.col("min_neighbor_distance").count()*0.999)
            .over("cell_id").alias("min_neighbor_distance_rank"))
    ########################################################
    ########################################################
    with process_time_ram("Refine overall min rank") as ctm:
        min_distance = intf_tx.filter(pl.col("min_neighbor_distance_rank")<=0.05).select(['molecule_id', 'cell_id'])
        
        min_distance = min_distance.join(interface, how='left', on='cell_id')
        
        min_distance = min_distance.with_columns(
            pl.Series(name="neighbor_distance", values=distance(tx_metadata.loc[min_distance['molecule_id'].to_list(),'tx_geom'].values,
                                                                cell_coords.loc[min_distance['neighbor_cell_id'].to_list(),"cell_boundary_geom"].values))
        )
        patch = min_distance.group_by(['molecule_id', "cell_id"]).agg(pl.min("neighbor_distance"))
        patch = patch.rename({"neighbor_distance": "min_neighbor_distance"})
        patch = patch.with_columns(
            (pl.col("min_neighbor_distance").rank(descending=False)/pl.col("min_neighbor_distance").count()*0.05)
            .over("cell_id").alias("min_neighbor_distance_rank"))
        intf_tx = intf_tx.update(patch, on=["molecule_id", "cell_id"])
    ########################################################
    ########################################################
    with process_time_ram("Refine overall min with different type rank") as ctm:
        min_distance = intf_tx.filter(pl.col("neighbor_distance_rank")<=0.05).select(['molecule_id', 'cell_id'])
        
        min_distance = min_distance.join(interface, how='left', on='cell_id')
        
        min_distance = min_distance.filter(pl.col("cell_type")!=pl.col("neighbor_celltype"))
        
        min_distance = min_distance.with_columns(
            pl.Series(name="neighbor_distance", values=distance(tx_metadata.loc[min_distance['molecule_id'].to_list(),'tx_geom'].values,
                                                                cell_coords.loc[min_distance['neighbor_cell_id'].to_list(),"cell_boundary_geom"].values))
        )
        patch = min_distance.group_by(['molecule_id', "cell_id"]).agg(pl.min("neighbor_distance"))
        patch = patch.with_columns(
            (pl.col("neighbor_distance").rank(descending=False)/pl.col("neighbor_distance").count()*0.05)
            .over("cell_id").alias("neighbor_distance_rank"))
        intf_tx = intf_tx.update(patch, on=["molecule_id", "cell_id"])
    ########################################################
    ########################################################
    with process_time_ram("Generate final features") as ctm:
        intf_tx = intf_tx.with_columns(
            pl.Series(name='distance_feature', values=-np.log(intf_tx['neighbor_distance_rank']/(1-intf_tx['neighbor_distance_rank']))),
            pl.Series(name='prior_distance_feature', values=-np.log(intf_tx['min_neighbor_distance_rank']/(1-intf_tx['min_neighbor_distance_rank']))))
        
        intf_tx = intf_tx.drop(["neighbor_distance","min_neighbor_distance", 
                                "neighbor_distance_rank", "min_neighbor_distance_rank"])
    
    return intf_tx



def generate_feature(adata: sc.AnnData,
                    layer: str,
                    tx_metadata: gpd.GeoDataFrame,
                    cell_coords: gpd.GeoDataFrame,
                    mask_distance: pd.DataFrame, 
                    mask_dist_cutoff: float=1,
                    num_rep: int=3,
                    method: str="split",
                    seed: int=42) -> pl.DataFrame:
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
                                                seed=seed)
    # Combine the two piceces of information 
    intf_tx = intf_tx.join(exp_1v1_df, how='left', 
                            on=['cell_type', "neighbor_celltype", "gene"])
    intf_tx = intf_tx.join(exp_1vR_df, how='left',
                            on=['cell_type', "gene"])
    
    return intf_tx 
    