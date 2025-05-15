# Data IO
import scanpy as sc
import pandas as pd
import polars as pl
import numpy as np 
import geopandas as gpd
# Data manipulation 
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import get_num_processes
from shapely import distance
from scipy.spatial import KDTree
# User entertainment
from MisTIC.utility import process_time_ram, even_split
from tqdm.auto import tqdm
# Typing 
from typing import Tuple


def expression_feature(adata: sc.AnnData,
                        layer: str,
                        seed: int=42,
                        max_de_cells: int=100000) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Use deseq2 with pseudo bulks to perform differential analysis among different cell types and cell type vs all other cell types 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell metainformation
    layer : str
        Name of the layer. It should be like counts_0, counts_1_proposed_update, etc
    seed: int, optional 
        For reproducibitlity, by default 42
    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        Differential analysis results 
    """
    n_cpus = get_num_processes(n_cpus=None)
    np.random.seed(seed)
    num_rep = 3
    # To prepare for one-vs-rest comparison, we first construct an augmented dataframe 
    # where the trailing samples are just the original counts that belong to 
    # all the cells except for one type 
    counts_df = []
    for l in tqdm(adata.uns["unique_leiden"], desc="Prepare for DE"):
        with process_time_ram("Generate pseudo-bulk for {}".format(l)) as ctm:
            # For each cell type 
            cell_ind = adata.obs.loc[adata.obs['leiden']==l,:].index
            rest_ind = adata.obs.loc[adata.obs['leiden']!=l,:].index
            temp_n = np.min([len(cell_ind), len(rest_ind), max_de_cells])
            # Subsample to equal sample size 
            cell_ind = np.random.choice(cell_ind, size=temp_n, replace=False)
            rest_ind = np.random.choice(rest_ind, size=temp_n, replace=False)
            # Extract corresponding gene counts 
            aug_sample = adata[cell_ind].to_df(layer).copy()
            temp = adata[rest_ind].to_df(layer).copy()
            # Add cell type info
            aug_sample['leiden'] = l
            temp['leiden'] = "cell_type_m_"+str(l)
            aug_sample.reset_index(drop=False, names=['cell_id'],inplace=True) 
            temp.reset_index(drop=False, names=['cell_id'],inplace=True)
            aug_sample = pd.concat([aug_sample, temp], axis=0, join='inner', ignore_index=True)
            # split per each cell type into num_rep nonoverlaping chunks 
            # Shuffle the data 
            aug_sample = aug_sample.groupby(['leiden'],
                                            observed=True,
                                            as_index=True).apply(lambda x: x.sample(frac=1, replace=False),
                                                                include_groups=False)
            # The dataframe should only have counts and leiden 
            aug_sample.reset_index(drop=False, names=['leiden', 'id_to_drop'], inplace=True)
            aug_sample.drop(columns=['cell_id','id_to_drop'], inplace=True)
            aug_sample = aug_sample.groupby("leiden", as_index=True).apply(lambda x: np.array_split(x, num_rep), include_groups=False)
            # Construct pseudo bulk by summing up the counts 
            for ind in aug_sample.index:
                for rep in range(num_rep):
                    temp = aug_sample[ind][rep].sum(axis=0).to_frame().T
                    temp['leiden'] = ind
                    counts_df.append(temp)
    # Concatnate list to a dataframe and use the index as "sample" id
    counts_df = pd.concat(counts_df, axis=0).reset_index(drop=True)    
    counts_df["sample_id"] = "Sample"+counts_df.index.astype(str)

    with process_time_ram("Deseq") as ctm:
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
    
    with process_time_ram("Performing one-vs-rest differential analysis...") as ctm:
        exp_1vR_list = []
        for celltype in tqdm(adata.uns["unique_leiden"]):
            # Since deseq2 will replace _ with -
            ct = celltype.replace("_", "-")
            # In the contrast, the test is the rest, the reference is the cell type 
            # Therefore, a negative log2fc means that the cell HIGHLY express the gene
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
        exp_1vR_df['rest_self_exp_feature'] = exp_1vR_df['log2FoldChange'] * exp_1vR_df['padj']
        # To avoid extreme values, we rank-transform the data and compute the log "odds"
        # The smaller the value (the higher a gene is expressed) the lower the rank (close to 0)
        exp_1vR_df['rest_self_exp_feature'] = exp_1vR_df.groupby(by=["cell_type"], as_index=False)['rest_self_exp_feature'].rank(pct=True)*0.999
        exp_1vR_df['rest_self_exp_feature'] = np.log(exp_1vR_df['rest_self_exp_feature']/(1-exp_1vR_df['rest_self_exp_feature']))
        
        exp_1vR_df.drop(columns=['log2FoldChange', 'padj'], inplace=True)
        exp_1vR_df = pl.from_pandas(exp_1vR_df)
    ##################
    # Perform prior DE 
    with process_time_ram("Prepare for prior DE") as ctm:
        pl_adata = pl.from_pandas(adata.to_df(layer), include_index=True)
        # Without the celltype information, we first compute the mean of all genes 
        overall_log2_mean = np.log2(adata.to_df(layer).mean(axis=0) + 1).to_frame().T
    with process_time_ram("Find NNs") as ctm:
        # Based on the DR results, we find the 50NNs
        expression_dist_tree = KDTree(adata.obsm['X_'+adata.uns['dr_method']+'_'+layer])
        _, adj_ind = expression_dist_tree.query(adata.obsm['X_'+adata.uns['dr_method']+'_'+layer], k=50, workers=-1)
        # Map index back to cell ids 
        adj = pl.from_numpy(adj_ind).with_columns(pl.Series(name="cell_id",
                                                    values=adata.obs_names)).unpivot(index="cell_id").drop("variable")
        adj = adj.with_columns(pl.Series(name="n",
                            values=adata.obs_names[adj['value']])).drop('value')
        # Split cells into small chunks 
        cell_id_list = even_split(adata.obs.index.to_numpy(), 1000)
    prior_exp_1vR_df = []
    for counter, cell_ids in enumerate(tqdm(cell_id_list)):
        with process_time_ram("Add count information: chunk "+str(counter)) as ctm:
            sub_adj = adj.filter(pl.col("cell_id").is_in(cell_ids)).join(pl_adata, how='left', left_on="n", right_on='cell_id').drop("n")
        with process_time_ram("Compute mean: chunk "+str(counter)) as ctm:
            gene_col = [col_n for col_n in sub_adj.columns if col_n != "cell_id"]
            sub_adj = sub_adj.group_by('cell_id').mean().with_columns(pl.col(gene) + 1 for gene in gene_col)
        with process_time_ram("Generate Features: chunk "+str(counter)) as ctm:
            # Find log2 fold changes
            log2fc = sub_adj.with_columns(pl.col(gene).log(base=2) - overall_log2_mean[gene] for gene in gene_col)
            # Generate Features 
            log2fc_rank = log2fc.select(["cell_id",
                            pl.concat_list(pl.exclude("cell_id"))
                            .list.eval(pl.element().rank(descending=True)/pl.element().count()*0.999).alias("rank").list.to_struct(
                                fields=gene_col,
                                n_field_strategy="max_width")]).unnest("rank")
            
            log2fc_feature = log2fc_rank.with_columns(pl.col(gene).log()-(1-pl.col(gene)).log() for gene in gene_col)
        with process_time_ram("Wide to long form : chunk "+str(counter)) as ctm:
            temp = log2fc_feature.unpivot(index="cell_id")
            temp = temp.rename({"value": "prior_rest_self_exp_feature",
                                "variable": 'gene'})
            prior_exp_1vR_df.append(temp)
    prior_exp_1vR_df = pl.concat(prior_exp_1vR_df, how='vertical')

    return exp_1vR_df, prior_exp_1vR_df


def distance_feature(adata: sc.AnnData,
                    tx_metadata: gpd.GeoDataFrame,
                    cell_coords: gpd.GeoDataFrame,
                    mask_distance: pd.DataFrame, 
                    mask_dist_cutoff: float=5,
                    nearest: int=3) -> pl.DataFrame:
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
        The threshold of cell-cell distances beyond which a cell is no longer considered a neighbor, by default 5
    nearest: int, optional 
        The number of nearest neighbors to consider, by default 3
        
    Returns
    -------
    dict
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
        # repeat each row kn times 
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
        
        intf_tx = intf_tx.sort("molecule_id", "neighbor_distance").with_columns(pl.int_range(pl.len()).over("molecule_id"))
        intf_tx = intf_tx.rename({'literal': "neighbor_index"})
        intf_tx = intf_tx.filter(pl.col("neighbor_index")<nearest)
        
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
    with process_time_ram("Refine overall min with different type rank") as ctm:
        min_distance = intf_tx.filter(pl.col("neighbor_distance_rank")<=0.9).select(['molecule_id', 'cell_id', 'neighbor_cell_id'])
        
        min_distance = min_distance.with_columns(
            pl.Series(name="neighbor_distance", values=distance(tx_metadata.loc[min_distance['molecule_id'].to_list(),'tx_geom'].values,
                                                                cell_coords.loc[min_distance['neighbor_cell_id'].to_list(),"cell_boundary_geom"].values))
        )
        min_distance = min_distance.sort("molecule_id", "neighbor_distance").with_columns(pl.int_range(pl.len()).over("molecule_id"))
        min_distance = min_distance.rename({'literal': "neighbor_index"})
        
        patch = min_distance.with_columns(
            (pl.col("neighbor_distance").rank(descending=False)/pl.col("neighbor_distance").count()*0.9)
            .over("cell_id").alias("neighbor_distance_rank"))
        intf_tx = intf_tx.update(patch, on=["molecule_id", "cell_id", "neighbor_cell_id"])
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
                    mask_dist_cutoff: float=5,
                    nearest: int=3,
                    seed: int=42,
                    max_de_cells: int=999999) -> pl.DataFrame:
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
        The threshold of cell-cell distances beyond which a cell is no longer considered a neighbor, by default 5
    nearest: int, optional 
        The number of nearest neighbors to consider, by default 3
    seed: int, optional 
        For reproducibitlity, by default 42
    Returns
    -------
    pl.DataFrame
        A dataframe containing information on the transcripts 
    """
    # Features based on distance 
    intf_tx = distance_feature(adata=adata,
                                tx_metadata=tx_metadata,
                                cell_coords=cell_coords,
                                mask_distance=mask_distance,
                                mask_dist_cutoff=mask_dist_cutoff,
                                nearest=nearest)
    # Features based on differential analysis 
    exp_1vR_df, prior_exp_1vR_df = expression_feature(adata=adata,
                                                    layer=layer,
                                                    seed=seed,
                                                    max_de_cells=max_de_cells)
    with process_time_ram("Combine features") as ctm:
        # Combine the two piceces of information 
        intf_tx = intf_tx.join(exp_1vR_df, how='left',
                                on=['cell_type', "gene"])
        intf_tx = intf_tx.join(prior_exp_1vR_df, how='left',
                            on=['cell_id', "gene"])
        # Rename columns to differentiate neighbor from self 
        intf_tx = intf_tx.join(exp_1vR_df.rename({"cell_type":"neighbor_celltype",
                                "rest_self_exp_feature":"neighbor_rest_self_exp_feature"}), how='left',
                                on=['neighbor_celltype', "gene"])
        intf_tx = intf_tx.with_columns(pl.col("neighbor_rest_self_exp_feature").neg())
        
        intf_tx = intf_tx.join(prior_exp_1vR_df.rename({"prior_rest_self_exp_feature": "prior_neighbor_rest_self_exp_feature",
                            "cell_id": "neighbor_cell_id"}), how='left',
                            on=['neighbor_cell_id', "gene"])
        intf_tx = intf_tx.with_columns(pl.col("prior_neighbor_rest_self_exp_feature").neg())
        # Rename the features 
        intf_tx = intf_tx.rename({"neighbor_rest_self_exp_feature": "neighbor_exp_feature",
                                "prior_neighbor_rest_self_exp_feature": "prior_neighbor_exp_feature",
                                "rest_self_exp_feature": "exp_feature",
                                "prior_rest_self_exp_feature": "prior_exp_feature"})
    
    return intf_tx
    