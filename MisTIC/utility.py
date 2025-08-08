# Data IO
import os 
import json 
import scanpy as sc
import geopandas as gpd
from geopandas import read_parquet
import pandas as pd
import polars as pl
# Data manipulation 
import re 
import numpy as np
from scipy.special import betainc
from scipy.spatial import KDTree
from scipy.signal import find_peaks, peak_widths
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely import distance
# User entertainment
from tqdm.auto import tqdm
# Typing and other info
from typing import Optional, Union, Tuple, Callable
from time import perf_counter
from contextlib import contextmanager
import psutil


@contextmanager
def process_time_ram(message: str=""):
    """Monitor time taken as well as memory usage

    Parameters
    ----------
    message : str, optional
        The message to be printed, by default ""
    """
    print("="*30)
    print(message)
    t1 = t2 = perf_counter()
    yield lambda: t2-t1
    t2 = perf_counter()
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print("Done. Time taken is {0:.4f} seconds.".format(t2-t1))
    print("="*30)


def process_adata(adata: sc.AnnData,
                layer: str,
                dr_method: str) -> sc.AnnData:
    """Generate UMAP embedding for an AnnData 

    Parameters
    ----------
    adata : sc.AnnData
        An AnnData to be updated 
    layer : str
        The layer upon which UMAP is computed 
    dr_method: str
        The dimension reduction method to be used. Either pca or umap
    Returns
    -------
    sc.AnnData
        AnnData with updated infomation 
    """
    # Make sure X is the counts 
    # Then for all subsequent processing, we can just proceed with the default 
    adata.X = adata.layers[layer].copy()
    if "log1p" in adata.uns:
        del adata.uns['log1p']
    sc.pp.normalize_total(adata, target_sum=1000)
    sc.pp.log1p(adata)
    # We also perform basic visualization 
    sc.pp.scale(adata)
    if dr_method == "umap":
        with process_time_ram("Computing UMAP") as ctm:
            sc.pp.pca(adata)
            sc.pp.neighbors(adata)    
            sc.tl.umap(adata)
    else:
        with process_time_ram("Computing PCA") as ctm:
            sc.pp.pca(adata, n_comps=2)
            sc.pp.neighbors(adata) 
    # Save the embedding to its own key
    # Note that X_umap always refers to the latest one 
    # Can use sc.pl.embedding(adata, basis="X_umap_???", color="cell_type") to plot specific embedding 
    adata.obsm['X_'+dr_method+'_'+layer] = adata.obsm["X_"+dr_method].copy()
    return adata


def import_data(cell_metadata: Union[str, pd.DataFrame],
                cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                detected_transcripts: Union[str, pd.DataFrame, gpd.GeoDataFrame],
                cell_by_gene_counts: Optional[Union[str, pd.DataFrame]]=None,
                cell_centroid_x_col: str='center_x',
                cell_centroid_y_col: str='center_y',
                celltype_col: Optional[str]=None,
                tx_x_col: str='global_x',
                tx_y_col: str='global_y',
                gene_col: str='gene',
                cell_col: str='cell_id',
                leiden_res: float=1,
                dr_method: str="umap"
                ) -> Tuple[sc.AnnData, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Read in the three pieces information needed for subsequent analysis: 
    cell metadata, cell boundary information, transcript information, and 
    optionally cell-by-gene matrix. Some data curation will be performed 

    Parameters
    ----------
    cell_metadata : Union[str, pd.DataFrame]
        Either the path to the csv file or a pandas dataframe containing the metadata for each cell whose 
        first column is assumed to contain the ID for each cell. The information in the file/object should 
        at least contain the xy coordinates of the centers of cells.
        If the users have already performed cell typing which is not required,
        the information can be stored here as a separate column.
    cell_boundary_polygons : Union[str, gpd.GeoDataFrame]
        Either the path to the parquet file or the geopandas GeoDataFrame containing the vertex information 
        for each cell. The first column is assumed to be the IDs for cells. It should contain one column 
        that records the coordinates of the vertices.
    detected_transcripts : Union[str, pd.DataFrame, gpd.GeoDataFrame]
        Either the path to the csv file or a pandas dataframe or geopandas GeoDataFrame containing the information of the detected transcripts.
        The first column is assumed to be some index. The information should contain the ID for a transcripte/molecule, 
        the ID of the cell it belongs to, its xy coordinate, and its gene information. 
    cell_by_gene_counts : Optional[Union[str, pd.DataFrame]], optional
        Either the path to the csv file or a pandas dataframe containing the cell-by-gene count matrix whose first 
        column is assumed to contain the ID for each cell while the rest of the columns should be the transcript counts
        for each cell. If this is not provided, the cell-by-gene matrix will be constructed from the detected_transcripts.
        If provided, the users are responsible for ensuring that the counts correspond to what's recorded in detected_transcripts , by default None
    cell_centroid_x_col : str, optional
        The column containing the x coordinates of cell centroids in cell metadata file, by default 'center_x'
    cell_centroid_y_col : str, optional
        The column containing the y coordinates of cell centroids in cell metadata file, by default 'center_y'
    celltype_col : Optional[str], optional
        The column containing the cell type information in cell metadata file, by default None
    tx_x_col : str, optional
        The column containing the x coordinates of transcript in detected transcript file, by default 'global_x'
    tx_y_col : str, optional
        The column containing the y coordinates of transcript in detected transcript file, by default 'global_y'
    gene_col : str, optional
        The column containing the gene information of transcript in detected transcript file, by default 'gene'
    cell_col : str, optional
        The column containing the cell id of transcript in detected transcript file, by default 'cell_id'
    leiden_res : float, optional
        The resolution for leiden clustering, by default 1
    dr_method : str, optional
        The dimension reduction method to be used. Either pca or umap, by default "umap"

    Returns
    -------
    Tuple[sc.AnnData, gpd.GeoDataFrame, gpd.GeoDataFrame]
        An AnnData object containing the original counts and some meta-information. 
        The detected transcript as well as the polygons. 
    Raises
    ------
    TypeError
        cell_metadata has to be a .csv file or a pandas DataFrame 
    TypeError
        cell_boundary_polygons has to be a .parquet file or a geopandas DataFrame 
    TypeError
        detected_transcripts has to be a .csv/.parquet file or a pandas/geopandas DataFrame 
    TypeError
        cell_by_gene_counts has to be a .csv file or a pandas DataFrame 
    """
    # Cell metadata
    with process_time_ram("Processing cell metadata") as ctm:
        if isinstance(cell_metadata, str) and ("csv" in os.path.splitext(cell_metadata)[1]):
            cell_meta = pd.read_csv(cell_metadata, index_col=0)
        elif isinstance(cell_metadata, pd.DataFrame):
            cell_meta = cell_metadata.copy()
        else: 
            raise TypeError("Only .csv file or pandas dataframe is allowed")
        cell_meta.index.rename(name='cell_id', inplace=True)
        cell_meta.rename(columns={cell_centroid_x_col: "center_x",
                                cell_centroid_y_col: "center_y"}, inplace=True, errors='raise')

    # Polygons of cells 
    with process_time_ram("Processing cell boundaries") as ctm:
        if isinstance(cell_boundary_polygons, str) and ("parquet" in os.path.splitext(cell_boundary_polygons)[1]):
            cell_coords = read_parquet(cell_boundary_polygons)
        elif isinstance(cell_boundary_polygons, gpd.GeoDataFrame):
            cell_coords = cell_boundary_polygons.copy()
        else: 
            raise TypeError("Only .parquet file or geopandas dataframe is allowed")
        cell_coords.index.rename(name='cell_id', inplace=True)
        if cell_coords.geometry.name != "cell_boundary_geom":
            cell_coords.rename_geometry("cell_boundary_geom", inplace=True)
        # Remove potential duplicated vertices in a polygon
        cell_coords['cell_boundary_geom'] = cell_coords['cell_boundary_geom'].remove_repeated_points(tolerance=0.0)
        # Sanity check 
        assert set(cell_coords.index) == set(cell_meta.index), "Cell populations in cell boundaries and cell metadata are different."
    
    # Transcript information 
    with process_time_ram("Processing Transcript information ") as ctm:
        # We also convert the pandas dataframe to geopandas geodataframe 
        # by constructing points from the locations of each transcript
        if isinstance(detected_transcripts, str) and ("parquet" in os.path.splitext(detected_transcripts)[1]):
            tx_metadata = read_parquet(detected_transcripts)
        elif isinstance(detected_transcripts, str) and ("csv" in os.path.splitext(detected_transcripts)[1]):
            tx_metadata = pd.read_csv(detected_transcripts, index_col=0)    
        elif isinstance(detected_transcripts, pd.DataFrame) or isinstance(detected_transcripts, gpd.GeoDataFrame):
            tx_metadata = detected_transcripts.copy()
        else: 
            raise TypeError("Only .parquet/.csv file or geopandas/pandas dataframe is allowed")

        tx_metadata.rename(columns={tx_x_col: "global_x",
                                    tx_y_col: "global_y"},
                           inplace=True, errors='raise')
        # Convert pd.DataFrame to geopandas 
        if not isinstance(tx_metadata, gpd.GeoDataFrame):
            tx_metadata = gpd.GeoDataFrame(tx_metadata, 
                                        geometry=gpd.points_from_xy(tx_metadata["global_x"], tx_metadata["global_y"]))
        # All index will be reset
        tx_metadata.reset_index(drop=True, inplace=True)
        tx_metadata.index = "tx_" + tx_metadata.index.astype(str)
        tx_metadata['molecule_id'] = tx_metadata.index
        if tx_metadata.geometry.name != "tx_geom":
            tx_metadata.rename_geometry("tx_geom", inplace=True)
        tx_metadata.rename(
            columns={
                gene_col: 'gene',
                cell_col: 'cell_id'
                }, inplace=True, errors='raise')
        # Sanity check 
        assert set(tx_metadata['cell_id']) == set(cell_meta.index), "Cell populations in transcript metadata and cell metadata are different."
        
    # Cell by gene counts matrix 
    with process_time_ram("Processing cell by gene matrix") as ctm:
        if cell_by_gene_counts is not None:
            if isinstance(cell_by_gene_counts, str) and ("csv" in os.path.splitext(cell_by_gene_counts)[1]):
                counts = pd.read_csv(cell_by_gene_counts, index_col=0)
            elif isinstance(cell_by_gene_counts, pd.DataFrame):
                counts = cell_by_gene_counts.copy()
            else: 
                raise TypeError("Only .csv file or pandas dataframe is allowed")
        else: 
            # Construct cell-by-gene from detected transcripts 
            counts = tx_metadata.groupby(['cell_id', "gene"],
                                        as_index=False)['molecule_id'].count()
            counts = pd.pivot(counts, values='molecule_id', columns="gene", index='cell_id').fillna(0)
            
        counts.index.rename(name='cell_id', inplace=True)
        # Sanity check 
        assert set(counts.index) == set(cell_meta.index), "Cell populations in cell-by-gene counts and cell metadata are different."
        assert set(counts.columns) == set(tx_metadata['gene']), "Genes in cell-by-gene counts and transcript metadata are different."
        

    # Create AnnData object to store the counts 
    # and facilitate future processing 
    with process_time_ram("Creating AnnData object") as ctm:
        adata = sc.AnnData(counts)
        adata.obs['x'] = cell_meta.loc[adata.obs_names, "center_x"]
        adata.obs['y'] = cell_meta.loc[adata.obs_names, "center_y"]
        adata.obs = gpd.GeoDataFrame(adata.obs,
                                    geometry=gpd.points_from_xy(adata.obs['x'], adata.obs['y']))
        adata.obs.rename_geometry("cell_centroid_geom", inplace=True)
        adata.var['col_index'] = [i for i in range(adata.var.shape[0])]
        adata.var['gene'] = adata.var_names
        adata.var.set_index("gene", inplace=True)
        # As we will alter the counts later on, we will save a copy of the 
        # original count in one of the layers 
        adata.layers['counts_0'] = adata.X.copy()
        adata.raw = adata.copy()
    
    # Then we perform basic normalization and transformation 
    with process_time_ram("Successfully read in data. Performing basic transformation") as ctm:
        adata = process_adata(adata=adata,
                            layer="counts_0",
                            dr_method=dr_method)
    # And we will use leiden to perform cell clustering
    if celltype_col is not None:
        with process_time_ram("Assigning pre-existing cell typing info from metadata.") as ctm:
            adata.obs['cell_type'] = cell_meta.loc[adata.obs_names, celltype_col]
            adata.obs['leiden'] = pd.factorize(cell_meta.loc[adata.obs_names, celltype_col])[0].astype(str)
    else:
        with process_time_ram("Performing Leiden clustering.") as ctm:
            sc.tl.leiden(adata, resolution=leiden_res, key_added='cell_type')
            adata.obs['leiden'] = adata.obs['cell_type'].astype(str)
    
    with process_time_ram("Adding meta info to adata") as ctm:
        # leiden always stores the latest version 
        adata.obs['counts_0_leiden'] = adata.obs['leiden'].copy()
        # Record some meta information 
        adata.uns['unique_leiden'] = np.unique(adata.obs['counts_0_leiden'])
        adata.uns['n_leiden'] = len(adata.uns['unique_leiden'])
        adata.uns['cell_type_leiden_map'] = adata.obs[["cell_type", "leiden"]].drop_duplicates(ignore_index=True)
        adata.uns['cell_type_leiden_map'].rename(columns={"cell_type": "cell_type_name",
                                                        "leiden": "cell_type_index"},
                                                inplace=True)
        adata.uns['dr_method'] = dr_method
        adata.uns['centroid_x_min'] = adata.obs['x'].min()
        adata.uns['centroid_x_max'] = adata.obs['x'].max()
        adata.uns['centroid_y_min'] = adata.obs['y'].min()
        adata.uns['centroid_y_max'] = adata.obs['y'].max()
        adata.uns['n_genes'] = adata.var.shape[0]
        adata.uns['current_layer'] = "counts_0"
    
    return adata, cell_coords, tx_metadata
    


def calculate_mask_distance(adata: sc.AnnData,
                            cell_coords: gpd.GeoDataFrame,
                            max_centroid_dist: float=50) -> gpd.GeoDataFrame:
    """Calculate cell-cell distance based on their cell masks. 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell meta-information
    cell_coords : gpd.GeoDataFrame
        The geodataframe recording the vertices of all the cells 
    max_centroid_dist : float, optional
        The threshold on cell-cell centroid distances beyond which we do not consider two cells being neighbors, by default 50

    Returns
    -------
    gpd.GeoDataFrame
        A dataframe where each row is a pair of neighboring cells as well as their distance information 
    """
    
    with process_time_ram("Compute cell-cell distance matrix based on their recorded centroids") as ctm:
        # First, we compute cell-cell distance matrix based on their recorded centroids 
        # As computing mask distances between all pairs of cells would take too long
        # we use the centroid distance to filter out the majority of the cells 
        # However, direct computation would consume a huge chunk of memory. Therefore, we use KDTree to 
        # find nearest neighbors. Currently, the number is set to 25 with an upper bound 
        # 
        centroid_dist_tree = KDTree(adata.obs[['x','y']])
        # The distance is sorted 
        _, adj_ind = centroid_dist_tree.query(adata.obs[['x','y']], k=25, 
                                            distance_upper_bound=max_centroid_dist, workers=-1)
        # This will directly give us cells and their at most 25 NNs
        # However, the adj_ind is the numeric index (row number)
        adj = pd.DataFrame(adj_ind, index=adata.obs_names).melt(ignore_index=False).drop(columns=['variable'])
        # Filter out values due to upper bound 
        adj = adj[adj['value']<adata.obs.shape[0]]
        # Replace row number by actual cell id 
        adj.loc[:,'n'] = adata.obs_names[adj['value']]
        adj_masks_ids = adj[adj.index != adj.loc[:, "n"]].drop(columns=['value'])
    # Then we compute the cell-cell distance based on their masks
    with process_time_ram("Compute cell-cell distance matrix based on their masks") as ctm:
        md = distance(
            cell_coords.loc[adj_masks_ids.index, "cell_boundary_geom"].values,
            cell_coords.loc[adj_masks_ids['n'].values, "cell_boundary_geom"].values
        )
        mask_distance = gpd.GeoDataFrame(adj_masks_ids).rename(columns={"n": "neighbor_cell_id"})
        mask_distance['mask_distance'] = md
        mask_distance.reset_index(inplace=True, drop=False)
        mask_distance = mask_distance.merge(adata.obs[['x', 'y', 'cell_centroid_geom']], how='left', 
                                            left_on='cell_id', right_index=True).set_geometry("cell_centroid_geom")
    return mask_distance


def even_split(array: np.array, 
               chunk_size: int) -> list:
    """Split an array into chunks of specified size 

    Parameters
    ----------
    array : np.array
        An array to be split
    chunk_size : int
        The target chunk size 

    Returns
    -------
    list
        A list of chunks of the original array 
    """
    return np.array_split(array, np.ceil(array.shape[0] / chunk_size), axis=0)


def extract_layer_num(layer: str) -> int:
    """Extracts the layer number 
    It extracts all the number in the layer string but only returns the first one

    Parameters
    ----------
    layer : str
        Name of the layer. It should be like counts_0, counts_1_proposed_update, etc

    Returns
    -------
    int
        The layer number
    """
    return int(re.findall(r'\d+', layer)[0])
    

def generate_count_patches(adata: Union[sc.AnnData, pl.DataFrame],
                           tx_to_reassign: pl.DataFrame,
                           tx_to_remove: pl.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Based on the transcript reassignment, generate patches to update the current count matrix 

    Parameters
    ----------
    adata : Union[sc.AnnData, pl.DataFrame]
        Either an AnnData to be updated or a polars dataframe with all zeros 
    tx_to_reassign : pl.DataFrame
        Transcripts that should be reassigned
    tx_to_remove : pl.DataFrame
        Transcripts that should be removed

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The first dataframe contains counts that should be subtracted from the current count
        The second dataframe contains counts that should be added to the current count
        The third dataframe contains counts that should be subtracted from the current count
    """
    # Generate three patches for the count matrix 
    if isinstance(adata, sc.AnnData):
        # This would be extremely slow
        counts_to_subtract = pl.from_pandas(pd.DataFrame(0, index=adata.obs_names, columns=adata.var_names), include_index=True)
        counts_to_add = pl.from_pandas(pd.DataFrame(0, index=adata.obs_names, columns=adata.var_names), include_index=True)
        rm_counts_to_subtract = pl.from_pandas(pd.DataFrame(0, index=adata.obs_names, columns=adata.var_names), include_index=True)
    elif isinstance(adata, pl.DataFrame):
        counts_to_subtract = adata.clone()
        counts_to_add = adata.clone()
        rm_counts_to_subtract = adata.clone()
    else: 
        raise TypeError("Unknown adata type")
    # For removal, we for each cell count how many genes occured 
    cell_to_remove = tx_to_reassign.group_by(['cell_id', "gene"]).len()
    rm_cell_to_remove = tx_to_remove.group_by(['cell_id', "gene"]).len()
    # For addition, we for each cell in the neighbor count how many genes occured 
    cell_to_add = tx_to_reassign.group_by(['neighbor_cell_id', "gene"]).len()
    cell_to_add = cell_to_add.rename({"neighbor_cell_id": "cell_id"})
    # Transform the dataframe from long to wide 
    subtract_patch = cell_to_remove.pivot(values="len", on="gene", index='cell_id').fill_null(0)
    add_patch = cell_to_add.pivot(values="len", on="gene", index='cell_id').fill_null(0)
    rm_subtract_patch = rm_cell_to_remove.pivot(values="len", on="gene", index='cell_id').fill_null(0)
    # The update the find matching rows and columns 
    counts_to_subtract = counts_to_subtract.update(subtract_patch, on="cell_id", how='left').to_pandas().set_index("cell_id", drop=True)
    counts_to_add = counts_to_add.update(add_patch, on="cell_id", how='left').to_pandas().set_index("cell_id", drop=True)
    rm_counts_to_subtract = rm_counts_to_subtract.update(rm_subtract_patch, on="cell_id", how='left').to_pandas().set_index("cell_id", drop=True)
    
    return counts_to_subtract, counts_to_add, rm_counts_to_subtract


def make_reassignment_adata(adata: sc.AnnData,
                            layer: str,
                            tx_to_reassign: pl.DataFrame,
                            tx_to_remove: pl.DataFrame,
                            trial_layer: Optional[str]=None,
                            dr_method: str='pca') -> sc.AnnData:
    """Make the count adjustment on the adata alone 
    This will not alter the tx information 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell metainformation
    layer : str
        The layer upon which the update is computed 
    tx_to_reassign : pl.DataFrame
        Transcripts that should be reassigned
    tx_to_remove : pl.DataFrame
        Transcripts that should be removed
    trial_layer : Optional[str] 
        If provided, the updated counts will be stored in this layer 
    dr_method : str, optional
        The dimension reduction method to be used. Either pca or umap, by default "umap"
    Returns
    -------
    sc.AnnData
        The adjusted adata
    """
    with process_time_ram("Updating gene counts") as ctm: 
        zero_counts = adata.to_df().copy()
        zero_counts.loc[:,:] = 0
        zero_counts = pl.from_pandas(zero_counts, include_index=True)
        counts_to_subtract, counts_to_add, rm_counts_to_subtract = generate_count_patches(adata=zero_counts,
                                                                                        tx_to_reassign=tx_to_reassign,
                                                                                        tx_to_remove=tx_to_remove)
        layer_num = extract_layer_num(layer)
        if trial_layer is None:
            trial_layer = "counts_"+str(int(layer_num+1))
            
        # Update adata
        adata.layers[trial_layer] = adata.layers[layer]+counts_to_add-counts_to_subtract-rm_counts_to_subtract
        if np.any(adata.layers[trial_layer]<0):
            raise Exception("Negative values generated. This might be due inconsistency between the count matrix and the tx data.")
        adata = process_adata(adata=adata, layer=trial_layer, dr_method=dr_method)
    return adata


def sample_logistic(shape: tuple, 
                    model_device: torch.device,
                    eps: float=1e-20):
    """Sample standard logistic random variables

    Parameters
    ----------
    shape : tuple
        Shape of the final result 
    model_device : torch.device
        Specify where the tensor will be stored 
    eps : float, optional
        Used to avoid -inf from log, by default 1e-20

    Returns
    -------
    torch.tensor
        Logistic RVs
    """
    U = torch.rand(shape).to(model_device)
    return torch.log(U/(1-U + eps) + eps)


def binary_gumbel_softmax_sample(logits: torch.tensor,
                                 temperature: float,
                                 model_device: torch.device,
                                 hard: bool=True) -> torch.tensor:
    """Gumbel softmax trick for binary variables 

    Parameters
    ----------
    logits : torch.tensor
        Unnormalized class probabilities 
    temperature : float
        Temperature
    model_device : torch.device
        Specify where the tensor will be stored 
    hard: bool, optional 
        Whether or not hard 0, 1 samples should be generated, by default True
    Returns
    -------
    torch.tensor
        Gumbel softmax "bernoulli" variables 
    """
    y = logits + sample_logistic(logits.size(), model_device=model_device)
    y_soft = torch.sigmoid(y/temperature)
    if hard:
        y_hard = torch.round(y_soft, decimals=0)
        y_hard = (y_hard - y_soft).detach() + y_soft
        return y_hard 
    else: 
        return y_soft


def calibrate_threshold(alpha_0: np.array,
                        alpha_1: np.array) -> Callable:
    """Find a threshold for dichotomization 

    Parameters
    ----------
    alpha_0 : np.array
        The intercept value computed based on the prior 50
    alpha_1 : np.array
        The slope value computed based on the prior 5

    Returns
    -------
    Callable
        A function that finds a proper threshold 
    """
    # The logic is if the transcripts are randomly scattered around, then with 
    # the prior, we will only be assigning roughly 0.2% transcripts. 
    # We want to set this as the threshold 0.5. 
    # Based on the alpha_0 and alpha_1, we can compute the percentile rank 
    # required to obtain 0.5 based on 1/(1+e^(-(alpha_0 + alpha_1 logit(p))).
    # Since we are using the product, we compute 0.5**(1/3)
    temp = np.power(0.5, 1/3)
    # Find the corresponding logit 
    logit = (np.log(temp/(1-temp+1e-20)+1e-20)-alpha_0)/(alpha_1+1e-20)
    # Convert back to probability
    threshold = np.power(1-1/(1+np.exp(-logit)), 3)
    a=np.log(0.5)/np.log(threshold+1e-20)
    def calibrator(x: np.array) -> Callable:
        return betainc(a, 1, x)
    return calibrator


def compute_gene_threshold(adata: sc.AnnData,
                        tx_reassign_info: pl.DataFrame) -> pl.DataFrame:
    """Compute reassignment thresholds for all the genes 

    Parameters
    ----------
    adata : sc.AnnData
        An Anndata containing gene information 
    tx_reassign_info : pl.DataFrame
        A polars dataframe with reassigning probabilities

    Returns
    -------
    pl.DataFrame
        A dataframe with a threshold for each gene 
    """
    all_genes = adata.var.index
    result = pl.DataFrame(data={"gene": all_genes, 
                                "threshold":0.0})
    for n_row, gene in tqdm(enumerate(all_genes)): 
        reassign_probs = tx_reassign_info.filter(pl.col('gene')==gene)['reassign_probs'] 
        smoothed_counts, bin_edges = np.histogram(reassign_probs, bins=30)
        # Detect peaks with prominence filter to avoid false positives
        peaks, _ = find_peaks(smoothed_counts, height=5, prominence=20)
        # Calculate peak widths with adjusted rel_height
        results_half = peak_widths(smoothed_counts, peaks, rel_height=1)
        # Extract and correct `left_ips` values
        left_ips_corrected = np.maximum(results_half[2], 0)
        if left_ips_corrected.shape[0] > 0:
            l_index = round(left_ips_corrected[-1])
            threshold = bin_edges[l_index]
        else:
            # If no peak is found, we set the threshold to 0.5
            threshold = 0.5
        if threshold < 0.5:
            # If the threshold is less than 0.5, we raise it to 0.5
            threshold = 0.5
        result[n_row, "threshold"] = threshold
    return result


class diagLinear(nn.Module):
    def __init__(self, 
                 features: int,
                 bias: bool=True,
                 initial_weights: Optional[Union[np.array, list]]=None,
                 initial_bias: Optional[Union[np.array, list]]=None) -> None:
        """A linear module with only diagonal elements 

        Parameters
        ----------
        features : int
            Number of features 
        bias : bool, optional
            If bias should be included, by default True
        initial_weights : Optional[Union[np.array, list]], optional
            Initial values for weights, by default None
        initial_bias : Optional[Union[np.array, list]], optional
            Initial values for bias, by default None
        """
        super().__init__()
        # If inital values are not provided, we initialize them as they do in PyTorch
        stdv = 1. / np.sqrt(features)
        if initial_weights is not None:
            self.weight = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32).reshape(1, features))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(1, features).uniform_(-stdv, stdv))
        if initial_bias is not None:
            self.bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32).reshape(1, features))
        else: 
            self.bias = torch.zeros_like(self.weight)
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(1, features).uniform_(-stdv, stdv))
            
    def forward(self, X: torch.tensor) -> torch.tensor:
        """The forward pass 

        Parameters
        ----------
        X : torch.tensor
            Input tensor 

        Returns
        -------
        torch.tensor
            The computed values
        """
        return X * self.weight + self.bias


class Positive(nn.Module):
    """This class is used to reparametrize model weights 

    """
    def forward(self, X: torch.tensor) -> torch.tensor:
        """The forward pass 

        Parameters
        ----------
        X : torch.tensor
            Input tensor 

        Returns
        -------
        torch.tensor
            The computed values
        """
        return F.softplus(X)
    

class JSONEncoder(json.JSONEncoder):
    """This class is used to save dictionary of pd.DataFrame to a json file

    """
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        return json.JSONEncoder.default(self, obj)


