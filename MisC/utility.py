# Data IO
import os 
import scanpy as sc
import geopandas as gpd
from geopandas import read_parquet
import pandas as pd
# Data manipulation 
import re 
import numpy as np
import torch
import torch.nn.functional as F
from shapely import distance
from scipy.spatial.distance import cdist
from itertools import combinations
# Typing 
from typing import Optional, Union, Tuple


def import_data(cell_by_gene_counts: Union[str, pd.DataFrame],
                cell_metadata: Union[str, pd.DataFrame],
                cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                detected_transcripts: Union[str, pd.DataFrame, gpd.GeoDataFrame],
                cell_centroid_x_col: str='center_x',
                cell_centroid_y_col: str='center_y',
                tx_x_col: str='global_x',
                tx_y_col: str='global_y',
                gene_col: str='gene',
                cell_col: str='cell_id',
                celltype_col: Optional[str]=None,
                leiden_res: float=1,
                preprocess: bool=True,
                ) -> Tuple[sc.AnnData, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Read in the four pieces information needed for subsequent analysis: 
    cell-by-gene counts, cell metadata, cell boundary information, and transcript information. Some 
    basic visualization and cell clustering will be performed. 

    Parameters
    ----------
    cell_by_gene_counts : Union[str, pd.DataFrame]
        Either the path to the csv file or a pandas dataframe containing the cell-by-gene count matrix whose first 
        column is assumed to contain the ID for each cell while the rest of the columns should be the transcript counts
        for each cell. 
    cell_metadata : Union[str, pd.DataFrame]
        Either the path to the csv file or a pandas dataframe containing the metadata for each cell whose 
        first column is assumed to contain the ID for each cell. The information in the file/object should 
        at least contain the xy coordinates of the centers of cells named center_x, and center_y, respectively.
        If the users have already performed cell typing which is not required and will only be used for visualization,
        the information can be stored here as a separate column: cell_type.
    cell_boundary_polygons : Union[str, gpd.GeoDataFrame]
        Either the path to the parquet file or the geopandas GeoDataFrame containing the vertex information 
        for each cell. The first column is assumed to be the IDs for cells. It should contain one column 'Geometry'
        that records the coordinates of the vertices.
    detected_transcripts : Union[str, pd.DataFrame]
        Either the path to the csv file or a pandas dataframe containing the information of the detected transcripts.
        The first column is assumed to be some index. The information should contain the ID for a transcripte/molecule, 
        the ID of the cell it belongs to, its xy coordinate named global_x, global_y respectively, and its gene information. 

    Returns
    -------
    Tuple[sc.AnnData, gpd.GeoDataFrame, gpd.GeoDataFrame]
        An AnnData object containing the original counts and some metainformation. The detected transcript as well as the polygons. 
    """
    
    # Cell by gene counts matrix 
    if isinstance(cell_by_gene_counts, str) and ("csv" in os.path.splitext(cell_by_gene_counts)[1]):
        counts = pd.read_csv(cell_by_gene_counts, index_col=0)
    elif isinstance(cell_by_gene_counts, pd.DataFrame):
        counts = cell_by_gene_counts
    else: 
        raise TypeError("Only .csv file or pandas dataframe is allowed")
    counts.index.rename(name='cell_id', inplace=True)
    # Cell metadata
    if isinstance(cell_metadata, str) and ("csv" in os.path.splitext(cell_metadata)[1]):
        cell_meta = pd.read_csv(cell_metadata, index_col=0)
    elif isinstance(cell_metadata, pd.DataFrame):
        cell_meta = cell_metadata
    else: 
        raise TypeError("Only .csv file or pandas dataframe is allowed")
    cell_meta.index.rename(name='cell_id', inplace=True)
    cell_meta.rename(columns={cell_centroid_x_col: "center_x",
                              cell_centroid_y_col: "center_y"}, inplace=True)
    # Polygons of cells 
    if isinstance(cell_boundary_polygons, str) and ("parquet" in os.path.splitext(cell_boundary_polygons)[1]):
        cell_coords = read_parquet(cell_boundary_polygons)
    elif isinstance(cell_boundary_polygons, gpd.GeoDataFrame):
        cell_coords = cell_boundary_polygons
    else: 
        raise TypeError("Only .parquet file or geopandas dataframe is allowed")
    cell_coords.index.rename(name='cell_id', inplace=True)
    cell_coords.rename_geometry("cell_boundary_geom", inplace=True)
    # Transcript information 
    # We also convert the pandas dataframe to geopandas geodataframe 
    # by constructing points from the locations of each transcript
    if isinstance(detected_transcripts, str) and ("parquet" in os.path.splitext(detected_transcripts)[1]):
        tx_metadata = read_parquet(detected_transcripts)
    elif isinstance(detected_transcripts, str) and ("csv" in os.path.splitext(detected_transcripts)[1]):
        tx_metadata = pd.read_csv(detected_transcripts, index_col=0)    
    elif isinstance(detected_transcripts, pd.DataFrame) or isinstance(detected_transcripts, gpd.GeoDataFrame):
        tx_metadata = detected_transcripts
    else: 
        raise TypeError("Only .parquet/.csv file or geopandas/pandas dataframe is allowed")

    if not isinstance(tx_metadata, gpd.GeoDataFrame):
        tx_metadata = gpd.GeoDataFrame(tx_metadata, 
                                       geometry=gpd.points_from_xy(tx_metadata[tx_x_col], tx_metadata[tx_y_col]))
        
        tx_metadata.index = ['tx_' + str(i+1) for i in range(tx_metadata.shape[0])]
        tx_metadata['molecule_id'] = tx_metadata.index
    tx_metadata.rename_geometry("tx_geom", inplace=True)
    tx_metadata.rename(
        columns={
            gene_col: 'gene',
            cell_col: 'cell_id'
            }, inplace=True)
    # Create AnnData object to store the counts 
    # and facilitate future processing 
    adata = sc.AnnData(counts)
    adata.obs['x'] = cell_meta.loc[adata.obs_names, "center_x"]
    adata.obs['y'] = cell_meta.loc[adata.obs_names, "center_y"]
    adata.obs = gpd.GeoDataFrame(adata.obs,
                                 geometry=gpd.points_from_xy(adata.obs['x'], adata.obs['y']))
    adata.obs.rename_geometry("cell_centroid_geom", inplace=True)
    adata.var['col_index'] = [i for i in range(adata.var.shape[0])]
    # As we will alter the counts later on, we will save a copy of the 
    # original count in one of the layers 
    adata.layers['counts_0'] = counts.copy()
    if preprocess:
        # Then we perform basic normalization and transformation 
        print("="*30)
        print("Successfully read in data. Performing basic transformation")
        sc.pp.normalize_total(adata, target_sum=1000)
        sc.pp.log1p(adata)
        # We also perform basic visualization 
        print("UMAP")
        adata.raw = adata.copy()
        sc.pp.scale(adata)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    # And we will use leiden to perform cell clustering
    if celltype_col is not None:
        print("Assigning pre-existing cell typing info from metadata.")
        adata.obs['cell_type'] = cell_meta.loc[adata.obs_names, celltype_col]
        adata.obs['leiden'] = pd.factorize(cell_meta.loc[adata.obs_names, celltype_col])[0]
    else:
        print("Performing Leiden clustering.")
        sc.tl.leiden(adata, resolution=leiden_res, key_added='cell_type')
        adata.obs['leiden'] = adata.obs['cell_type'].astype(int)
    
    adata.uns['counts_0_leiden'] = np.unique(adata.obs['leiden'])
    adata.uns['counts_0_n_leiden'] = len(adata.uns['counts_0_leiden'])
    adata.uns['centroid_x_min'] = adata.obs['x'].min()
    adata.uns['centroid_x_max'] = adata.obs['x'].max()
    adata.uns['centroid_y_min'] = adata.obs['y'].min()
    adata.uns['centroid_y_max'] = adata.obs['y'].max()
    adata.uns['n_genes'] = adata.var.shape[0]
    
    print("Done~")
    print("="*30)
    return adata, cell_coords, tx_metadata
    


def calculate_mask_distance(adata: sc.AnnData,
                            cell_coords: gpd.GeoDataFrame,
                            max_centroid_dist: int=50,
                            min_centroid_dist: int=0) -> gpd.GeoDataFrame:
    """Calculate cell-cell distance based on their cell masks. The calculation is only done among 
    neighboring cells of different types. 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell metainformation
    cell_coords : gpd.GeoDataFrame
        The geodataframe recording the vertices of all the cells 
    max_centroid_dist : int, optional
        The threshold on cell-cell centroid distances beyond which we do not consider two cells being neighbors, by default 15
    min_centroid_dist : int, optional
        The threshold on cell-cell centroid distances under which we do not consider two cells being neighbors, by default 0
    cluster_col : str, optional
        Column name for cell clustering, by default 'leiden'

    Returns
    -------
    gpd.GeoDataFrame
        A dataframe where each row is a pair of neighboring cells as well as their distance information 
    """
    
    # First, we compute cell-cell distance matrix based on their recorded centroids 
    centroid_dist = cdist(adata.obs[['x','y']],adata.obs[['x','y']])
    centroid_dist = pd.DataFrame(centroid_dist, index = adata.obs_names, columns=adata.obs_names)
    # As computing mask distances between all pairs of cells would take too long
    # we use the centroid distance to filter out the majority of the cells 
    # We then create the adjacency matrix: 
    # Any cell that is within in the specified threshold: max_centroid_dist is considered a neighbor
    adj = (centroid_dist > min_centroid_dist) & (centroid_dist<= max_centroid_dist)
    # For each cell, we record its neighbors as a list
    adj = adj.agg(lambda x: adj.columns[x.values].tolist(), axis=1)
    adj = pd.DataFrame(adj, columns = ['n'])
    # We further exclude cells from the same cell cluster
    # since we are only interested in comparing the transcript abundances among cells of different types 
    adj = adj.agg(lambda x: [
            y for y in x.n if adata.obs.loc[y, 'leiden']!=adata.obs.loc[x.name, 'leiden']], axis=1)
    # We then exclude cells surrounded by cells of its own type 
    adj_nonself = adj[adj.apply(len)>0]
    # Then we compute the cell-cell distance based on their masks
    adj_nonself_masks_ids = adj_nonself.explode()
    md = distance(
        cell_coords.loc[adj_nonself_masks_ids.index, "cell_boundary_geom"].values,
        cell_coords.loc[adj_nonself_masks_ids.values, "cell_boundary_geom"].values
    )
    mask_distance = pd.DataFrame(
        adj_nonself_masks_ids, columns=['neighbor_cell_id'])
    mask_distance['mask_distance'] = md
    mask_distance.reset_index(inplace=True)
    mask_distance = mask_distance.merge(adata.obs[['x', 'y', 'cell_centroid_geom']], how='left', 
                                        left_on='cell_id', right_index=True)
    # mask_distance = gpd.GeoDataFrame(mask_distance, 
    #                                  geometry=gpd.points_from_xy(mask_distance.x, mask_distance.y))
    # mask_distance.rename_geometry("centroid_geom", inplace=True)
    return mask_distance



def extract_layer_num(layer: str) -> int:
    """Extracts the layer number 

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
    

def mask_eval(l: list) -> Tuple[bool, list]:
    """Given a list of coordinates, check if it contains only on polygon

    Parameters
    ----------
    l : list
        List of potentially vertices of different polygons

    Returns
    -------
    Tuple[bool, list]
        If the list defines only one polygon, and the coordinates of its vertices 
    """
    unique_l = []
    for a,b in combinations(l,2):
        if (len(unique_l) ==0) or (a not in np.array(unique_l)):
            unique_l.append(a)
        if np.array_equal(a, b):
            continue
        else:
            if b not in np.array(unique_l):
                unique_l.append(b)
    return len(unique_l) > 1, unique_l


def generate_count_patches(adata,
                           tx_to_reassign):
    # Generate two patches for the count matrix 
    counts_to_subtract = pd.DataFrame(0, index=adata.obs_names, columns=adata.var_names)
    counts_to_add = pd.DataFrame(0, index=adata.obs_names, columns=adata.var_names)
    # For removal, we for each cell count how many genes occured 
    cell_to_remove = tx_to_reassign.groupby(by=['cell_id', "gene"], as_index=False).size()
    # For addition, we for each cell in the neighbor count how many genes occured 
    cell_to_add = tx_to_reassign.groupby(by=['neighbor_cell_id', "gene"], as_index=False).size()
    cell_to_add.rename(columns={"neighbor_cell_id": "cell_id"}, inplace=True)
    # Transform the dataframe from long to wide 
    subtract_patch = pd.pivot(cell_to_remove, values="size", columns="gene", index='cell_id').fillna(0)
    add_patch = pd.pivot(cell_to_add, values="size", columns="gene", index='cell_id').fillna(0)
    # The update the find matching rows and columns 
    counts_to_subtract.update(subtract_patch)
    counts_to_add.update(add_patch)
    
    return counts_to_subtract, counts_to_add


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def binary_gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return torch.sigmoid(y / temperature)

def multinomial_gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)