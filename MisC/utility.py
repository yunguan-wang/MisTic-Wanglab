# Data IO
import scanpy as sc
import geopandas as gpd
from geopandas import read_parquet
import pandas as pd
# Data manipulation 
import re 
import numpy as np
from shapely import Point, Polygon, distance
from scipy.spatial.distance import cdist
from scipy.stats import norm
from itertools import combinations
# Typing 
from typing import Optional, Union, Tuple


def import_data(cell_by_gene_counts: Union[str, pd.DataFrame],
                cell_metadata: Union[str, pd.DataFrame],
                cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                detected_transcripts: Union[str, pd.DataFrame],
                x_col: str='center_x',
                y_col: str='center_y',
                tx_x_col: str='global_x',
                tx_y_col: str='global_y',
                tx_geom_col: str='point',
                cell_geom_col: str='Geometry',
                gene_col: str='gene',
                cell_col: str='cell_id',
                celltype_col: str=None,
                leiden_res: float=1,
                preprocess = True,
                ) -> Tuple[sc.AnnData, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Read in the four pieces information needed for subsequent analysis: 
    cell-by-gene counts, cell metadata, cell boundary information, and transcript information. Some 
    basic visualization and cell clustering will be performed. 
    
    #TODO: This function needs to sanitize all columns names to be used in later processes.
    #TODO: Rename xy coordinates to 'x', and 'y', and cell names to 'cell_id' and cell type to 'cell_type'

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
    if isinstance(cell_by_gene_counts, str):
        counts = pd.read_csv(cell_by_gene_counts, index_col=0)
    else:
        counts = cell_by_gene_counts
    
    # Cell metadata
    if isinstance(cell_metadata, str):
        cell_meta = pd.read_csv(cell_metadata, index_col=0)
    else: 
        cell_meta = cell_metadata
    
    # Polygons of cells 
    if isinstance(cell_boundary_polygons, str):
        cell_coords = read_parquet(cell_boundary_polygons)
    else: 
        cell_coords = cell_boundary_polygons
    # Transcript information 
    # We also convert the pandas dataframe to geopandas geodataframe 
    # by constructing points from the locations of each transcript
    if isinstance(detected_transcripts, str):
        tx_metadata = pd.read_csv(detected_transcripts, index_col=0)    
    else:
        tx_metadata = detected_transcripts

    if not isinstance(tx_metadata, gpd.GeoDataFrame):
        tx_metadata = gpd.GeoDataFrame(tx_metadata, 
                                    geometry=gpd.points_from_xy(tx_metadata[tx_x_col], tx_metadata[tx_y_col]))
        tx_metadata.rename_geometry(tx_geom_col, inplace=True)
        tx_metadata.indx = ['tx_' + str(i+1) for i in range(tx_metadata.shape[0])]

    # Sanitize column names
    cell_coords.rename({cell_geom_col: 'Geometry'}, axis=1)
    tx_metadata.rename(
        {
            tx_geom_col: 'point',
            gene_col: 'gene',
            cell_col: 'cell_id'
            }, axis=1)
    # Create AnnData object to store the counts 
    # and facilitate future processing 
    adata = sc.AnnData(counts)
    adata.obs['x'] = cell_meta.loc[adata.obs_names,x_col]
    adata.obs['y'] = cell_meta.loc[adata.obs_names,y_col] 
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
        adata.obs['cell_type'] = cell_meta.loc[adata.obs_names,celltype_col]
        adata.obs['leiden'] = cell_meta.loc[adata.obs_names,celltype_col] 
    else:
        print("Performing Leiden clustering.")
        sc.tl.leiden(adata, resolution=leiden_res, key_added='cell_type')
        adata.obs['leiden'] = adata.obs['cell_type']
    print("Done~")
    print("="*30)
    return adata, cell_coords, tx_metadata
    


def calculate_mask_distance(adata: sc.AnnData,
                            cell_coords: gpd.GeoDataFrame,
                            max_centroid_dist: int=50,
                            min_centroid_dist: int=0,
                            cluster_col: str='leiden',
                            geometry_col: str='Geometry',
                            re_cal_centroid_dist: bool=False) -> gpd.GeoDataFrame:
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
    geometry_col : str, optional
        Column name for the polygons, by default 'Geometry'
    re_cal_centroid_dist : bool, optional
        If centroid distances should be recorded, by default False

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
            y for y in x.n if adata.obs.loc[y, cluster_col]!=adata.obs.loc[x.name, cluster_col]], axis=1)
    # We then exclude cells surrounded by cells of its own type 
    adj_nonself = adj[adj.apply(len)>0]
    # Then we compute the cell-cell distance based on their masks
    adj_nonself_masks_ids = adj_nonself.explode()
    md = distance(
        cell_coords.loc[adj_nonself_masks_ids.index, geometry_col].values,
        cell_coords.loc[adj_nonself_masks_ids.values, geometry_col].values
    )
    mask_distance = pd.DataFrame(
        adj_nonself_masks_ids, columns=['neighbor_by_centroid'])
    mask_distance['mask_distance'] = md
    mask_distance.reset_index(inplace=True)
    mask_distance = mask_distance.merge(adata.obs[['x', 'y']], how='left', 
                                        left_on='cell_id', right_index=True)
    mask_distance = gpd.GeoDataFrame(mask_distance, 
                                     geometry=gpd.points_from_xy(mask_distance.x, mask_distance.y))
    mask_distance.rename_geometry("centroid_geom", inplace=True)
    # recalculating the centroid distance is expnsive, so it should be avoided
    if re_cal_centroid_dist:
        v1 = adata.obs.loc[mask_distance.index, ['x','y']].values
        v2 = adata.obs.loc[mask_distance.neighbor_by_centroid, ['x','y']].values
        mask_distance['centroid_distance'] = np.sum((v1-v2)**2, axis=1)**0.5
    return mask_distance


def annotate_tx_mask_distance(
        adata: sc.AnnData,
        tx_metadata: gpd.GeoDataFrame,
        cell_coords: gpd.GeoDataFrame,
        mask_distance: pd.DataFrame, 
        mask_dist_cutoff: float=1, 
        # x_col: str='x',
        # y_col: str='y',
        # gene_col: str='gene',
        cell_col: str='cell_id',
        tx_geom_col: str = 'point',
        cell_geome_col: str='Geometry',
        cluster_col: str='cell_type',
        ) -> gpd.GeoDataFrame:
    """Depending on the cell-cell distance computed based on cell masks, this function computes 
    the distances of all the transcripts of a cell to all the neighboring cells 

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
    x_col : str, optional
        The column name for the x coordinate, by default 'global_x'
    y_col : str, optional
        The column name for the y coordinate, by default 'global_y'
    cell_col : str, optional
        The column name for cell id, by default 'cell_id'
    gene_col : str, optional
        The column name for gene names, by default 'gene'
    cluster_col : str, optional
        The column name for cell types, by default 'leiden'
    geometry_col : str, optional
        The column name for geometry, by default 'Geometry'

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
    valid_cells = interface[cell_col].unique()
    # We extract transcripts of those interface cells 
    intf_tx = tx_metadata[tx_metadata[cell_col].isin(valid_cells)]
    # Then we compute the distance between transcript and cell mask of neighboring cell
    intf_tx = intf_tx.merge(
        interface[[cell_col,'neighbor_by_centroid','mask_distance']], on=cell_col, how='left')
    intf_tx['mask_geom'] = cell_coords.loc[intf_tx.neighbor_by_centroid.values, cell_geome_col].values
    intf_tx['tx_mask_distance'] = distance(intf_tx[tx_geom_col], intf_tx['mask_geom'])
    intf_tx['cell_type'] = adata.obs.loc[intf_tx.cell_id, cluster_col].values
    intf_tx['neighbor_celltype'] = adata.obs.loc[intf_tx.neighbor_by_centroid, cluster_col].values
    return intf_tx


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


def smooth_transition_function(m, alpha):
    assert m >= np.sqrt(3), "m needs to be >= \sqrt(3)"
    assert alpha > 0, "alpha needs to be positive"
    def fun(x):
        
        transformed_x = 2*(x**alpha)-1
        kernel = transformed_x/(1-np.pow(transformed_x,2))
        return 1/(1+np.exp(-2*m*kernel))
    return fun


####################################
# Wait till future updates
def mix_norm_cdf(x, model):
    weights, means, covars = model.weights_, model.means_.reshape(-1), model.covariances_.reshape(-1)
    mcdf = 0.0
    for i in range(len(weights)):
        mcdf += weights[i] * norm.cdf(x, loc=means[i], scale=np.sqrt(covars[i]))
    return mcdf



def random_downsample(counts: pd.DataFrame, cells: list , n_remove:int = 5):
    cell_counts = counts.loc[cells,:].copy()
    cell_counts.index = cell_counts.index.astype(str)
    n_remove = np.minimum((cell_counts.sum(axis=1)/10).values, n_remove).astype(int)
    for c, n in zip(cells, n_remove):
        genes = cell_counts.columns[cell_counts.loc[c,:]>0]
        while True:
            remove = np.random.choice(genes, n)
            remove, remove_counts = np.unique(remove, return_counts=True)
            if (cell_counts.loc[c, remove] >= remove_counts).all():
                break
        cell_counts.loc[c, remove] -= remove_counts
    return cell_counts


    