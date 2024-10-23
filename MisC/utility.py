import numpy as np
import scanpy as sc
import geopandas as gpd
from geopandas import read_parquet
import pandas as pd

from shapely import Point, Polygon, distance
from scipy.spatial.distance import cdist

from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM
from typing import Optional, Union, Tuple


def import_data(cell_by_gene_counts: Union[str, pd.DataFrame],
                cell_metadata: Union[str, pd.DataFrame],
                cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                detected_transcripts: Union[str, pd.DataFrame]) -> Tuple[sc.AnnData, gpd.GeoDataFrame, gpd.GeoDataFrame]:
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
    
    tx_metadata = gpd.GeoDataFrame(tx_metadata, 
                                   geometry=gpd.points_from_xy(tx_metadata.global_x, tx_metadata.global_y))
    tx_metadata.rename_geometry("Geometry", inplace=True)
    tx_metadata.set_index("molecule_id", inplace=True)
    # Create AnnData object to store the counts 
    # and facilitate future processing 
    adata = sc.AnnData(counts)
    adata.obs['x'] = cell_meta.loc[adata.obs_names,'center_x']
    adata.obs['y'] = cell_meta.loc[adata.obs_names,'center_y']
    if "cell_type" in cell_meta.columns:
        adata.obs['cell_type'] = cell_meta.loc[adata.obs_names,'cell_type']
    # As we will alter the counts later on, we will save a copy of the 
    # original count in one of the layers 
    adata.layers['counts_0'] = counts.copy()
    # Then we perform basic normalization and transformation 
    print("="*30)
    print("Successfully read in data. Performing basic transformation")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    # We also perform basic visualization 
    print("UMAP")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    # And we will use leiden to perform cell clustering 
    print("Leiden")
    sc.tl.leiden(adata, resolution=1)
    print("Done~")
    print("="*30)
    return adata, tx_metadata, cell_coords
    


def calculate_mask_distance(adata: sc.AnnData,
                            cell_coords: gpd.GeoDataFrame,
                            max_centroid_dist: int=15,
                            min_centroid_dist: int=0,
                            cluster_col: str='leiden',
                            geometry_col: str='Geometry',
                            re_cal_centroid_dist: bool=False) -> pd.DataFrame:
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
    pd.DataFrame
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
    # recalculating the centroid distance is expansive, so it should be avoided
    if re_cal_centroid_dist:
        v1 = adata.obs.loc[mask_distance.index, ['x','y']].values
        v2 = adata.obs.loc[mask_distance.neighbor_by_centroid, ['x','y']].values
        mask_distance['centroid_distance'] = np.sum((v1-v2)**2, axis=1)**0.5
    return mask_distance


def annotate_tx_mask_distance(adata: sc.AnnData,
                                tx_metadata: gpd.GeoDataFrame,
                                cell_coords: gpd.GeoDataFrame,
                                mask_distance: pd.DataFrame, 
                                mask_dist_cutoff: float=1, 
                                x_col: str='global_x',
                                y_col: str='global_y',
                                cell_col: str='cell_id',
                                gene_col: str='gene',
                                cluster_col: str='leiden',
                                geometry_col: str='Geometry') -> gpd.GeoDataFrame:
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
    # We extract transcripts of those interface cells 
    intf_tx = tx_metadata[tx_metadata.cell_id.isin(interface.cell_id)][[x_col,y_col,cell_col,gene_col,geometry_col]]
    intf_tx.columns = ['x','y','cell_id','gene','tx_geom']
    intf_tx.reset_index(inplace=True)
    # Then we compute the distance between transcript and cell mask of neighboring cell
    intf_tx = intf_tx.merge(interface[['cell_id','neighbor_by_centroid','mask_distance']], on='cell_id')
    intf_tx['mask_geom'] = cell_coords.loc[intf_tx.neighbor_by_centroid.values, geometry_col].values
    intf_tx['tx_mask_distance'] = distance(intf_tx['tx_geom'], intf_tx['mask_geom'])
    intf_tx['celltype'] = adata.obs.loc[intf_tx.cell_id, cluster_col].values
    intf_tx['neighbor_celltype'] = adata.obs.loc[intf_tx.neighbor_by_centroid, cluster_col].values
    return intf_tx


def mix_norm_cdf(x, model):
    weights, means, covars = model.weights_, model.means_.reshape(-1), model.covariances_.reshape(-1)
    mcdf = 0.0
    for i in range(len(weights)):
        mcdf += weights[i] * norm.cdf(x, loc=means[i], scale=np.sqrt(covars[i]))
    return mcdf


def reassign_tx(adata: sc.AnnData, 
                layer: str, 
                intf_tx: gpd.GeoDataFrame, 
                tx_mask_d_max: float = 2.0,
                hard_threshold: Optional[float]=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate updates to the cell-by-gene counts matrix and the transcript matrix 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell metainformation
    layer : str
        The layer upon which the update is computed 
    intf_tx : gpd.GeoDataFrame
        Distance information on transcripts to neighboring cells 
    tx_mask_d_max : float, optional
        The threshold beyond which we do not consider a certain transcript as a membrane transcript, by default 2.0
    hard_threshold : Optional[float], optional
        The threshold on the difference of percentages, by default None

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Updates to counts and transcript matrices 
    """
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
        lambda x: percent_pos.loc[x['celltype'],x['gene']], axis=1).values
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
                                                gmm_dict["{}-{}".format(x['celltype'], x['neighbor_celltype'])]), axis=1).values

        intf_tx['reassign'] = np.random.binomial(1, intf_tx['remove_prob'])
    else:
        # If its greater than the threshold, we reassign the transcript 
        intf_tx['reassign'] = (intf_tx['pct_diff']>=hard_threshold).astype(int)
    
    tx_to_reassign = intf_tx[intf_tx['reassign']==1]
    # We only keep the cell that is closest to the transcript 
    tx_to_reassign = tx_to_reassign.groupby(tx_to_reassign.molecule_id).first()
    
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
    tx_assignment_update = tx_to_reassign[['neighbor_by_centroid']].rename(columns={"neighbor_by_centroid": "cell_id"})
    tx_assignment_original = tx_to_reassign[['cell_id']]

    return counts_to_subtract, counts_to_add, tx_assignment_update, tx_assignment_original






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


    