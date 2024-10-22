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
                detected_transcripts: Union[str, pd.DataFrame]) -> Tuple[sc.AnnData, pd.DataFrame, gpd.GeoDataFrame]:
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
        the ID of the cell it belongs to, its xy coordinate named x, y respectively, and its gene information. 

    Returns
    -------
    Tuple[sc.AnnData, pd.DataFrame, gpd.GeoDataFrame]
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
    if isinstance(detected_transcripts, str):
        tx_metadata = pd.read_csv(detected_transcripts, index_col=0)    
    else:
        tx_metadata = detected_transcripts
        
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
                            x_col: str='x',
                            y_col: str='y',
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
    x_col : str, optional
        Column name for the x coordinate, by default 'x'
    y_col : str, optional
        Column name for the y coordinate, by default 'y'
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
    # recalculating the centroid distance is expansive, so it should be avoided
    if re_cal_centroid_dist:
        v1 = cell_coords.loc[mask_distance.index, [x_col, y_col]].values
        v2 = cell_coords.loc[mask_distance.neighbor_by_centroid, [x_col, y_col]].values
        mask_distance['centroid_distance'] = np.sum((v1-v2)**2, axis=1)**0.5
    return mask_distance


def annotate_tx_mask_distance(
        df, # df is return from calculate_mask_distance
        tx_metadata,
        cell_coords,
        x_col = 'global_x',
        y_col = 'global_y',
        cell_col = 'cell_id',
        gene_col = 'gene',
        tx_col = 'molecule_id',
        ):
    df = df.copy()
    intf_tx = tx_metadata[
        tx_metadata.cell_id.isin(df.index)][[x_col,y_col,cell_col,gene_col,tx_col]]
    intf_tx.columns = ['x','y','cell_id','gene','molecule_id']
    intf_tx = intf_tx.merge(
        df[['neaghbor_by_centroid','mask_distance']], left_on='cell_id', right_index=True
        )
    intf_tx['tx_geom'] = intf_tx[['x','y']].apply(lambda x: Point(x.iloc[0],x.iloc[1]),axis=1)
    intf_tx['mask_geom'] = cell_coords.loc[
        intf_tx.neaghbor_by_centroid.values, 'Geometry'].values
    intf_tx['tx_mask_distance'] = distance(intf_tx['tx_geom'], intf_tx['mask_geom'])
    intf_tx['celltype'] = cell_coords.loc[intf_tx.cell_id,'leiden'].values
    intf_tx['neaby_celltype'] = cell_coords.loc[intf_tx.neaghbor_by_centroid,'leiden'].values
    return intf_tx


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

def mix_norm_cdf(x, model):
    weights, means, covars = model.weights_, model.means_.reshape(-1), model.covariances_.reshape(-1)
    mcdf = 0.0
    for i in range(len(weights)):
        mcdf += weights[i] * norm.cdf(x, loc=means[i], scale=np.sqrt(covars[i]))
    return mcdf


def remove_tx(
        counts: pd.DataFrame, 
        cell_coords: pd.DataFrame, 
        intf_tx: pd.DataFrame, # returned by annotate_tx_mask_distance
        tx_mask_d_max: float = 2.0,
        hard_threshold: Optional[float]=None):

    intf_tx = intf_tx.copy()
    s_total = counts.groupby(cell_coords.leiden).count()
    s_above_0 = (counts>0).groupby(cell_coords.leiden).sum()
    percent_pos = s_above_0 / s_total

    intf_tx = intf_tx[intf_tx.tx_mask_distance<tx_mask_d_max].sort_values('tx_mask_distance')
    intf_tx['pct_exp_celltype'] = intf_tx.apply(
        lambda x: percent_pos.loc[x['celltype'],x['gene']], axis=1).values
    intf_tx['pct_exp_nearby'] = intf_tx.apply(
        lambda x: percent_pos.loc[x['neaby_celltype'],x['gene']], axis=1).values
    intf_tx['pct_diff'] = intf_tx.pct_exp_nearby - intf_tx.pct_exp_celltype
    
    if hard_threshold is None:
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
                                                gmm_dict["{}-{}".format(x['celltype'], x['neaby_celltype'])]), axis=1).values

        intf_tx['remove'] = np.random.binomial(1, intf_tx['remove_prob'])
    else:
        intf_tx['remove'] = (intf_tx['pct_diff']>=hard_threshold).astype(int)
    
    c_g_remove = intf_tx[intf_tx['remove']==1]
    # c_g_remove = intf_tx[intf_tx['pct_diff']>=0.25]
    c_g_remove = c_g_remove.groupby(c_g_remove.molecule_id).first()
    cells = c_g_remove.cell_id.unique()
    counts_to_subtract = counts.loc[cells,:].copy()
    removed_tx_ids = []
    removed_tx_new_cell_ids = []
    removed_gene_ids = []
    for c in cells:
        # Remove Tx from c
        removed_genes = c_g_remove.loc[c_g_remove.cell_id==c].gene.tolist()
        removed_gene_ids += removed_genes # actual gene names for reassigned transcripts
        removed_genes, remove_counts = np.unique(removed_genes, return_counts=True)
        counts_to_subtract.loc[c, removed_genes] -= remove_counts
        # Add Tx to c's neighbor
        removed_tx_ids += c_g_remove.loc[c_g_remove.cell_id==c].molecule_id.tolist() # get the transcripts that were reassigned
        removed_tx_new_cell_ids += c_g_remove.loc[c_g_remove.cell_id==c].neaghbor_by_centroid.tolist() # get the cell ids that each transcript were reassigned to

    counts_to_add = pd.DataFrame(1, index = removed_tx_ids, columns = ['count'])
    counts_to_add['cell_id'] = removed_tx_new_cell_ids
    counts_to_add['gene'] = removed_gene_ids
    counts_to_add = counts_to_add.groupby(['cell_id','gene']).sum().reset_index()
    counts_to_add = pd.pivot(counts_to_add, values = 'count', columns = 'gene', index = 'cell_id').fillna(0)

    return counts_to_subtract, counts_to_add, intf_tx









    