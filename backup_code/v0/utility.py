import numpy as np
import scanpy as sc
from geopandas import read_parquet
import pandas as pd
from scipy.spatial.distance import cdist
from shapely import Point, Polygon, distance
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM
from typing import Optional


def import_data(h5ad_path, 
                detected_transcripts_path,
                cell_metadata_path,
                cell_by_gene_path,
                cell_boundaries_path):
    # Read annData
    adata = sc.read_h5ad(h5ad_path)
    # Read transcript data 
    tx_metadata = pd.read_csv(detected_transcripts_path, index_col=0)
    cell_meta = pd.read_csv(cell_metadata_path, index_col=0)
    # cell by gene count matrix 
    counts = pd.read_csv(cell_by_gene_path, index_col=0)
    # Cell boundary
    cell_coords = read_parquet(cell_boundaries_path)
    # We filter out gene columns with "Blank"
    blanks = [x for x in counts.columns if 'Blank' in x]
    counts = counts.drop(blanks, axis=1)
    # As cell ids are different in cell_meta and counts 
    cell_id_mapping = pd.Series(
        ['cell_' + str(x) for x in range(len(counts))],
        index = counts.index,
    )
    counts.index = cell_id_mapping.loc[counts.index]
    counts = counts.loc[adata.obs_names]
    cell_meta.index = cell_id_mapping[cell_meta.index]
    cell_meta = cell_meta.loc[adata.obs_names]
    tx_metadata = tx_metadata[tx_metadata.cell_id!=0]
    tx_metadata = tx_metadata[~tx_metadata.gene.isin(blanks)]
    tx_metadata.cell_id = cell_id_mapping[tx_metadata.cell_id].values
    tx_metadata = tx_metadata[tx_metadata.cell_id.isin(adata.obs_names)]
    tx_metadata['molecule_id'] = ['m' + str(i+1) for i in range(len(tx_metadata))]
    adata.obs['x'] = cell_meta.loc[adata.obs_names,'center_x']
    adata.obs['y'] = cell_meta.loc[adata.obs_names,'center_y']

    metadata = adata.obs
    entity_ids = metadata.index.values
    cell_coords.EntityID = cell_id_mapping.loc[cell_coords.EntityID].values
    cell_coords = cell_coords[cell_coords.EntityID.isin(entity_ids)]
    cell_coords = cell_coords.drop_duplicates(subset=['EntityID','Geometry'])
    cell_coords.index = cell_coords.EntityID

    return adata, counts, tx_metadata, cell_coords
    






def mix_norm_cdf(x, model):
    weights, means, covars = model.weights_, model.means_.reshape(-1), model.covariances_.reshape(-1)
    mcdf = 0.0
    for i in range(len(weights)):
        mcdf += weights[i] * norm.cdf(x, loc=means[i], scale=np.sqrt(covars[i]))
    return mcdf


def calculate_mask_distance(
    centroid_dist: pd.DataFrame,
    cell_coords: pd.DataFrame,
    max_centroid_dist: int = 15,
    min_centroid_dist: int = 0,
    cluster_col:str = 'leiden',
    x_col:str = 'x',
    y_col:str = 'y',
    geometry_col:str = 'Geometry',
    re_cal_centroid_dist:bool = False
    ):
    '''
    centroid_dist: cell by cell distance matrix by centroid location, 
    max_centroid_dist: maximal distance to consider neighbors, 
    max_centroid_dist: minimal distance to consider distant cells, 
    cell_coords: cell by cell distance matrix by centroid location, 
    centroid_dist: cell by cell distance matrix by centroid location, 
    '''
    adj = (centroid_dist > min_centroid_dist) & (centroid_dist<= max_centroid_dist)
    adj = adj.agg(
        lambda x: adj.columns[x.values].tolist(), axis=1)
    adj = pd.DataFrame(adj, columns = ['n'])
    # pandas suggest using transform, but it did not work.
    adj = adj.agg(
        lambda x: [
            y for y in x.n if cell_coords.loc[y, cluster_col]!=cell_coords.loc[x.name, cluster_col]
            ], axis=1)
    adj_nonself = adj[adj.apply(len)>0]
    adj_nonself_masks_ids = adj_nonself.explode()
    md = distance(
        cell_coords.loc[adj_nonself_masks_ids.index, geometry_col].values,
        cell_coords.loc[adj_nonself_masks_ids.values, geometry_col].values
    )
    masks_distances = pd.DataFrame(
        adj_nonself_masks_ids, columns=['neaghbor_by_centroid'])
    masks_distances['mask_distance'] = md
    # recalculating the centroid distance is expansive, so it should be avoided
    if re_cal_centroid_dist:
        v1 = cell_coords.loc[masks_distances.index, [x_col, y_col]].values
        v2 = cell_coords.loc[masks_distances.neaghbor_by_centroid, [x_col, y_col]].values
        masks_distances['centroid_distance'] = np.sum((v1-v2)**2, axis=1)**0.5
    return masks_distances

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
        lambda x: percent_pos.loc[x['cell_type'],x['gene']], axis=1).values
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
                                                gmm_dict["{}-{}".format(x['cell_type'], x['neaby_celltype'])]), axis=1).values

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


def process_counts(df: pd.DataFrame, target_sum = 200):
    df = sc.AnnData(df)
    sc.pp.normalize_total(df, target_sum=target_sum)
    sc.pp.log1p(df)
    return df.to_df()


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
    intf_tx['cell_type'] = cell_coords.loc[intf_tx.cell_id,'leiden'].values
    intf_tx['neaby_celltype'] = cell_coords.loc[intf_tx.neaghbor_by_centroid,'leiden'].values
    return intf_tx









    
# def remove_tx(
#         counts: pd.DataFrame, 
#         cell_coords: pd.DataFrame, 
#         intf_tx: pd.DataFrame, # returned by annotate_tx_mask_distance
#         tx_mask_d_max: float = 2.0,
#         hard_threshold: Optional[float]=None):
#     intf_tx = intf_tx.copy()
#     s_total = counts.groupby(cell_coords.leiden).count()
#     s_above_0 = (counts>0).groupby(cell_coords.leiden).sum()
#     percent_pos = s_above_0 / s_total

#     intf_tx = intf_tx[intf_tx.tx_mask_distance<tx_mask_d_max].sort_values('tx_mask_distance')

#     intf_tx['pct_exp_celltype'] = intf_tx.apply(
#         lambda x: percent_pos.loc[x['cell_type'],x['gene']], axis=1).values
#     intf_tx['pct_exp_nearby'] = intf_tx.apply(
#         lambda x: percent_pos.loc[x['neaby_celltype'],x['gene']], axis=1).values
#     intf_tx['pct_diff'] = intf_tx.pct_exp_nearby - intf_tx.pct_exp_celltype

#     if hard_threshold is None:
#         gmm_dict = {}
#         for cell_type0 in percent_pos.index:
#             for cell_type1 in percent_pos.index:
#                 if cell_type0 == cell_type1:
#                     continue
#                 # 0 vs other (if > 0 then gene expressed)
#                 # only consider > 0
#                 fold_change = np.log2(percent_pos.loc[cell_type0,:]+1) - np.log2(percent_pos.loc[cell_type1, :]+1)
#                 model = GMM(2)
#                 model.fit(fold_change.values.reshape((-1,1)))
#                 gmm_dict["{}-{}".format(cell_type0, cell_type1)] = model
#         intf_tx['remove_prob'] = intf_tx.apply(lambda x: mix_norm_cdf(x['pct_diff'],
#                                                 gmm_dict["{}-{}".format(x['cell_type'], x['neaby_celltype'])]), axis=1).values

#         intf_tx['remove'] = np.random.binomial(1, intf_tx['remove_prob'])
#     else:
#         intf_tx['remove'] = (intf_tx['pct_diff']>=hard_threshold).astype(int)

#     c_g_remove = intf_tx[intf_tx['remove']==1]
#     c_g_remove = c_g_remove.groupby(c_g_remove.molecule_id).first()
#     c_g_remove = c_g_remove.groupby('cell_id').gene.agg(lambda x: x.tolist())

#     cells = c_g_remove.index
#     cell_counts = counts.loc[cells,:].copy()
#     for c in cells:
#         remove = c_g_remove[c]
#         remove, remove_counts = np.unique(remove, return_counts=True)
#         cell_counts.loc[c, remove] -= remove_counts
#     return cell_counts, intf_tx