#%%
import os
from geopandas import read_parquet, GeoDataFrame, GeoSeries,points_from_xy
import shapely
import numpy as np
import scanpy as sc
import pandas as pd
import numpy as np
from shapely import Point, Polygon, distance
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import adjustText
# %matplotlib inline
#%%
def plot_patch(
        adata, center=None, dx=2000,dy=2000, x_col='center_x', y_col='center_y', 
        color_col=['leiden'], save_prefix=False,**kwarg):
    if center is not None:
        x,y = center    
        plot_adata = adata[
            (adata.obs[x_col]>=x) &
            (adata.obs[x_col]<=x+dx) &
            (adata.obs[y_col]>=y) &
            (adata.obs[y_col]<=y+dy)
            ].copy()
        plot_adata.obs[x_col] -= x
        plot_adata.obs[y_col] -= y
    else:
        plot_adata = adata.copy()
    if isinstance(color_col,list):
        for col in color_col:
            if save_prefix:
                save = save_prefix + '_' + col
            else:
                save=None
            sc.pl.scatter(plot_adata, color=col, x=x_col,y=y_col, save=save, **kwarg)

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

def plot_boundaries(
    cell_coords, bbox = [2000, 4700, 800, 800], geom_col='polygon'):
    '''
    image windows in um, bot_left_x, top_left_y, x_length, y_length.
    adata must be the raw object.
    '''
    if bbox is not None:
        bounding_box = shapely.geometry.box(
            minx=bbox[0], 
            miny=bbox[1], 
            maxx=bbox[0] + bbox[2], 
            maxy=bbox[1] + bbox[3])
        cell_polys = cell_coords[cell_coords[geom_col].apply(lambda x: x.within(bounding_box))]
    else:
        cell_polys = cell_coords
    # cmaps = ['Reds', 'Blues']
    plt.figure(figsize=(10, 10), facecolor="white") 
    for _ , row in cell_polys.iterrows():
            shape = row[geom_col]
            plt.plot(*shape.exterior.xy, color=row["color"], linewidth=0.1)
            plt.fill(*shape.exterior.xy, color=row["color"], alpha=0.75)
# %%
input_path = '/data/wanglab/project/doublet_detection/merscope_hcc1'
os.chdir(input_path)
np.random.seed(2)
# %%
# All data is downloaded from VizGen, except for the spacia_meta, which is generated based on 
# in house cell typing.
# spacia_meta = pd.read_csv('hcc1_spacia_meta.txt', sep='\t', index_col=0)
# counts = pd.read_csv('cell_by_gene.csv', index_col=0)
# blanks = [x for x in counts.columns if x[:5] == 'Blank']
# counts = counts.drop(blanks, axis=1)
# counts.index = ['cell_' + str(x+1) for x in counts.index]
# counts = counts.loc[spacia_meta.index]
# # Construct shapely Polygon object from scratch.
# cell_masks = pd.read_csv('cell_coords.csv', index_col=0) # Cell masks were extracted from individual files from VizGen
# cell_masks.index = ['cell_' + str(x+1) for x in cell_masks.index]
# cell_masks = cell_masks.loc[spacia_meta.index]
# cell_masks['n_polygon'] = cell_masks.X.apply(lambda x: len(x.split('_')))
# cell_masks.X = cell_masks.X.str.split('_')
# cell_masks.Y = cell_masks.Y.str.split('_')
# cell_masks['polygon'] = cell_masks.apply(
#     lambda x: Polygon([(x.X[i], x.Y[i]) for i in range(x.n_polygon)]), axis=1)
# # Save as geopandas parquet file
# GeoDataFrame(
#     cell_masks[['polygon']],geometry='polygon'
#     ).to_parquet('cell_polygons.parquet', index=True)
#%%
spacia_meta = pd.read_csv('hcc1_spacia_meta.txt', sep='\t', index_col=0)
counts = pd.read_csv('cell_by_gene.csv', index_col=0)
blanks = [x for x in counts.columns if x[:5] == 'Blank']
counts = counts.drop(blanks, axis=1)
counts.index = ['cell_' + str(x+1) for x in counts.index]
counts = counts.loc[spacia_meta.index]
tx_meta = pd.read_csv('transcript_meta.csv', index_col=0)
cell_masks = read_parquet('cell_polygons.parquet')
adata = sc.AnnData(counts)
adata.obs = spacia_meta
adata.layers['Raw'] = adata.X.copy() # Save raw counts for later
# sc.pp.normalize_total(adata)
# sc.pp.log1p(adata)
# sc.pp.scale(adata)
# sc.tl.pca(adata)
# sc.pp.neighbors(adata)
# sc.tl.umap(adata)
#%%
# You can start from here if you saved your adata file.
# adata = sc.read_h5ad('hcc1.h5ad')
# polygons = read_parquet('cell_polygons.parquet')
# spacia_meta = pd.read_csv('hcc1_spacia_meta.txt', sep='\t', index_col=0)
# %%
## Cannot work on the full dataset due to memory size issue
## See if you can do it on the full dataset, otherwise run this in blocks just to get the cell-cell mask distance
adata_uq = adata[(adata.obs.X<=6000) & ((adata.obs.Y>=6000))].copy() # subsetted upper quadrant
plot_patch(adata, x_col='X', y_col='Y', color_col=['cell_type'])
cell_dist = cdist(
    adata_uq[adata_uq.obs.cell_type=='Tumor_cells'].obs[['X','Y']],
    adata_uq[adata_uq.obs.cell_type!='Tumor_cells'].obs[['X','Y']])
cell_dist = pd.DataFrame(
    cell_dist, 
    index = adata_uq[adata_uq.obs.cell_type=='Tumor_cells'].obs_names, 
    columns = adata_uq[adata_uq.obs.cell_type!='Tumor_cells'].obs_names)
cell_meta = spacia_meta.copy()
cell_meta['polygon'] = cell_masks.loc[cell_meta.index,'polygon'].values
colors = sns.color_palette('hls', n_colors=cell_meta.cell_type.nunique()).as_hex()
color_dict = pd.Series(
    colors, index=cell_meta.cell_type.unique()
)
cell_meta['color'] = color_dict.loc[cell_meta.cell_type].values
# Calculate distance of tumor cells to nearest non-tumor cells based on mask
mask_distance = calculate_mask_distance(
    cell_dist,
    cell_meta,
    max_centroid_dist=30,
    x_col='X',
    y_col='Y',
    cluster_col = 'cell_type',
    geometry_col='polygon')
# Get rid of tumors with nearby non-tumor cells
interface_tumor = mask_distance[mask_distance.mask_distance<=2].index.unique()
adata_orphan = adata_uq[~adata_uq.obs_names.isin(interface_tumor)]
# Replacing tumors cells at the interface with non-tumors with non-tumor cells

# Randomly swap the expression profile and identify of these cells
# Pick the non-tumor cells as replacements
non_tumor_samples = adata_uq[
    adata_uq.obs.cell_type!='Tumor_cells'].to_df().sample(len(interface_tumor)).index
sample_celltypes = adata_uq.obs.loc[non_tumor_samples,'cell_type'].astype(str).values
# Constract the interface adata with swapped cells
# The interace cells will have the same cell id but swapped gene expression.
adata_interface = adata_uq[interface_tumor].copy()
adata_interface.X = adata_uq[non_tumor_samples].layers['Raw']
adata_interface.layers['Raw'] = adata_interface.X.copy()
adata_interface.obs.cell_type = sample_celltypes
# Construct the synthetic data by merging the orphan tumors adata and the synthetic interface adata
adata_synthetic = adata_orphan.copy()
adata_synthetic.X = adata_synthetic.layers['Raw']
adata_synthetic = sc.concat([adata_synthetic, adata_interface])
# Remake the metadata
cell_meta_synthetic = cell_meta.copy()
cell_meta_synthetic.loc[interface_tumor, 'cell_type'] = sample_celltypes
cell_meta_synthetic['color'] = color_dict.loc[cell_meta_synthetic.cell_type].values

# Identify tumor cells to add transcripts to
# Need to recalculate tumor to non tumor cell distance as the cell identify was changed
cell_dist_synthetic = pd.DataFrame(
    cdist(
        adata_synthetic[adata_synthetic.obs.cell_type=='Tumor_cells'].obs[['X','Y']],
        adata_synthetic[adata_synthetic.obs.cell_type!='Tumor_cells'].obs[['X','Y']]
        ),
    index = adata_synthetic[adata_synthetic.obs.cell_type=='Tumor_cells'].obs_names, 
    columns = adata_synthetic[adata_synthetic.obs.cell_type!='Tumor_cells'].obs_names
    )
mask_distance_synthetic = calculate_mask_distance(
    cell_dist_synthetic,
    cell_meta_synthetic,
    max_centroid_dist=30,
    x_col='X',
    y_col='Y',
    cluster_col = 'cell_type',
    geometry_col='polygon')
# Here we use a much smaller mask distance to simulate contacted masks, can try even smaller size
interface_synthetic = mask_distance_synthetic[
    mask_distance_synthetic.mask_distance<=0.5].copy() 

interface_synthetic['tumor_cell'] = interface_synthetic.index
interface_synthetic = interface_synthetic.sort_values('mask_distance')
# only simulated TX from one neighbor. 
# TXs from another neighbor is conceptually equivalent and computationally a simple repeat.
interface_synthetic.drop_duplicates('tumor_cell', inplace=True)
n_points = 200
# doublet level means the fraction of cell in the interaface to have at least one added TX
doublet_level = 0.4
max_noise_tx = 10
simulated_points = {}
# manipulated_cells = []
neighbors = []
for tumor_cell in interface_synthetic.index:
    if np.random.uniform(0,1,1)[0]>doublet_level:
        continue
    neighbor = interface_synthetic.loc[tumor_cell,'neaghbor_by_centroid']
    tumor_p = cell_meta_synthetic.loc[tumor_cell,'polygon']
    neighbor_p = cell_meta_synthetic.loc[neighbor,'polygon']
    points = GeoSeries(tumor_p).sample_points(size=n_points).iloc[0]
    dist_to_neighbor = distance(
        points.geoms, np.repeat(neighbor_p,n_points)
    )
    valid_points = np.array(points.geoms)[dist_to_neighbor<2][:np.random.randint(1,max_noise_tx)].tolist()
    if len(valid_points)==0:
        continue
    simulated_points[tumor_cell]= valid_points
    neighbors += [neighbor] * len(valid_points)
#%%
# Differential analysis
sc.pp.normalize_total(adata, target_sum=1000)
sc.pp.log1p(adata)
sc.tl.rank_genes_groups(adata, 'cell_type', pts=True)
deg = sc.get.rank_genes_groups_df(adata, None)
deg = deg.merge(deg[deg.group=='Tumor_cells'][['names', 'pct_nz_group']],on='names')
# Filter the deg that can be added to synthetic target cells
deg = deg[
    (deg.pvals_adj<=0.05) & 
    (deg.logfoldchanges>=0.6) &
    # Here I limit to genes not highly expressed in tumor cells, because some of these cells
    # could still be contaminated by other cells   
    (deg.pct_nz_group_y <= 0.2) & 
    ((deg.pct_nz_group_x - deg.pct_nz_reference)>=0.2)
]
ct_deg = deg.groupby('group').names.agg(lambda x : x.tolist())
# Randomly assign gene symbol for each simulated transcript
interface_synthetic['neighbor_ct'] = cell_meta_synthetic.loc[
    interface_synthetic.neaghbor_by_centroid.values, 'cell_type'].values
interface_synthetic.loc[simulated_points.keys(),'synthetic_tx'] = pd.Series(
    [*simulated_points.values()]).values
interface_synthetic = interface_synthetic.dropna()
interface_synthetic['tx_symbol'] = interface_synthetic.apply(
    lambda x: np.random.choice(
        ct_deg.loc[x.neighbor_ct], len(x.synthetic_tx)
        ).tolist(),
    axis=1)
synthetic_txs = pd.DataFrame(
    interface_synthetic.synthetic_tx.explode())
synthetic_txs['gene'] = interface_synthetic.tx_symbol.explode()
synthetic_txs['cell'] = synthetic_txs.index
synthetic_txs['num'] = 1
synthetic_txs['nearest_neighbor'] = neighbors
count_delta = synthetic_txs.groupby(['cell','gene']).num.count()
count_delta = count_delta.reset_index().pivot(
    columns='gene', values='num', index='cell').fillna(0).astype(int)

# Update count matrix in synthetic adata
synthetic_counts = adata_synthetic.to_df()
synthetic_counts.loc[count_delta.index, count_delta.columns] += count_delta.values
adata_synthetic.X = synthetic_counts.values

# Check if the added genes are indeed lowly expressed in target tumor cells
print(
    'Synthetic tx expression in target cells in original data:', 
    adata_synthetic.to_df().loc[count_delta.index, count_delta.columns].sum().sum())
print(
    'Synthetic tx expression in target cells total synthetic original_counts:', 
    count_delta.sum().sum())
print(
    'random selected, equal-sized sets of genes in target cells original original_counts:', 
    adata_synthetic.to_df().loc[
        count_delta.index, 
        np.random.choice(adata_synthetic.to_df().columns, count_delta.shape[1])
        ].sum().sum())
#%%
tx_meta_interface = tx_meta[tx_meta.cell.isin(interface_synthetic.index)].copy()
tx_meta_interface['tx_type'] = 'real'
tx_meta_interface['point'] = points_from_xy(tx_meta_interface.global_x,tx_meta_interface.global_y)
tx_meta_interface.rename({'cell':'cell_id'}, inplace=True, axis=1)
synthetic_tx_meta = synthetic_txs.copy().drop('num',axis=1)
synthetic_tx_meta.columns = ['point', 'gene', 'cell_id','neighbor']
synthetic_tx_meta['tx_type'] = 'synthetic'
synthetic_tx_meta.index = ['tx_s_' + str(i+1) for i in range(synthetic_tx_meta.shape[0])]
tx_meta_interface = pd.concat([tx_meta_interface, synthetic_tx_meta])
tx_meta_interface = GeoDataFrame(tx_meta_interface, geometry='point')
keep = ['gene', 'point', 'cell_id', 'tx_type', 'x', 'y','neighbor']
tx_meta_interface = tx_meta_interface[keep]
tx_meta_interface['x'] = tx_meta_interface.point.apply(lambda x: np.array(x.xy).flatten()[0])
tx_meta_interface['y'] = tx_meta_interface.point.apply(lambda x: np.array(x.xy).flatten()[1])
synthetic_counts.to_csv('synthetic_counts_with_doublets.csv')
tx_meta_interface.to_parquet('synthetic_counts_tx_metadata.parquet', index=True)
cell_meta_synthetic.loc[synthetic_counts.index].to_csv('synthetic_counts_with_doublets_metadata.csv')
#%%
# test plotting for debug purposes
# for cell in tx_meta_interface[tx_meta_interface.tx_type=='synthetic'].cell.value_counts().index[:5]:
#     _ = plt.figure()
#     cell_tx = tx_meta_interface[tx_meta_interface.cell==cell]
#     cell_poly = cell_masks.loc[cell,'polygon']
#     xs, ys = cell_poly.exterior.xy
#     plt.fill(xs, ys, alpha=0.5, fc='b', ec='none')
#     texts = []
#     for g, row in cell_tx.iterrows():
#         added_genes = cell_tx[cell_tx.tx_type!='real'].gene.unique()
#         try:
#             x, y = [float(i) for i in row['point'].split('(')[1][:-1].split(' ')]
#         except:
#             x, y = np.array(row['point'].xy).flatten()
#         plt.scatter(x, y, c= 'k' if row['tx_type'] == 'real' else 'r')
#         if row.gene in added_genes:
#             texts.append(plt.text(x, y, row.gene, c='m'))
#     adjustText.adjust_text(texts)
# %%
