
#%%
import os
from geopandas import read_parquet
import shapely
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.spatial.distance import cdist
from shapely import Point, Polygon, distance
import seaborn as sns
from PIL import Image
from scipy.stats import entropy
#%%
def plot_boundaries(
          cell_coords, cg_pairs, adata, bbox = [2000, 4700, 800, 800]):
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
        cell_polys = cell_coords[cell_coords['Geometry'].within(bounding_box)]
    else:
        cell_polys = cell_coords
    # cmaps = ['Reds', 'Blues']
    annotate_cells = []
    annotate_colors = []
    for cg in cg_pairs:
        _cells = adata[adata.obs.leiden==cg[0]].obs_names
        _cells = [x for x in _cells if x in cell_polys.index]
        _cell_color = cell_polys.loc[_cells,'color'][0]
        annotate_cells += _cells
        _gene_v = adata.to_df().loc[_cells, cg[1]]
        norm = mpl.colors.Normalize(vmin=-0.25, vmax=0.75, clip=True)
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            '', ['#ffffff', _cell_color])
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        gene_colors = list(map(lambda x: mpl.colors.to_hex(x), mapper.to_rgba(_gene_v)))
        annotate_colors += gene_colors
    gene_colors = pd.DataFrame(annotate_colors, index = annotate_cells, columns=['fill_color'])
    try:
        cell_polys.drop('fill_color',axis=1, inplace=True)
    except:
        pass
    cell_polys = cell_polys.merge(gene_colors, left_index=True, right_index=True, how='left')
    cell_polys.fill_color.fillna('w', inplace=True)
    plt.figure(figsize=(10, 10), facecolor="white") 
    for _ , row in cell_polys.iterrows():
            shape = row["Geometry"]
            if shape.geom_type.startswith('Multi'):
                for geom in shape.geoms:
                    plt.plot(*geom.exterior.xy, color=row["color"], linewidth=2)
                    plt.fill(*geom.exterior.xy, color=row["fill_color"], alpha=0.75)
    handles = []                
    for g, row in color_dict.iterrows():
        handles.append(mpl.patches.Patch(color=row.color, label=g))
    plt.legend(handles=handles, loc = 'center left', bbox_to_anchor=(1,0.5))

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
        ):
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

    c_g_remove = intf_tx[intf_tx['pct_diff']>=0.25]
    c_g_remove = c_g_remove.groupby(c_g_remove.molecule_id).first()
    c_g_remove = c_g_remove.groupby('cell_id').gene.agg(lambda x: x.tolist())

    cells = c_g_remove.index
    cell_counts = counts.loc[cells,:].copy()
    for c in cells:
        remove = c_g_remove[c]
        remove, remove_counts = np.unique(remove, return_counts=True)
        cell_counts.loc[c, remove] -= remove_counts
    return cell_counts, intf_tx

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
    intf_tx['celltype'] = cell_coords.loc[intf_tx.cell_id,'leiden'].values
    intf_tx['neaby_celltype'] = cell_coords.loc[intf_tx.neaghbor_by_centroid,'leiden'].values
    return intf_tx
#%%
input_path = '/data/wanglab/miethke/merscope_psc/'
output_path = os.path.join('/data/wanglab/miethke/merscope_psc/', 'results')
os.chdir(input_path)
np.random.seed(2)
# adata = sc.read_h5ad('merged_merscope_baysor.h5ad')
# adata = adata[adata.obs.sample_id=='B1']
adata = sc.read_h5ad('B1/processed_merscope.h5ad')
# sc.tl.leiden(
#     adata, resolution=0.5, restrict_to=('leiden',['fibroblasts']), key_added='leiden_sub')
# adata.obs.leiden = adata.obs.leiden_sub
# sc.pl.umap(adata, color='leiden_sub')
clusters = adata.obs.leiden.astype(str)
adata.obs['celltypes'] = clusters

#%%
tx_metadata = pd.read_csv(
    'B1/detected_transcripts.csv', index_col=0)
cell_meta = pd.read_csv('B1/cell_metadata.csv', index_col=0)
counts = pd.read_csv('B1/cell_by_gene.csv', index_col=0)
blanks = [x for x in counts.columns if 'Blank' in x]
counts = counts.drop(blanks, axis=1)
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
#%%
# Process cell boundaries
sample_id ='B1'
cell_coords = read_parquet('{}/cell_boundaries.parquet'.format(sample_id))
metadata = adata.obs
entity_ids = metadata.index.values
cell_coords.EntityID = cell_id_mapping.loc[cell_coords.EntityID].values
cell_coords = cell_coords[cell_coords.EntityID.isin(entity_ids)]
cell_coords = cell_coords.drop_duplicates(subset=['EntityID','Geometry'])
cell_coords.index = cell_coords.EntityID
#%%
cell_coords = cell_coords.merge(
    adata.obs[['x','y','leiden']], right_index=True, left_index=True, how='left')
cell_dist = cdist(adata.obs[['x','y']],adata.obs[['x','y']])
cell_dist = pd.DataFrame(cell_dist, index = adata.obs_names, columns=adata.obs_names)
# It is now evident masks can still be very close even when centroid distance is above 40
mask_distance  = calculate_mask_distance(
    cell_dist, cell_coords, max_centroid_dist=75, re_cal_centroid_dist=True)
sns.scatterplot(
    data = mask_distance, x='centroid_distance', y='mask_distance', s= 5)

#%%
mask_dist_cutoff = 1
interface = mask_distance[mask_distance.mask_distance<=mask_dist_cutoff]
intf_tx = annotate_tx_mask_distance(
    interface, tx_metadata, cell_coords)
updated_counts, intf_tx = remove_tx(counts, cell_coords, intf_tx)
# percent_pos.T.sort_values('NonInfMphage', ascending=False).head(25)
#%%
corrected_counts = counts.copy()
corrected_counts.loc[updated_counts.index,updated_counts.columns] = updated_counts.values
corrected_adata = sc.AnnData(corrected_counts)
corrected_adata.obs = adata.obs.loc[corrected_adata.obs_names]
corrected_adata.obs['celltype'] = adata.obs.loc[corrected_adata.obs_names, 'leiden']
sc.pp.normalize_total(corrected_adata, target_sum=1000)
sc.pp.log1p(corrected_adata)
corrected_adata.raw = corrected_adata.copy()
sc.pp.scale(corrected_adata)
sc.pp.pca(corrected_adata)
sc.pp.neighbors(corrected_adata, n_pcs=20)
sc.tl.leiden(corrected_adata, resolution=0.5)
sc.tl.umap(corrected_adata)
sc.pl.umap(corrected_adata, color=['leiden','celltype'])
#%%

crit = mask_distance.sort_values('mask_distance')
crit = crit[~crit.index.duplicated(keep='first')]
interface_cells = crit[crit.mask_distance<=0.1].index.tolist()
isolated_cells = crit[crit.mask_distance>=5].index.tolist()
training_set = np.random.choice(corrected_adata.obs_names, size=5000, replace=False)

#%%
# train_adata = adata[training_cells].raw.to_adata()
# sc.tl.pca(train_adata)
# # sc.pp.scale(train_adata)
# sc.pp.neighbors(train_adata, n_pcs=25, n_neighbors=5)
# sc.tl.umap(train_adata)
# sc.pl.umap(train_adata, color='leiden')

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn.decomposition import PCA
x_train = corrected_adata[training_set].raw.to_adata().to_df()
y_train = corrected_adata.obs.loc[training_set, 'leiden'].astype(str).values
# pca_model = PCA(n_components=25).fit(x_train)
# x_train = pca_model.transform(x_train)
# classifier = KNeighborsClassifier(n_neighbors=10)
classifier = LogisticRegression(max_iter=10000)
# xgboots_params = {
#     "n_estimators": 400,
#     "max_leaf_nodes": 5,
#     "max_depth": 10,
#     "random_state": 0,
#     "min_samples_split": 5,
#     "learning_rate": 0.2, 
#     "subsample": 0.5
# }
# classifier = ensemble.GradientBoostingClassifier(**xgboots_params)
classifier.fit(x_train, y_train)
#%%
# # simulate doublets
# simulated_type = []
# simulated_probs = []
# core_pca_avg = pd.DataFrame(x_train).groupby(y_train).mean()
# for fraction_2nd in [0.1,0.3,0.5]:
#     doublet = []
#     for i in range(1500):
#         core_idx = np.random.randint(0, len(x_train))
#         core_ct = y_train[core_idx]
#         core_pca = x_train[core_idx]
#         tmp = x_train[y_train!=core_ct]
#         mix_pca = tmp[np.random.randint(0, len(tmp)),:]
#         simulated_cell = (1-fraction_2nd)*core_pca + fraction_2nd*mix_pca
#         doublet.append(simulated_cell)
#         simulated_type.append('Mixing fraction {}'.format(fraction_2nd))
#     doublet = np.array(doublet).reshape(-1,25)
#     probs = classifier.predict_proba(doublet)
#     simulated_probs.append(probs)
# simulated_probs = np.array(simulated_probs).reshape(4500,-1)
# %%
interface_test = adata[interface_cells].raw.to_adata().to_df()
isolated_test = adata[isolated_cells].raw.to_adata().to_df()
rest_uncorrected = [
    x for x in updated_counts.index if x not in interface_cells + isolated_cells + training_set.tolist()]
corrected = corrected_adata[rest_uncorrected].raw.to_adata().to_df()
uncorrected = adata[rest_uncorrected].raw.to_adata().to_df()

corrected_probs = classifier.predict_proba(corrected)
uncorrected_probs = classifier.predict_proba(uncorrected)
interface_probs = classifier.predict_proba(interface_test)
isolated_probs = classifier.predict_proba(isolated_test)

#%%
entropies = pd.DataFrame(
    entropy(
        np.concatenate(
            [corrected_probs, uncorrected_probs, interface_probs, isolated_probs]
            ),
            axis=1
        ),
    columns = ['entropy'])
entropies['type'] = ['corrected']*len(corrected) + ['uncorrected']*len(corrected) + ['interface']*len(interface_probs) + ['isolated']*len(isolated_probs)
sns.displot(data = entropies, x = 'entropy', kind="kde", hue='type',common_norm=False)
#%%
entropy_diff = entropy(uncorrected_probs,axis=1) - entropy(corrected_probs,axis=1)
# plt.hist(entropy_diff, bins=50)
# _= plt.figure()
sns.scatterplot(
    x = entropy_diff, 
    y = entropy(uncorrected_probs,axis=1), 
    s = 10)
plt.xlabel('Entropy diff uncorrected - correted')
plt.ylabel('Entropy uncorrected')

# %%
# Plot cell, tx and neighborhs
fig, axs = plt.subplots()
cell = 'cell_47348'
cell_geom = cell_coords.loc[cell, 'Geometry']
neighbor_geoms = intf_tx[intf_tx.cell_id==cell]['mask_geom'].unique()
txs = intf_tx[intf_tx.cell_id==cell].sort_values('tx_mask_distance')
txs = txs.groupby(txs.molecule_id).first()
tx = txs['tx_geom'].values
tx_dist = txs['tx_mask_distance'].values
tx_pct_diff = txs['pct_diff'].values
for geom in cell_geom.geoms:
    xs, ys = geom.exterior.xy
    axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')

for m in neighbor_geoms:
    for geom in m.geoms:  
        xs, ys = geom.exterior.xy
        axs.fill(xs, ys, alpha=0.5, fc='b', ec='none')
xs = []
ys = []
for t in tx:
    x, y = t.xy    
    xs.append(x[0])
    ys.append(y[0])
# sns.scatterplot(x = xs, y = ys, hue=tx_dist)
sns.scatterplot(x = xs, y = ys, hue=tx_pct_diff, palette='coolwarm')
# %%


# %%
#%%
# adata = adata.raw.to_adata()
# adata.raw = adata.copy()
# sc.pp.scale(adata)
# sc.tl.pca(adata)
# sc.pp.neighbors(adata, n_pcs=20)
# # sc.pl.pca_overview(adata)
# sc.tl.leiden(adata, resolution=0.5)
# sc.tl.umap(adata)
# sc.pl.umap(adata, color=['leiden','celltypes'])
# Cell makss
#%%
# Baysor segmentated
# baysor_res_a1 = pd.read_csv(
#     'A1/baysor/segmentation.csv', index_col=0)
# baysor_res_b1 = pd.read_csv(
#     'B1/baysor/segmentation.csv', index_col=0)
# baysor_meta = pd.read_csv('B1/baysor/segmentation_cell_stats.csv', index_col=0)
# baysor_res_b1 = baysor_res_b1[baysor_res_b1.confidence>=0.95]
# baysor_res_b1 = baysor_res_b1[baysor_res_b1.assignment_confidence>=0.9]
# baysor_res_b1 = baysor_res_b1[baysor_res_b1.cell.isin(adata.obs_names)]
# counts = baysor_res_b1.groupby(['cell','gene']).count().iloc[:,0]
# counts = counts.reset_index().pivot(index='cell', columns='gene', values='x').fillna(0)