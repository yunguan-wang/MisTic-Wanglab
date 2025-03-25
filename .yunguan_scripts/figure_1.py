#%%
import os
from geopandas import read_parquet,points_from_xy
import geopandas as gpd
import numpy as np
import scanpy as sc
import pandas as pd
import numpy as np
from shapely import Point, Polygon, distance
from scipy.spatial.distance import cdist
import matplotlib.pylab as plt
import seaborn as sns
# %matplotlib inline
# %%
def dual_exp_dot(plot_data, vmin=None, vmax=None, color_bar=True, ax=None, title=None):
    x_col, y_col = plot_data.columns
    plot_data = plot_data[plot_data.sum(axis=1)>0]
    plot_data = plot_data/plot_data.max()
    plot_data['x_bin'] = pd.cut(plot_data[x_col], 10).factorize(sort=True)[0]
    plot_data['y_bin'] = pd.cut(plot_data[y_col], 10).factorize(sort=True)[0]
    plot_data = plot_data.groupby(['x_bin','y_bin']).count().iloc[:,0].reset_index().astype(float)
    plot_data.iloc[:,:2] = (plot_data.iloc[:,:2].astype(float)+1.)/10.
    plot_data.columns = [x_col,y_col,'Fraction']
    plot_data['Fraction'] /= plot_data.Fraction.sum()
    plot_data['Fraction'] = np.log10(plot_data['Fraction'])
    if color_bar:
        fig, ax = plt.subplots(figsize=(4,4))
    sns.scatterplot(
        data = plot_data, x = x_col, y = y_col, hue='Fraction', palette='Reds',
        s=200, legend=None, ax=ax, marker='s')
    if color_bar:
        norm = plt.Normalize(
            vmin or plot_data['Fraction'].min(), vmax or plot_data['Fraction'].max())
        sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
        sm.set_array([])
        # Remove the legend and add a colorbar
        cax = fig.add_axes(
            [ax.get_position().x1+0.02, 0.11, 0.04, ax.get_position().height])
        ax.figure.colorbar(sm, cax=cax)
        cax.set_ylabel('log10(Fraction)')
    if title:
        ax.set_title(title)
    return plot_data, ax
# %%
input_path = '/data/wanglab/project/doublet_detection/data/'
output_path = '/data/wanglab/project/doublet_detection/results'
os.chdir(input_path)
sns.set_theme(style='white',font_scale=1.5)
plt.rcParams["legend.markerscale"] = 2
sc.settings.figdir = output_path
np.random.seed(2)
# %%
# read lung single cell rnaseq_data
lung_sc = sc.read_h5ad(input_path + '/scRNA_lung/scrna.h5ad')
lung_sc = lung_sc[lung_sc.obs.disease=='lung adenocarcinoma']
lung_sc = lung_sc[lung_sc.obs.origin=='tumor_primary']
luad = lung_sc[lung_sc.obs.dataset=='Kim_Lee_2020']
# luad = lung_sc[lung_sc.obs.assay=="10x 3' v2"]
luad.var_names = luad.var.feature_name.astype(str)
luad.X = luad.layers['count']
sc.pp.downsample_counts(luad, 1000)
sc.pp.normalize_per_cell(luad)
sc.pp.log1p(luad)
xenium_ns = sc.read_h5ad(input_path + '/xenium/lung_ns/processed_adata.h5ad')
xenium_ms = sc.read_h5ad(input_path + '/xenium/lung_ms/processed_adata.h5ad')
luad_bulk = pd.read_csv(
    input_path + '/bulk_lung/data_mrna_seq_v2_rsem.txt', sep='\t'
    ).dropna().set_index('Hugo_Symbol').drop('Entrez_Gene_Id', axis=1).T
luad_bulk = np.log1p(luad_bulk)
visium = sc.read_10x_h5(input_path + '/visium_lung/filtered.h5')
sc.pp.normalize_per_cell(visium)
sc.pp.log1p(visium)

# %%
gene_pair = ['CD3E','CD19']
for i in range(5):
    plot_data = [luad_bulk, visium, xenium_ns, xenium_ms, luad][i]
    if i == 0:
        plot_data = plot_data[gene_pair]
    else:
        plot_data = plot_data.to_df()[gene_pair]
    _ = dual_exp_dot(plot_data, vmax=np.log10(0.2))
# %%
gene_pair = ['CD3E','EPCAM']
for i in range(5):
    plot_data = [luad_bulk, visium, xenium_ns, xenium_ms, luad][i]
    if i == 0:
        plot_data = plot_data[gene_pair]
    else:
        plot_data = plot_data.to_df()[gene_pair]
    _ = dual_exp_dot(plot_data, vmax=np.log10(0.2), vmin=0)
# %%
for gene_pair in [
    ['CD3E','CD19'],
    ['CD3E','EPCAM'],
    ['MPEG1','EPCAM']]:
    fig, axes = plt.subplots(1, 5, figsize = (15,3), sharey=True)
    for i in range(5):
        plot_data = [luad_bulk, visium, xenium_ns, xenium_ms, luad][i]
        if i == 0:
            plot_data = plot_data[gene_pair]
        else:
            plot_data = plot_data.to_df()[gene_pair]
        _,ax = dual_exp_dot(
            plot_data, vmax=np.log10(0.2), vmin=-4, color_bar=False, ax=axes[i],
            title=['Bulk','Visium', 'Nuclei Seg.', 'Membrane Seg.', 'scRNA'][i])
    norm = plt.Normalize(-4,-1)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])
    # Remove the legend and add a colorbar
    cax = fig.add_axes(
        [ax.get_position().x1+0.02, 0.11, 0.04, ax.get_position().height])
    ax.figure.colorbar(sm, cax=cax)
    cax.set_ylabel('log10(Fraction)')
    plt.savefig(output_path + '/Dual_exp_{}.pdf'.format('+'.join(gene_pair)), bbox_inches='tight')
# %%
# %%
xenium_cn_g = [x for x in xenium_ns.var_names if x in xenium_ms.var_names]
luad_377 = luad[:,[x for x in luad.var_names if x in xenium_cn_g]].copy()
luad_377.obs.cell_type_predicted = luad_377.obs.cell_type_predicted.astype(str).replace(
    {
        'T cell CD8': 'T_cell',
        'T cell CD4': 'T_cell',
        'Endothelial cell':'Endothelial_cell',
        'Epithelial cell malignant': 'Tumor_cell',
        'B cell':'B_cell',
        'Alveolar cell type 1':'Tumor_cell'
    }
)
cts = ['B_cell', 'T_cell', 'Endothelial_cell','Macrophage', 'Tumor_cell', 'Fibroblast']
sc.tl.rank_genes_groups(
    luad_377, groupby='cell_type_predicted', pts=True, method='wilcoxon', use_raw=False)
deg = sc.get.rank_genes_groups_df(
    luad_377, 
    group = None, 
    log2fc_min=0.6, 
    pval_cutoff=0.05)
deg = deg[deg.group.isin(cts)]
deg = deg[deg.pct_nz_reference<=0.01]
deg = deg[deg.pct_nz_group>=0.05]
deg = deg[~deg.names.isin(['GDF15','MALL'])]
top_genes = deg.groupby('group').head(10).names.values
sc.pl.dotplot(
    luad_377[luad_377.obs.cell_type_predicted.isin(cts)], 
    var_names=top_genes, use_raw=False, groupby='cell_type_predicted', swap_axes=True,
    save='ME_genes.pdf'
)
deg.to_csv(output_path + '/luad_377_deg.csv')
luad_377.write_h5ad(output_path + '/luad_377.h5ad')
#%%
me_genes = {}
cmap = 'rocket'
luad_377.obs['leiden'] = luad_377.obs['cell_type_predicted']
for g, df in deg.groupby('group'):
    me_genes[g] = df.names[:5].tolist()
fig, axes = plt.subplots(
    1, 3, figsize = (15,5), sharey=True, layout="compressed")
for i, _adata in enumerate([luad_377, xenium_ms, xenium_ns]):
    df_exp = _adata.to_df()
    ctype = _adata.obs.leiden.astype(str)
    ctype = ctype.replace(
        {
            'Fibroblst.1':'Fibroblast',
            'Fibroblst.2':'Fibroblast',
            'T_cell_1':'T_cell',
            'Plasma_cell':'B_cell',
            'T_cell_2': 'T_cell',
            'Tumor': 'Tumor_cell',
            'Fibroblst': 'Fibroblast'
        })
    cosine_dist = pd.DataFrame(index=cts, columns=cts)
    for c in cts:
        _df = df_exp.loc[ctype==c]
        base_exp = _df[me_genes[c]].sum(axis=1)
        for g in cts:
            me_exp = _df[me_genes[g]].sum(axis=1)
            cs = 1 - cdist(base_exp.values.reshape(1,-1), me_exp.values.reshape(1,-1), metric='cosine')[0,0]
            cosine_dist.loc[c,g] = cs
    cosine_dist = cosine_dist.fillna(0).astype(float)
    sns.heatmap(cosine_dist, vmin=0, ax=axes[i], cbar=False, cmap=cmap,linewidths=0.1)
norm = plt.Normalize(0,1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
# Remove the legend and add a colorbar
cax = fig.add_axes([1.01, 0.4, 0.03, 0.58])
fig.colorbar(sm, cax=cax)
plt.savefig(output_path + '/ME_expression_overall.pdf')
# %%
plot_data = []
xenium_ns.obs.leiden = xenium_ns.obs.leiden.astype(str).replace(
    {
        'Fibroblst.1':'Fibroblast',
        'Fibroblst.2':'Fibroblast',
        'T_cell_1':'T_cell',
        'Plasma_cell':'B_cell',
        'T_cell_2': 'T_cell',
        'Tumor': 'Tumor_cell',
        'Fibroblst': 'Fibroblast'
    })
counts_xenium_ns = sc.read_10x_h5(
    '/data/wanglab/project/doublet_detection/data/xenium/lung_ns/cell_feature_matrix.h5')
counts_xenium_ns = counts_xenium_ns[xenium_ns.obs_names]
for nt_ct in ['B_cell', 'Endothelial_cell', 'Fibroblast', 'Macrophage', 'T_cell']:
    non_tumor_me = me_genes[nt_ct]
    dist_tumor = cdist(
    xenium_ns.obs.loc[xenium_ns.obs.leiden=='Tumor_cell', ['x_centroid','y_centroid']].values,
    xenium_ns.obs.loc[xenium_ns.obs.leiden==nt_ct, ['x_centroid','y_centroid']].values)
    dist_tumor = pd.DataFrame(
        dist_tumor, index=xenium_ns.obs.loc[xenium_ns.obs.leiden=='Tumor_cell'].index)
    dist_tumor_nc = dist_tumor.min(axis=1)
    dist_tumor_nc = dist_tumor_nc[dist_tumor_nc<=30]
    dist_tumor_me_exp = counts_xenium_ns.to_df().loc[dist_tumor_nc.index, non_tumor_me].sum(axis=1)
    tmp = pd.concat([dist_tumor_nc, dist_tumor_me_exp], axis=1)
    tmp.columns = ['nearest_nt_dist','sum.ntG_exp']
    tmp['Cell_type'] = nt_ct
    plot_data.append(tmp)
plot_data = pd.concat(plot_data)
# %%
sns.lmplot(
    data=plot_data[plot_data.Cell_type.isin(['Endothelial_cell', 'Macrophage', 'T_cell','Fibroblast'])], 
    x="nearest_nt_dist", y="sum.ntG_exp", hue='Cell_type',
    order=2,scatter=False, height=4
    )
plt.xlabel('Distance to nearest cell')
plt.ylabel('Sum. Counts \nNonTumor gene')
# plt.savefig(output_path + '/Tumor_ME_counts_vs_distance.pdf', bbox_inches='tight')
# %%
