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
#%%
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
sc.tl.rank_genes_groups(xenium_ns, groupby='leiden',method='wilcoxon')
sc.pl.rank_genes_groups_dotplot(xenium_ns, n_genes=10, min_logfoldchange=0.6, swap_axes=True)
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
    plt.savefig(output_path + '/Dual_exp_{}.pdf'.format('+'.join(gene_pair)))
# %%
