#%%
import pandas as pd
import scanpy as sc
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist
import sys
from scipy.stats import entropy
sys.path.insert(1,'/users/wanzx2/codes/quality_of_life_improvement')
from scrna_utils import *
from spatial_utils import *
# %%
input_path = '/data/wanglab/miethke/merscope_psc/'
output_path = os.path.join('/data/wanglab/miethke/merscope_psc/', 'results')
os.chdir(input_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
sns.set_theme(style='white',font_scale=1.5)
plt.rcParams["legend.markerscale"] = 2
sc.settings.figdir = output_path
np.random.seed(2)
adata = sc.read_h5ad('merged_merscope_baysor.h5ad')
adata.obsm['spatial'] = adata.obs[['x','y']].values
adata.uns['spatial'] = adata.obs[['x','y']].values
adata.uns['log1p']['base'] = None
adata = adata[adata.obs.leiden!='PlasmaBcell']
adata = adata[adata.obs.sample_id == 'B1']
baysor_meta = pd.read_csv('B1/baysor/segmentation_cell_stats.csv', index_col=0)
baysor_exp = pd.read_csv('B1/baysor/segmentation.csv', index_col=0)
baysor_exp = baysor_exp[baysor_exp.cell.isin(adata.obs_names)]
# %%
cell_dist = cdist(adata.obs[['x','y']],adata.obs[['x','y']])
cell_dist = pd.DataFrame(cell_dist, index = adata.obs_names, columns =adata.obs_names)

# Get nearest cell type based on distance
# Too slow, needs optimization
# nearest_ctype = cell_dist.apply(
#     lambda x: x[adata.obs['leiden']!=adata.obs.loc[x.name,'leiden']], 
#     axis=1)
# nearest_ctype.loc[:,:] = np.argpartition(nearest_ctype.fillna(np.inf), 5, axis=1)
# nearest_ctype = nearest_ctype.iloc[:,:5].astype(int).agg(
#     lambda x: adata.obs.loc[adata.obs_names[x],'leiden'].astype(str).unique(), axis=1)
#
nearest_dist_nonself = cell_dist.apply(
    lambda x: x[adata.obs['leiden']!=adata.obs.loc[x.name,'leiden']].min(), 
    axis=1)
interfaces = nearest_dist_nonself.sort_values().groupby(adata.obs.leiden.astype(str)).head(200)
interfaces = interfaces[interfaces<=10]
interfaces = pd.DataFrame(interfaces, columns = ['dist_to_nearest_nonself'])
interfaces['celltype'] = adata.obs.loc[interfaces.index, 'leiden'].astype(str)

cores = nearest_dist_nonself.sort_values(ascending=False).groupby(adata.obs.leiden.astype(str)).head(200)
cores = cores[cores>=15]
cores = pd.DataFrame(cores, columns = ['dist_to_nearest_nonself'])
cores['celltype'] = adata.obs.loc[cores.index, 'leiden'].astype(str)

cores_2 = nearest_dist_nonself.sort_values().groupby(adata.obs.leiden.astype(str)).head(600)
cores_2 = cores_2[
    (cores_2<15) & (cores_2>10)]
cores_2 = pd.DataFrame(cores_2, columns = ['dist_to_nearest_nonself'])
cores_2['celltype'] = adata.obs.loc[cores_2.index, 'leiden'].astype(str)
cores_2 = cores_2[
    ~cores_2.index.isin(cores.index.tolist() + interfaces.index.tolist())]
#%%
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=15)
x_train = adata[cores.index].obsm['X_pca_harmony'][:,:25]
y_train = adata[cores.index].obs.leiden.astype(str)
neigh.fit(x_train, y_train.values)
# %%
core_2_test = adata[cores_2.index].obsm['X_pca_harmony'][:,:25]
interfaces_test = adata[interfaces.index].obsm['X_pca_harmony'][:,:25]
core_probs = neigh.predict_proba(x_train)
core2_probs = neigh.predict_proba(core_2_test)
interfaces_probs = neigh.predict_proba(interfaces_test)
# %%
entropies = pd.DataFrame(
    entropy(np.concatenate([core_probs, core2_probs, interfaces_probs]),axis=1),
    columns = ['entropy'])
entropies['type'] = ['core']*len(core_probs) + ['intermediate']*len(core2_probs) + ['interface']*len(interfaces_probs)
entropies.index = cores.index.tolist() + cores_2.index.tolist() + interfaces.index.tolist()
entropies['celltyep'] = adata.obs.loc[entropies.index, 'leiden'].astype(str)
sns.displot(data = entropies, x = 'entropy', kind="kde", hue='type',common_norm=False,)
# %%
sns.boxplot(
    data = entropies, y = 'entropy', x = 'celltyep', hue='type')
plt.xticks(rotation=90)
# %%
# simulate doublets
simulated_type = []
simulated_probs = []
for fraction_2nd in [0.1,0.2,0.4]:
    doublet = []
    core_pca_avg = pd.DataFrame(x_train).groupby(y_train.values).mean()
    for i in range(1500):
        core_idx = np.random.randint(0, len(x_train))
        core_ct = y_train.iloc[core_idx]
        core_pca = x_train[core_idx]
        mix_pca = core_pca_avg.drop(core_ct).sample().values
        simulated_cell = (1-fraction_2nd)*core_pca + fraction_2nd*mix_pca
        doublet.append(simulated_cell)
        simulated_type.append('Mixing fraction {}'.format(fraction_2nd))
    doublet = np.array(doublet).reshape(-1,25)
    probs = neigh.predict_proba(doublet)
    simulated_probs.append(probs)
simulated_probs = np.array(simulated_probs).reshape(4500,-1)
# %%
entropies = pd.DataFrame(
    entropy(np.concatenate([core_probs, core2_probs, interfaces_probs,simulated_probs]),axis=1),
    columns = ['entropy'])
entropies['type'] = ['core']*len(core_probs) + ['intermediate']*len(core2_probs) + ['interface']*len(interfaces_probs) + simulated_type
p = sns.kdeplot(
    data = entropies, x = 'entropy',  hue='type',common_norm=False,
    hue_order = ['core', 'intermediate', 'interface', 'Mixing fraction 0.1','Mixing fraction 0.2', 'Mixing fraction 0.4'])
lss = ['-', '-', '-', '-.','-.','-.']
handles = p.legend_.legendHandles
for line, ls, handle in zip(p.lines[::-1], lss, handles):
    line.set_linestyle(ls)
    handle.set_ls(ls)
plt.show()
# %%
'''
1. Cells that are closer to nonself cell types have higher cluster assignment entropy compared to 
cells that are in the core (distant to other cell types). Consistent with the rationale of 
the nature of segmentation error caused cell doublets.
2. simulated cells with 60%~90% purity shown a similar degree of entropy increase, with 80%
purity being similar to what is observed in data.
3. 




'''
#%%
def calculate_se(exp):
    E = np.log1p(exp).mean()
    S = np.log((exp + 1).mean())
    return S-E
exp = np.exp(adata.raw.to_adata().to_df())-1
noninfmphage = adata[adata.obs.leiden=='NonInfMphage'].obs_names
hsc = adata[adata.obs.leiden=='HSC_COL1_high'].obs_names
s_e = calculate_se(exp.loc[noninfmphage])
print(s_e.sort_values()[::-1].head(25))
s_e = calculate_se(exp.loc[hsc])
print(s_e.sort_values()[::-1].head(25))