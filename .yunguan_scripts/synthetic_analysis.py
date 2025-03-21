#%%
import os
from geopandas import read_parquet
import geopandas as gpd
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
import sys
sys.path.append('/users/wanzx2/codes/project/spatial_doubltet/MisC/MisC')
from misc import *
# %matplotlib inline
# %%
input_path = '/data/wanglab/project/doublet_detection/merscope_hcc1'
os.chdir(input_path)
np.random.seed(2)
counts = pd.read_csv('synthetic_counts_with_doublets.csv', index_col=0)
tx_meta = read_parquet('synthetic_counts_tx_metadata.parquet')
spacia_meta = pd.read_csv('synthetic_counts_with_doublets_metadata.csv', index_col=0)
spacia_meta = spacia_meta.loc[counts.index]
cell_masks = read_parquet('cell_polygons.parquet').loc[counts.index]
cell_masks.columns = ['Geometry']
# tx_meta['x'] = tx_meta.point.apply(lambda x: np.array(x.xy).flatten()[0])
# tx_meta['y'] = tx_meta.point.apply(lambda x: np.array(x.xy).flatten()[1])
# assign unique id to each tx
tx_meta.index = ['tx_'+str(i+1) for i in range(tx_meta.shape[0])]
# %%
model = misc()
model.import_data(
    counts.copy(), spacia_meta, cell_masks, tx_meta.copy(), x_col='X',y_col='Y',
    celltype_col='cell_type')
model.reassign_tx(1)
# %%
adata = model.adata
updated_counts = pd.DataFrame(
    adata.layers['counts_0_proposed_update'],
    adata.obs_names,
    adata.var_names)
updated_counts = updated_counts[updated_counts<0].fillna(0)
true_delta = tx_meta[tx_meta.tx_type!='real'].groupby(['cell_id','gene']).count().iloc[:,0]
true_delta = true_delta.reset_index().pivot(
    index = 'cell_id',
    columns = 'gene',
    values='point'
).fillna(0)
pred_delta = updated_counts.loc[true_delta.index, true_delta.columns]
true_delta = true_delta.stack()
pred_delta = pred_delta.stack()
mask = (true_delta>0) | (pred_delta!=0)
x_vec = true_delta[mask]
y_vec = -pred_delta[mask]
# %%
# sns.jointplot(
#     x=x_vec + np.random.uniform(0,0.1, len(x_vec)),
#     y=y_vec + np.random.uniform(0,0.1, len(x_vec)),
#     kind='kde')
sns.scatterplot(
    x=x_vec + np.random.uniform(0,0.1, len(x_vec)),
    y=y_vec + np.random.uniform(0,0.1, len(x_vec)),
)
plt.xlabel('Number of simulated Tx in cell')
plt.ylabel('Predicted Tx in cell')

# %%
updated_counts = updated_counts[updated_counts>0].fillna(0)
# %%
