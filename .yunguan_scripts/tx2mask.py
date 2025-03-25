#%%
import os
from geopandas import read_parquet
import numpy as np
import pandas as pd
import numpy as np
from shapely import Point, distance
# %%
input_path = '/data/wanglab/project/doublet_detection/merscope_hcc1'
os.chdir(input_path)
np.random.seed(2)
# All data is downloaded from VizGen, except for the spacia_meta, which is generated based on 
# in house cell typing.
spacia_meta = pd.read_csv('hcc1_spacia_meta.txt', sep='\t', index_col=0)
cell_masks = read_parquet('cell_polygons.parquet')
tx_meta = pd.read_csv('detected_transcripts.csv', index_col=0)
tx_meta = tx_meta[~tx_meta.gene.str.contains('Blank')]
cell_meta = pd.read_csv('cell_metadata.csv', index_col=0)
cell_meta.index = ['cell_' + str(x+1) for x in cell_meta.index]
spacia_meta['polygon'] = cell_masks.loc[spacia_meta.index, 'polygon'].values
spacia_meta['fov'] = cell_meta.loc[spacia_meta.index, 'fov'].values
#%%
t = 0
for fov in spacia_meta.fov.unique():
    fov_tx = tx_meta[tx_meta.fov==fov].copy()
    fov_tx.index = ['tx_' + str(t+i) for i in range(fov_tx.shape[0])]
    fov_cell = spacia_meta[spacia_meta.fov==fov]
    fov_tx.loc[:,'point'] = fov_tx.apply(lambda x: Point(x.global_x, x.global_y), axis=1)
    points = np.repeat(fov_tx.point.values, fov_cell.shape[0])
    masks = fov_cell.polygon.tolist() * fov_tx.shape[0]
    tx_mask_dist = distance(points, masks)
    crit = tx_mask_dist == 0
    tx_kept = np.repeat(fov_tx.index, fov_cell.shape[0])[crit]
    masks_kept = np.array(fov_cell.index.tolist() * fov_tx.shape[0])[crit]
    fov_tx.loc[tx_kept,'cell'] = masks_kept
    fov_tx = fov_tx.dropna()
    if t == 0:
        fov_tx.to_csv('transcript_meta.csv')
    else:
        fov_tx.to_csv('transcript_meta.csv', mode='a', header=None)
    with open('tx_annotation_progress', 'a') as f:
        f.write('{}\n'.format(fov))
    t += fov_tx.shape[0]
