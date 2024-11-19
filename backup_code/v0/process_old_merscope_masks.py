#%%
import pandas as pd
import numpy as np
import h5py
import os
import sys
#%%
# Cell makss
input_path = sys.argv[1]
boundries_fn = os.listdir(input_path + '/cell_boundaries')
for bfn in boundries_fn:
    cell_coords = pd.Series()
    # print(bfn)
    bfn = os.path.join(input_path, 'cell_boundaries', bfn)
    f = h5py.File(bfn,'r')
    for cell in list(f['featuredata']):
        coords = np.array((f['featuredata'][cell]['zIndex_0']['p_0']['coordinates'][0]))
        if coords.shape[0] >= 5:  
            cell_coords[cell] = coords
    f.close()
    cell_coords = cell_coords.to_frame(name='coord')
    cell_coords['X'] = cell_coords.coord.apply(lambda x: '_'.join(x[:,0].round(2).astype(str)))
    cell_coords['Y'] = cell_coords.coord.apply(lambda x: '_'.join(x[:,1].round(2).astype(str)))
    if os.path.exists(os.path.join(input_path, 'cell_coords.csv')):
        cell_coords.iloc[:,1:].to_csv(input_path + '/cell_coords.csv', mode='a', header=False)
    else:
         cell_coords.iloc[:,1:].to_csv(input_path + '/cell_coords.csv')
# %%
