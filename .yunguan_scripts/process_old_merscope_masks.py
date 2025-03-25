#%%
import pandas as pd
import numpy as np
import h5py
import os
from itertools import combinations
import sys
#%%
def mask_eval(l):
    '''
    l : list of mask coords
    '''
    unique_l = []
    for a,b in combinations(l,2):
        if (len(unique_l) ==0) or (a not in np.array(unique_l)):
            unique_l.append(a)
        if np.array_equal(a, b):
            continue
        else:
            if b not in np.array(unique_l):
                unique_l.append(b)
    return len(unique_l) > 1, unique_l
#%%
# Cell makss
# input_path = sys.argv[1]
input_path = '/data/wanglab/project/doublet_detection/merscope_hcc1'
boundries_fn = os.listdir(input_path + '/cell_boundaries')
for bfn in boundries_fn:
    cell_coords = pd.Series()
    # print(bfn)
    bfn = os.path.join(input_path, 'cell_boundaries', bfn)
    f = h5py.File(bfn,'r')
    diff_coords = []
    for cell in list(f['featuredata']):
        coords = []
        for i in range(7):
            tmp = np.array((f['featuredata'][cell]['zIndex_'+str(i)]['p_0']['coordinates'][0]))
            coords.append(tmp)
        non_unique, unique_v = mask_eval(coords)
        if non_unique:
            print(cell)
    print(bfn)
    f.close()
    # cell_coords = cell_coords.to_frame(name='coord')
    # cell_coords['X'] = cell_coords.coord.apply(lambda x: '_'.join(x[:,0].round(2).astype(str)))
    # cell_coords['Y'] = cell_coords.coord.apply(lambda x: '_'.join(x[:,1].round(2).astype(str)))
    # if os.path.exists(os.path.join(input_path, 'cell_coords.csv')):
    #     cell_coords.iloc[:,1:].to_csv(input_path + '/cell_coords.csv', mode='a', header=False)
    # else:
    #      cell_coords.iloc[:,1:].to_csv(input_path + '/cell_coords.csv')

#%%
