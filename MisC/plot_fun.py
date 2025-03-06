import matplotlib as mpl 
import matplotlib.pyplot as plt 
from sklearn import preprocessing
import numpy as np 
import pandas as pd 
import geopandas as gpd 
import polars as pl 


def inspect_cell(cell_id,
                 m):
    all_to_cell_id = m.intf_tx.filter(pl.col('cell_id') == cell_id)['neighbor_cell_id'].unique().to_numpy()
    all_cell_id = np.append(all_to_cell_id, cell_id)
    intf_tx = m.intf_tx.group_by("molecule_id").agg(pl.all().sort_by("reassign_probs", descending=False).last())
    
    cmap = mpl.colormaps['inferno']
    colors = cmap(np.linspace(0, 1, all_cell_id.shape[0]))
    


    m.cell_coords.loc[all_cell_id, "cell_boundary_geom"].plot(color=colors,
                                                            figsize=(20,20))
    plt.show()

