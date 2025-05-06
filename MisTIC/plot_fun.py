# Data IO 
import numpy as np 
import pandas as pd 
import geopandas as gpd 
import polars as pl 
from sklearn import preprocessing
# Plotting functions 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
# Typing 
from MisTIC.mistic_class import mistic 
from typing import Optional 


def inspect_cell(cell_id: str,
                 m: mistic,
                 gene: Optional[str]=None,
                 neighbor_index: Optional[int]=None, 
                 inspect: str="reassign_probs",
                 reassign_probs_threshold: float=0,
                 fig_size: tuple=(15,15)) -> None:
    """Visual inpsect different aspects of a cell 

    Parameters
    ----------
    cell_id : str
        Cell id 
    m : mistic
        The mistic object 
    gene : Optional[str], optional
        Name of a gene. If None, then all genes will be shown, by default None
    neighbor_index : Optional[int], optional
        Neighbor index to show. If None, then must have tx_reassign_info, by default None
    inspect : str, optional
        What to inspect, by default "reassign_probs"
    reassign_probs_threshold : float, optional
        Threshold on reassign_probs, by default 0
    fig_size : tuple, optional
        Figure size, by default (15,15)
    """
    assert inspect in ["distance_feature",
                    "prior_distance_feature",
                    "exp_feature",
                    "prior_exp_feature",
                    "neighbor_exp_feature",
                    "prior_neighbor_exp_feature",
                    "reassign_probs"], "inspect has to be one of distance_feature, prior_distance_feature, exp_feature, prior_exp_feature, neighbor_exp_feature, prior_neighbor_exp_feature, reassign_probs"
    # Find all neighbor cells 
    all_to_cell_id = m.intf_tx.filter(pl.col('cell_id') == cell_id)['neighbor_cell_id'].unique().to_numpy()
    all_cell_id = np.append(all_to_cell_id, cell_id)
    # Extract transcripts 
    if neighbor_index is None:
        tx_to_inspect = m.tx_reassign_info.clone()
    else: 
        tx_to_inspect = m.intf_tx.filter(pl.col('neighbor_index')==neighbor_index)
        
    tx_to_inspect = tx_to_inspect.filter(pl.col("cell_id") == cell_id)
    if gene is not None:
        tx_to_inspect = tx_to_inspect.filter(pl.col('gene') == gene)
    # Add location info
    temp = pl.from_pandas(m.tx_metadata.loc[m.tx_metadata['cell_id']==cell_id, ['molecule_id', 'global_x', "global_y"]])
    tx_to_inspect = tx_to_inspect.join(temp, how='left', on='molecule_id')
    # Color different cells 
    cmap = mpl.colormaps['inferno']
    colors = cmap(np.linspace(0, 1, all_cell_id.shape[0]))
    # Plot 
    m.cell_coords.loc[all_cell_id, "cell_boundary_geom"].plot(color=colors,
                                                                figsize=fig_size)
    # Size the dots according to inspect 
    tx_to_inspect = tx_to_inspect.with_columns(pl.Series(name='size',
                                                            values=preprocessing.minmax_scale(tx_to_inspect[inspect])))
    plt.scatter(tx_to_inspect['global_x'], tx_to_inspect['global_y'],
        s=10*tx_to_inspect['size']**2+0.1, c="grey", alpha=0.5)
    for counter, to_id in enumerate(all_to_cell_id):    
        subset = tx_to_inspect.filter(pl.col("neighbor_cell_id")==to_id)
        if inspect == 'reassign_probs':
            subset = subset.filter(pl.col("reassign_probs")>=reassign_probs_threshold)
        plt.scatter(subset['global_x'],
                    subset['global_y'],
                    s=10*subset['size']**2+0.1, color=colors[counter,:])
    
    plt.show()

