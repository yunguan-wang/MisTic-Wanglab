import os 
import h5py
import pandas as pd
from geopandas import GeoDataFrame  
from shapely import Polygon
import numpy as np 
from itertools import combinations
from typing import Tuple 


def mask_eval(l: list) -> Tuple[bool, list]:
    """Given a list of coordinates, check if it contains only on polygon

    Parameters
    ----------
    l : list
        List of potentially vertices of different polygons

    Returns
    -------
    Tuple[bool, list]
        If the list defines only one polygon, and the coordinates of its vertices 
    """
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


def assemble_cell_coords(cls, 
                             input_path: str,
                             output_path: str) -> None:
        """Assemble the file containing the polygons of cell masks

        Parameters
        ----------
        input_path : str
            The input folder
        output_path : str
            The output folder
        """
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
                else: 
                    cell_coords[cell] = unique_v
            print(bfn)
            f.close()
            cell_coords = cell_coords.to_frame(name='coord')
            cell_coords['X'] = cell_coords.coord.apply(lambda x: '_'.join(x[0][:,0].round(2).astype(str)))
            cell_coords['Y'] = cell_coords.coord.apply(lambda x: '_'.join(x[0][:,1].round(2).astype(str)))
            if os.path.exists(os.path.join(input_path, 'cell_coords.csv')):
                cell_coords.iloc[:,1:].to_csv(input_path + '/cell_coords.csv', mode='a', header=False)
            else:
                    cell_coords.iloc[:,1:].to_csv(input_path + '/cell_coords.csv')
        
        cell_masks.index = ['cell_' + str(x+1) for x in cell_masks.index]
        cell_masks = cell_masks.loc[spacia_meta.index]
        cell_masks['n_polygon'] = cell_masks.X.apply(lambda x: len(x.split('_')))
        cell_masks.X = cell_masks.X.str.split('_')
        cell_masks.Y = cell_masks.Y.str.split('_')
        cell_masks['polygon'] = cell_masks.apply(
            lambda x: Polygon([(x.X[i], x.Y[i]) for i in range(x.n_polygon)]), axis=1)
        # Save as geopandas parquet file
        GeoDataFrame(
            cell_masks[['polygon']],geometry='polygon'
            ).to_parquet('cell_polygons.parquet', index=True)