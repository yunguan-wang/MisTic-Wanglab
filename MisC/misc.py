# Data IO
import h5py
import os
import pandas as pd
import scanpy as sc
import geopandas as gpd
# Data manipulation 
import numpy as np 
# Utility function 
from utility import import_data, calculate_mask_distance, extract_layer_num
from transcript_reassign import propose_reassignment, test_proposed_reassignment, make_reassignment
# Typing 
from typing import Tuple, Union, Optional 



class misc:
    def __init__(self) -> None:
        self.adata = None
        self.cell_coords = None
        self.tx_metadata = None
        self.current_layer = "counts_0"
        self.mask_distance = None

    def import_data(self,
                    cell_by_gene_counts: Union[str, pd.DataFrame],
                    cell_metadata: Union[str, pd.DataFrame],
                    cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                    detected_transcripts: Union[str, pd.DataFrame],
                    **kwargs) -> None:
        self.adata, self.cell_coords, self.tx_metadata = import_data(cell_by_gene_counts=cell_by_gene_counts,
                                                                      cell_metadata=cell_metadata,
                                                                      cell_boundary_polygons=cell_boundary_polygons,
                                                                      detected_transcripts=detected_transcripts,
                                                                      **kwargs)
        self.mask_distance = calculate_mask_distance(adata=self.adata,
                                                      cell_coords=self.cell_coords)
    
    def _reassign_tx(self) -> None:
        counts_to_subtract, counts_to_add, tx_assignment_addition, tx_assignment_removal = propose_reassignment(
            adata=self.adata, 
            tx_metadata=self.tx_metadata,
            cell_coords=self.cell_coords,
            layer=self.current_layer,
            mask_distance=self.mask_distance)
        test_result = test_proposed_reassignment(
            adata=self.adata,
            layer=self.current_layer,
            counts_to_subtract=counts_to_subtract,
            counts_to_add=counts_to_add)
        self.adata, self.tx_metadata = make_reassignment(
            adata=self.adata,
            layer=self.current_layer,
            tx_metadata=self.tx_metadata,
            tx_assignment_addition=tx_assignment_addition,
            tx_assignment_removal=tx_assignment_removal,
            test_result=test_result)
    
    def reassign_tx(self, n_iter: int) -> None:
        for _ in range(n_iter):
            self._reassign_tx()
            layer_num = extract_layer_num(self.current_layer)
            self.current_layer = "counts_"+str(int(layer_num+1))
            
    @classmethod 
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
            if os.path.exists(os.path.join(output_path, 'cell_coords.csv')):
                cell_coords.iloc[:,1:].to_csv(output_path + '/cell_coords.csv', mode='a', header=False)
            else:
                cell_coords.iloc[:,1:].to_csv(output_path + '/cell_coords.csv')  
        # csv to parquet code here 
        
    @classmethod 
    def foo(cls):
        pass 