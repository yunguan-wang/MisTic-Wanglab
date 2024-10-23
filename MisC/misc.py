# Data IO
import pandas as pd
import scanpy as sc
import geopandas as gpd

# Utility function 
from utility import import_data, calculate_mask_distance
from transcript_reassign import propose_reassignment, test_proposed_reassignment, make_reassignment
# Typing 
from typing import Tuple, Union, Optional 



class misc:
    def __init__(self):
        self.adata = None
        self.cell_coords = None
        self.tx_metadata = None
        self.current_layer = "counts_0"
        self.mask_distance = None

    def import_data(self,
                    cell_by_gene_counts: Union[str, pd.DataFrame],
                    cell_metadata: Union[str, pd.DataFrame],
                    cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                    detected_transcripts: Union[str, pd.DataFrame]) -> None:
        self.adata, self.cell_coords, self.tx_metadata =  import_data(cell_by_gene_counts=cell_by_gene_counts,
                                                                      cell_metadata=cell_metadata,
                                                                      cell_boundary_polygons=cell_boundary_polygons,
                                                                      detected_transcripts=detected_transcripts)
        self.mask_distance  = calculate_mask_distance(adata=self.adata,
                                                      cell_coords=self.cell_coords)
    
    def _reassign_tx(self):
        counts_to_subtract, counts_to_add, tx_assignment_addition, tx_assignment_removal = propose_reassignment(adata=self.adata, 
                                                                                                                tx_metadata=self.tx_metadata,
                                                                                                                cell_coords=self.cell_coords,
                                                                                                                layer=self.current_layer,
                                                                                                                mask_distance=self.mask_distance)
        test_result = test_proposed_reassignment(adata=self.adata,
                                                 layer=self.current_layer,
                                                 counts_to_subtract=counts_to_subtract,
                                                 counts_to_add=counts_to_add)
        self.adata, self.tx_metadata = make_reassignment(adata=self.adata,
                                                         layer=self.current_layer,
                                                         tx_metadata=self.tx_metadata,
                                                         tx_assignment_addition=tx_assignment_addition,
                                                         tx_assignment_removal=tx_assignment_removal)
    
    def reassign_tx(self, n_iter: int):
        for n in range(n_iter):
            self._reassign_tx()
            
    