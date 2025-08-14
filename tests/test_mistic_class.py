import pytest
import os 
import numpy as np 
import pandas as pd 
import geopandas as gpd
from geopandas import read_parquet

from MisTIC.mistic_class import mistic

datafolder = "./tests/test_data"
tx_meta = pd.read_csv(os.path.join(datafolder, "detected_transcripts.csv"), index_col=0)
cell_by_gene = pd.read_csv(os.path.join(datafolder, "cell_by_gene_counts.csv"), index_col=0)
cell_metadata = pd.read_csv(os.path.join(datafolder, "cell_metadata.csv"), index_col=0)
cell_coords = read_parquet(os.path.join(datafolder, "cell_boundary_polygons.parquet"))

@pytest.mark.parametrize("reassign_threshold_grid, remove_threshold_grid, temperature, top_k", [
    (0.5, 0, 0.0, None),
    ([0.4, 0.5], 0, 0.1, 3)
])
def test_mistic(reassign_threshold_grid, 
                remove_threshold_grid,
                temperature,
                top_k):
    m = mistic(cell_centroid_x_col="center_x",
                cell_centroid_y_col="center_y",
                cell_col="cell_id",
                celltype_col = "cell_type",
                tx_x_col='global_x',
                tx_y_col='global_y')
    m.import_data(cell_metadata=cell_metadata,
              cell_boundary_polygons = cell_coords,
              detected_transcripts = tx_meta,
              cell_by_gene_counts = cell_by_gene)
    m.patchfy_data()
    m.initialize_parameters()
    m.training_loop(n_epochs=3)
    m.compute_reassign_probs()
    m.correct_tx(reassign_threshold_grid=reassign_threshold_grid,
                 remove_threshold_grid=remove_threshold_grid)
    m.recluster(temperature=temperature,
                top_k=top_k,
                overwrite_previous_trials=True)
    
    assert ("counts_0_corrected_cell_type_0" in m.adata.obs.columns)
    
    
