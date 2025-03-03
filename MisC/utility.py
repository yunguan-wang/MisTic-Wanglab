# Data IO
import os 
import json 
import scanpy as sc
import geopandas as gpd
from geopandas import read_parquet
import pandas as pd
# Data manipulation 
import re 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely import distance
from scipy.spatial import KDTree
# Typing and other info
from typing import Optional, Union, Tuple
from time import perf_counter
from contextlib import contextmanager
import psutil


@contextmanager
def process_time_ram(message: str=""):
    print("="*30)
    print(message)
    t1 = t2 = perf_counter()
    yield lambda: t2-t1
    t2 = perf_counter()
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print("Done. Time taken is {0:.4f} seconds.".format(t2-t1))
    print("="*30)


def process_adata(adata: sc.AnnData,
                layer: str,
                dr_method: str) -> sc.AnnData:
    """Generate UMAP embedding for an AnnData 

    Parameters
    ----------
    adata : sc.AnnData
        An AnnData to be updated 
    layer : str
        The layer upon which UMAP is computed 

    Returns
    -------
    sc.AnnData
        AnnData with updated infomation 
    """
    # Make sure X is the counts 
    # Then for all subsequent processing, we can just proceed with the default 
    adata.X = adata.layers[layer].copy()
    if "log1p" in adata.uns:
        del adata.uns['log1p']
    sc.pp.normalize_total(adata, target_sum=1000)
    sc.pp.log1p(adata)
    # We also perform basic visualization 
    sc.pp.scale(adata)
    if dr_method == "umap":
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)    
        sc.tl.umap(adata)
    else:
        sc.pp.pca(adata, n_comps=2)
        sc.pp.neighbors(adata) 
    # Save the embedding to its own key
    # Note that X_umap always refers to the latest one 
    # Can use sc.pl.embedding(adata, basis="X_umap_???") to plot specific embedding 
    adata.obsm['X_'+dr_method+'_'+layer] = adata.obsm["X_"+dr_method].copy()
    return adata


def import_data(cell_metadata: Union[str, pd.DataFrame],
                cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                detected_transcripts: Union[str, pd.DataFrame, gpd.GeoDataFrame],
                cell_by_gene_counts: Optional[Union[str, pd.DataFrame]]=None,
                cell_centroid_x_col: str='center_x',
                cell_centroid_y_col: str='center_y',
                tx_x_col: str='global_x',
                tx_y_col: str='global_y',
                gene_col: str='gene',
                cell_col: str='cell_id',
                celltype_col: Optional[str]=None,
                leiden_res: float=1,
                dr_method: str="pca"
                ) -> Tuple[sc.AnnData, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Read in the four pieces information needed for subsequent analysis: 
    cell-by-gene counts, cell metadata, cell boundary information, and transcript information. Some 
    basic visualization and cell clustering will be performed. 

    Parameters
    ----------
    cell_by_gene_counts : Union[str, pd.DataFrame]
        Either the path to the csv file or a pandas dataframe containing the cell-by-gene count matrix whose first 
        column is assumed to contain the ID for each cell while the rest of the columns should be the transcript counts
        for each cell. 
    cell_metadata : Union[str, pd.DataFrame]
        Either the path to the csv file or a pandas dataframe containing the metadata for each cell whose 
        first column is assumed to contain the ID for each cell. The information in the file/object should 
        at least contain the xy coordinates of the centers of cells named center_x, and center_y, respectively.
        If the users have already performed cell typing which is not required and will only be used for visualization,
        the information can be stored here as a separate column: cell_type.
    cell_boundary_polygons : Union[str, gpd.GeoDataFrame]
        Either the path to the parquet file or the geopandas GeoDataFrame containing the vertex information 
        for each cell. The first column is assumed to be the IDs for cells. It should contain one column 'Geometry'
        that records the coordinates of the vertices.
    detected_transcripts : Union[str, pd.DataFrame, gpd.GeoDataFrame]
        Either the path to the csv file or a pandas dataframe containing the information of the detected transcripts.
        The first column is assumed to be some index. The information should contain the ID for a transcripte/molecule, 
        the ID of the cell it belongs to, its xy coordinate named global_x, global_y respectively, and its gene information. 

    Returns
    -------
    Tuple[sc.AnnData, gpd.GeoDataFrame, gpd.GeoDataFrame]
        An AnnData object containing the original counts and some metainformation. The detected transcript as well as the polygons. 
    """

    # Cell metadata
    with process_time_ram("Processing cell metadata") as ctm:
        if isinstance(cell_metadata, str) and ("csv" in os.path.splitext(cell_metadata)[1]):
            cell_meta = pd.read_csv(cell_metadata, index_col=0)
        elif isinstance(cell_metadata, pd.DataFrame):
            cell_meta = cell_metadata
        else: 
            raise TypeError("Only .csv file or pandas dataframe is allowed")
        cell_meta.index.rename(name='cell_id', inplace=True)
        cell_meta.rename(columns={cell_centroid_x_col: "center_x",
                                cell_centroid_y_col: "center_y"}, inplace=True, errors='raise')

    # Polygons of cells 
    with process_time_ram("Processing cell boundaries") as ctm:
        if isinstance(cell_boundary_polygons, str) and ("parquet" in os.path.splitext(cell_boundary_polygons)[1]):
            cell_coords = read_parquet(cell_boundary_polygons)
        elif isinstance(cell_boundary_polygons, gpd.GeoDataFrame):
            cell_coords = cell_boundary_polygons
        else: 
            raise TypeError("Only .parquet file or geopandas dataframe is allowed")
        cell_coords.index.rename(name='cell_id', inplace=True)
        if cell_coords.geometry.name != "cell_boundary_geom":
            cell_coords.rename_geometry("cell_boundary_geom", inplace=True)
        # Remove potential duplicated vertices in a polygon
        cell_coords['cell_boundary_geom'] = cell_coords['cell_boundary_geom'].remove_repeated_points(tolerance=0.0)
        # Sanity check 
        assert set(cell_coords.index) == set(cell_meta.index), "Cell populations in cell boundaries and cell metadata are different."
    
    # Transcript information 
    with process_time_ram("Processing Transcript information ") as ctm:
        # We also convert the pandas dataframe to geopandas geodataframe 
        # by constructing points from the locations of each transcript
        if isinstance(detected_transcripts, str) and ("parquet" in os.path.splitext(detected_transcripts)[1]):
            tx_metadata = read_parquet(detected_transcripts)
        elif isinstance(detected_transcripts, str) and ("csv" in os.path.splitext(detected_transcripts)[1]):
            tx_metadata = pd.read_csv(detected_transcripts, index_col=0)    
        elif isinstance(detected_transcripts, pd.DataFrame) or isinstance(detected_transcripts, gpd.GeoDataFrame):
            tx_metadata = detected_transcripts
        else: 
            raise TypeError("Only .parquet/.csv file or geopandas/pandas dataframe is allowed")

        if not isinstance(tx_metadata, gpd.GeoDataFrame):
            tx_metadata = gpd.GeoDataFrame(tx_metadata, 
                                        geometry=gpd.points_from_xy(tx_metadata[tx_x_col], tx_metadata[tx_y_col]))
        
        tx_metadata.reset_index(drop=True, inplace=True)
        tx_metadata.index = "tx_" + tx_metadata.index.astype(str)
        tx_metadata['molecule_id'] = tx_metadata.index
        if tx_metadata.geometry.name != "tx_geom":
            tx_metadata.rename_geometry("tx_geom", inplace=True)
        tx_metadata.rename(
            columns={
                gene_col: 'gene',
                cell_col: 'cell_id'
                }, inplace=True, errors='raise')
        # Sanity check 
        assert set(tx_metadata['cell_id']) == set(cell_meta.index), "Cell populations in transcript metadata and cell metadata are different."
        
    # Cell by gene counts matrix 
    with process_time_ram("Processing cell by gene matrix") as ctm:
        if cell_by_gene_counts is not None:
            if isinstance(cell_by_gene_counts, str) and ("csv" in os.path.splitext(cell_by_gene_counts)[1]):
                counts = pd.read_csv(cell_by_gene_counts, index_col=0)
            elif isinstance(cell_by_gene_counts, pd.DataFrame):
                counts = cell_by_gene_counts
            else: 
                raise TypeError("Only .csv file or pandas dataframe is allowed")
        else: 
            counts = tx_metadata.groupby(['cell_id', "gene"],
                                        as_index=False)['molecule_id'].count()
            counts = pd.pivot(counts, values='molecule_id', columns="gene", index='cell_id').fillna(0)
            
        counts.index.rename(name='cell_id', inplace=True)
        # Sanity check 
        assert set(counts.index) == set(cell_meta.index), "Cell populations in cell-by-gene counts and cell metadata are different."
        assert set(counts.columns) == set(tx_metadata['gene']), "Genes in cell-by-gene counts and transcript metadata are different."
        

    # Create AnnData object to store the counts 
    # and facilitate future processing 
    with process_time_ram("Creating AnnData object") as ctm:
        adata = sc.AnnData(counts)
        adata.obs['x'] = cell_meta.loc[adata.obs_names, "center_x"]
        adata.obs['y'] = cell_meta.loc[adata.obs_names, "center_y"]
        adata.obs = gpd.GeoDataFrame(adata.obs,
                                    geometry=gpd.points_from_xy(adata.obs['x'], adata.obs['y']))
        adata.obs.rename_geometry("cell_centroid_geom", inplace=True)
        adata.var['col_index'] = [i for i in range(adata.var.shape[0])]
        adata.var['gene'] = adata.var_names
        adata.var.set_index("gene", inplace=True)
        # As we will alter the counts later on, we will save a copy of the 
        # original count in one of the layers 
        adata.layers['counts_0'] = adata.X.copy()
        adata.raw = adata.copy()
    
    # Then we perform basic normalization and transformation 
    with process_time_ram("Successfully read in data. Performing basic transformation") as ctm:
        adata = process_adata(adata=adata,
                            layer="counts_0",
                            dr_method=dr_method)
    # And we will use leiden to perform cell clustering
    if celltype_col is not None:
        with process_time_ram("Assigning pre-existing cell typing info from metadata.") as ctm:
            adata.obs['cell_type'] = cell_meta.loc[adata.obs_names, celltype_col]
            adata.obs['leiden'] = pd.factorize(cell_meta.loc[adata.obs_names, celltype_col])[0].astype(str)
    else:
        with process_time_ram("Performing Leiden clustering.") as ctm:
            sc.tl.leiden(adata, resolution=leiden_res, key_added='cell_type')
            adata.obs['leiden'] = adata.obs['cell_type'].astype(str)
    
    with process_time_ram("Adding meta info to adata") as ctm:
        # leiden always stores the latest version 
        adata.obs['counts_0_leiden'] = adata.obs['leiden'].copy()
        # Record some meta information 
        adata.uns['unique_leiden'] = np.unique(adata.obs['counts_0_leiden'])
        adata.uns['n_leiden'] = len(adata.uns['unique_leiden'])
        adata.uns['cell_type_leiden_map'] = adata.obs[["cell_type", "leiden"]].drop_duplicates(ignore_index=True)
        adata.uns['cell_type_leiden_map'].rename(columns={"cell_type": "cell_type_name",
                                                        "leiden": "cell_type_index"},
                                                inplace=True)
        adata.uns['dr_method'] = dr_method
        adata.uns['centroid_x_min'] = adata.obs['x'].min()
        adata.uns['centroid_x_max'] = adata.obs['x'].max()
        adata.uns['centroid_y_min'] = adata.obs['y'].min()
        adata.uns['centroid_y_max'] = adata.obs['y'].max()
        adata.uns['n_genes'] = adata.var.shape[0]
        adata.uns['current_layer'] = "counts_0"
    
    return adata, cell_coords, tx_metadata
    


def calculate_mask_distance(adata: sc.AnnData,
                            cell_coords: gpd.GeoDataFrame,
                            max_centroid_dist: float=50) -> gpd.GeoDataFrame:
    """Calculate cell-cell distance based on their cell masks. 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell metainformation
    cell_coords : gpd.GeoDataFrame
        The geodataframe recording the vertices of all the cells 
    max_centroid_dist : float, optional
        The threshold on cell-cell centroid distances beyond which we do not consider two cells being neighbors, by default 15

    Returns
    -------
    gpd.GeoDataFrame
        A dataframe where each row is a pair of neighboring cells as well as their distance information 
    """
    
    with process_time_ram("Compute cell-cell distance matrix based on their recorded centroids") as ctm:
        # First, we compute cell-cell distance matrix based on their recorded centroids 
        # As computing mask distances between all pairs of cells would take too long
        # we use the centroid distance to filter out the majority of the cells 
        # However, direct computation would consume a huge chunk of memory. Therefore, we use KDTree to 
        # find nearest neighbors. Currently, the number is set to 10 with an upper bound 
        # 
        centroid_dist_tree = KDTree(adata.obs[['x','y']])
        # The distance is sorted 
        _, adj_ind = centroid_dist_tree.query(adata.obs[['x','y']], k=25, 
                                            distance_upper_bound=max_centroid_dist, workers=-1)
        # This will directly give us cells and their at most 10 NNs
        # However, the adj_ind is the numeric index (row number)
        adj = pd.DataFrame(adj_ind, index=adata.obs_names).melt(ignore_index=False).drop(columns=['variable'])
        # Filter out values due to upper bound 
        adj = adj[adj['value']<adata.obs.shape[0]]
        # Replace row number by actual cell id 
        adj.loc[:,'n'] = adata.obs_names[adj['value']]
        adj_masks_ids = adj[adj.index != adj.loc[:, "n"]].drop(columns=['value'])
    # Then we compute the cell-cell distance based on their masks
    with process_time_ram("Compute cell-cell distance matrix based on their masks") as ctm:
        md = distance(
            cell_coords.loc[adj_masks_ids.index, "cell_boundary_geom"].values,
            cell_coords.loc[adj_masks_ids['n'].values, "cell_boundary_geom"].values
        )
        mask_distance = gpd.GeoDataFrame(adj_masks_ids).rename(columns={"n": "neighbor_cell_id"})
        mask_distance['mask_distance'] = md
        mask_distance.reset_index(inplace=True, drop=False)
        mask_distance = mask_distance.merge(adata.obs[['x', 'y', 'cell_centroid_geom']], how='left', 
                                            left_on='cell_id', right_index=True).set_geometry("cell_centroid_geom")
    return mask_distance


def even_split(array: np.array, 
               chunk_size: int) -> list:
    """Split an array into chunks of specified size 

    Parameters
    ----------
    array : np.array
        An array to be split
    chunk_size : int
        The target chunk size 

    Returns
    -------
    list
        A list of chunks of the original array 
    """
    return np.array_split(array, np.ceil(array.shape[0] / chunk_size), axis=0)



def extract_layer_num(layer: str) -> int:
    """Extracts the layer number 
    It extracts all the number in the layer string but only returns the first one

    Parameters
    ----------
    layer : str
        Name of the layer. It should be like counts_0, counts_1_proposed_update, etc

    Returns
    -------
    int
        The layer number
    """
    return int(re.findall(r'\d+', layer)[0])
    

def generate_count_patches(adata: sc.AnnData,
                           tx_to_reassign: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Based on the transcript reassignment, generate patches to update the current count matrix 

    Parameters
    ----------
    adata : sc.AnnData
        An AnnData to be updated 
    tx_to_reassign : pd.DataFrame
        Transcripts that should be reassigned

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The first dataframe contains counts that should be subtracted from the current count
        The second dataframe contains counts that should be added to the current count
    """
    # Generate two patches for the count matrix 
    counts_to_subtract = pd.DataFrame(0, index=adata.obs_names, columns=adata.var_names)
    counts_to_add = pd.DataFrame(0, index=adata.obs_names, columns=adata.var_names)
    # For removal, we for each cell count how many genes occured 
    cell_to_remove = tx_to_reassign.groupby(by=['cell_id', "gene"], as_index=False).size()
    # For addition, we for each cell in the neighbor count how many genes occured 
    cell_to_add = tx_to_reassign.groupby(by=['neighbor_cell_id', "gene"], as_index=False).size()
    cell_to_add.rename(columns={"neighbor_cell_id": "cell_id"}, inplace=True)
    # Transform the dataframe from long to wide 
    subtract_patch = pd.pivot(cell_to_remove, values="size", columns="gene", index='cell_id').fillna(0)
    add_patch = pd.pivot(cell_to_add, values="size", columns="gene", index='cell_id').fillna(0)
    # The update the find matching rows and columns 
    counts_to_subtract.update(subtract_patch)
    counts_to_add.update(add_patch)
    
    return counts_to_subtract, counts_to_add


def make_reassignment_adata(adata: sc.AnnData,
                            layer: str,
                            tx_to_reassign: pd.DataFrame,
                            trial_layer: Optional[str]=None,
                            dr_method: str='pca') -> sc.AnnData:
    """Make the count adjustment on the adata alone 
    This will not alter the tx information 

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing cell metainformation
    layer : str
        The layer upon which the update is computed 
    tx_to_reassign : pd.DataFrame
        Transcripts that should be reassigned
    trial_layer : Optional[str] 
        If provided, the updated counts will be stored in this layer 
    Returns
    -------
    sc.AnnData
        The adjusted adata
    """
    with process_time_ram("Updating gene counts") as ctm: 
        counts_to_subtract, counts_to_add = generate_count_patches(adata=adata,
                                                                tx_to_reassign=tx_to_reassign)
        layer_num = extract_layer_num(layer)
        if trial_layer is None:
            trial_layer = "counts_"+str(int(layer_num+1))
            
        # Update adata
        adata.layers[trial_layer] = adata.layers[layer]+counts_to_add-counts_to_subtract 
        if np.any(adata.layers[trial_layer]<0):
            raise Exception("Negative values generated. This might be due inconsistency between the count matrix and the tx data.")
        adata = process_adata(adata=adata, layer=trial_layer, dr_method=dr_method)
    return adata


def sample_logistic(shape: tuple, 
                    model_device: torch.device,
                    eps: float=1e-20):
    """Sample gumbel random variables

    Parameters
    ----------
    shape : tuple
        Shape of the final result 
    model_device : torch.device
        Specify where the tensor will be stored 
    eps : float, optional
        Used to avoid -inf from log, by default 1e-20

    Returns
    -------
    torch.tensor
        Gumbel RVs
    """
    U = torch.rand(shape).to(model_device)
    return torch.log(U/(1-U + eps) + eps)


def binary_gumbel_softmax_sample(logits: torch.tensor,
                                 temperature: float,
                                 model_device: torch.device,
                                 hard: bool=True) -> torch.tensor:
    """Gumbel softmax trick for binary variables 

    Parameters
    ----------
    logits : torch.tensor
        Unnormalized class probabilities 
    temperature : float
        Temperature
    model_device : torch.device
        Specify where the tensor will be stored 

    Returns
    -------
    torch.tensor
        Gumbel softmax "bernoulli" variables 
    """
    y = logits + sample_logistic(logits.size(), model_device=model_device)
    y_soft = torch.sigmoid(y/temperature)
    if hard:
        y_hard = torch.round(y_soft, decimals=0)
        y_hard = (y_hard - y_soft).detach() + y_soft
        return y_hard 
    else: 
        return y_soft


class diagLinear(nn.Module):
    def __init__(self, 
                 features,
                 bias):
        super().__init__()
        stdv = 1. / np.sqrt(features)
        self.weight = nn.Parameter(torch.FloatTensor(1, features).uniform_(-stdv, stdv))
        self.bias = torch.zeros_like(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, features).uniform_(-stdv, stdv))
            
    def forward(self, X):
        return X * self.weight + self.bias

class Positive(nn.Module):
    """This class is used to reparametrize model weights 

    """
    def forward(self, X):
        return F.softplus(X)
    

class JSONEncoder(json.JSONEncoder):
    """This class is used to save dictionary of pd.DataFrame to a json file

    """
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        return json.JSONEncoder.default(self, obj)


