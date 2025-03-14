"""The Command-Line Interface (CLI) of MisC

The CLI of MisC can be accessed via ``python -m MisC``.

:Example:

    Get help:
    
    .. code-block:: bash

        python -m MisC -h
    
    Check version and authors:
    
    .. code-block:: bash
    
        python -m MisC --version 
        python -m MisC --author

"""


import os
import sys
import torch
import argparse

from MisC.__version__ import __version__, __author__
from MisC.misc_class import misc

parser = argparse.ArgumentParser(description="Misc")

parser.add_argument("-v", "--version", action="version",
                    version=__version__, help="Display the version of the software")
parser.add_argument("--author", action="version", version=__author__,
                    help="Check the author list of the algorithm")

# Instantiate object 
parser.add_argument("--cell_centroid_x_y_col", nargs=2, type=str, default=['center_x', 'center_y'], 
                    help="The column containing the x and y coordinates of cell centroids in cell metadata file")
parser.add_argument("--celltype_col", nargs='?', type=str,
                    help="The column containing the cell type information in cell metadata file")
parser.add_argument("--tx_x_y_col", nargs=2, type=str, default=["global_x", "global_y"], 
                    help="The column containing the x and y coordinates of transcript in detected transcript file")
parser.add_argument("--gene_col", type=str, default="gene",
                    help="The column containing the gene of transcript in detected transcript file")
parser.add_argument("--cell_col", type=str, default="cell_id",
                    help="The column containing the cell id of transcript in detected transcript file")
parser.add_argument("--leiden_res", type=float, default=1,
                    help="The resolution for leiden clustering")
parser.add_argument("--dr_method", type=str, default="umap",
                    help="The dimension reduction method to be used. Either pca or umap")
parser.add_argument("--max_centroid_dist", type=float, default=50,
                    help="When computing mask distances, if a centroid distance is greater than this value, the mask distance will not be computed.")
parser.add_argument("--mask_dist_cutoff", type=float, default=5,
                    help="The threshold of cell-cell mask distances beyond which a cell is no longer considered a neighbor")
parser.add_argument("--nearest", type=int, default=3,
                    help="Maximum number of nearest neighbors to perform reassignment")
parser.add_argument("--prior_50_reassign_prob", type=float, default=0.01,
                    help="The prior probability of reassigning a transcript that's ranked 50%")
parser.add_argument("--prior_5_reassign_prob", type=float, default=0.99,
                    help="The prior probability of reassigning a transcript that's ranked 5%")

# Import data 
parser.add_argument("--cell_metadata", type=str, required=True,
                    help="The path to the csv file that contains the coordinates of the centroids of cells")
parser.add_argument("--cell_boundary_polygons", type=str, required=True,
                    help="The path to the parquet file that contains the coordinates of the vertices of polygons of cells")
parser.add_argument("--detected_transcripts", type=str, required=True,
                    help="The path to the csv/parquet file that contains the coordinates, genes, and cell assignment of detected transcripts")
parser.add_argument("--cell_by_gene_counts", type=str,
                    help="The path to the csv file that contains the cell by gene count. If not provided, this will be inferred from the detected transcript file.")

# Model training 
parser.add_argument("--percent_cell_per_patch", type=float, default=0.1,
                    help="The rough percentage of cells within each minibatch")
parser.add_argument("--num_overlap", type=int, default=7,
                    help='Number of overlapping patches')
parser.add_argument("--n_epochs", type=int, default=20,
                    help="Number of epochs")

# reassign tx 
parser.add_argument("--criterion", type=str, default="0.5",
                    help="Threshold on reassigning the transcript. Should either be 'auto' or a number in [0,1]")

# Model saving 
parser.add_argument("--dir_name", type=str, default=".",
                    help="The directory path at which the saved model should be.")
parser.add_argument("--model_name", type=str, default="misc",
                    help="The model name")

# Maybe consider
# config.yaml


def main(cmdargs: argparse.Namespace):
    """The main method for MisC

    Parameters:
    ----------
    cmdargs: argparse.Namespace
        The command line argments and flags 
    """
    cell_centroid_x_col, cell_centroid_y_col = cmdargs.cell_centroid_x_y_col
    celltype_col = cmdargs.celltype_col 
    tx_x_col, tx_y_col = cmdargs.tx_x_y_col
    gene_col = cmdargs.gene_col
    cell_col = cmdargs.cell_col 
    leiden_res = cmdargs.leiden_res 
    dr_method = cmdargs.dr_method
    max_centroid_dist = cmdargs.max_centroid_dist
    mask_dist_cutoff = cmdargs.mask_dist_cutoff
    nearest = cmdargs.nearest
    prior_50_reassign_prob = cmdargs.prior_50_reassign_prob
    prior_5_reassign_prob = cmdargs.prior_5_reassign_prob
    
    
    m = misc(cell_centroid_x_col=cell_centroid_x_col,
             cell_centroid_y_col=cell_centroid_y_col,
            celltype_col=celltype_col,
            tx_x_col=tx_x_col,
            tx_y_col=tx_y_col,
            gene_col=gene_col,
            cell_col=cell_col,
            leiden_res=leiden_res,
            dr_method=dr_method,
            max_centroid_dist=max_centroid_dist,
            mask_dist_cutoff=mask_dist_cutoff,
            nearest=nearest,
            prior_50_reassign_prob=prior_50_reassign_prob,
            prior_5_reassign_prob=prior_5_reassign_prob)
    
    cell_metadata = cmdargs.cell_metadata
    cell_boundary_polygons = cmdargs.cell_boundary_polygons
    detected_transcripts = cmdargs.detected_transcripts
    cell_by_gene_counts = cmdargs.cell_by_gene_counts
    
    m.import_data(cell_metadata=cell_metadata,
                  cell_boundary_polygons=cell_boundary_polygons,
                  detected_transcripts=detected_transcripts,
                  cell_by_gene_counts=cell_by_gene_counts)
    
    percent_cell_per_patch = cmdargs.percent_cell_per_patch
    num_overlap = cmdargs.num_overlap
    m.patchfy_data(percent_cell_per_patch=percent_cell_per_patch,
                   num_overlap=num_overlap)
    
    m.initialize_parameters()
    
    n_epochs = cmdargs.n_epochs
    
    m.training_loop(n_epochs=n_epochs)
    m.compute_reassign_probs()
    
    
    criterion = cmdargs.criterion
    if criterion != "auto":
        try:
            criterion = float(criterion)
        except TypeError:
            print("Only a criterion within [0,1] or 'auto' is allowed.") 
    m.reassign_tx(criterion=criterion)
    
    # saving model 
    m.save_model(dir_name=cmdargs.dir_name,
                 model_name=cmdargs.model_name,
                 save_reassigning_result=True)
    sys.exit(0)


if __name__ == "__main__":
    cmdargs = parser.parse_args()
    main(cmdargs=cmdargs)