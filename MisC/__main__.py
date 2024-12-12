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
parser.add_argument("--tx_x_y_col", nargs=2, type=str, default=["global_x", "global_y"], 
                    help="The column containing the x and y coordinates of transcript in detected transcript file")
parser.add_argument("--gene_col", type=str, default="gene")
parser.add_argument("--cell_col", type=str, default="cell_id")
parser.add_argument("--celltype_col", nargs='?', type=str)
parser.add_argument("--leiden_res", type=float, default=1)
parser.add_argument("--no_preprocess", action="store_true")
parser.add_argument("--max_centroid_dist", type=float, default=50)
parser.add_argument("--min_centroid_dist", type=float, default=0)
parser.add_argument("--mask_dist_cutoff", type=float, default=1)
parser.add_argument("--num_rep", type=int, default=3)
parser.add_argument("--method", type=str, default="split")
parser.add_argument("--reparametrize", action="store_true")

# Import data 
parser.add_argument("--cell_metadata", type=str, required=True)
parser.add_argument("--cell_boundary_polygons", type=str, required=True)
parser.add_argument("--detected_transcripts", type=str, required=True)
parser.add_argument("--cell_by_gene_counts", type=str)

# Model training 
parser.add_argument("--prior_50_reassign_prob", type=float, default=0.01)
parser.add_argument("--prior_5_reassign_prob", type=float, default=0.5)
parser.add_argument("--n_epochs", type=int, default=1)

# reassign tx 
parser.add_argument("--criteria", type=float, default=0.5)

# Model saving 
parser.add_argument("--path", type=str, default="./misc.pt")

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
    tx_x_col, tx_y_col = cmdargs.tx_x_y_col
    gene_col = cmdargs.gene_col
    cell_col = cmdargs.cell_col 
    celltype_col = cmdargs.celltype_col 
    leiden_res = cmdargs.leiden_res 
    preprocess = not cmdargs.no_preprocess
    max_centroid_dist = cmdargs.max_centroid_dist
    min_centroid_dist = cmdargs.min_centroid_dist
    mask_dist_cutoff = cmdargs.mask_dist_cutoff
    num_rep = cmdargs.num_rep
    method = cmdargs.method
    reparametrize = cmdargs.reparametrize
    
    
    m = misc(cell_centroid_x_col=cell_centroid_x_col,
             cell_centroid_y_col=cell_centroid_y_col,
             tx_x_col=tx_x_col,
             tx_y_col=tx_y_col,
             gene_col=gene_col,
             cell_col=cell_col,
             celltype_col=celltype_col,
             leiden_res=leiden_res,
             preprocess=preprocess,
             max_centroid_dist=max_centroid_dist,
             min_centroid_dist=min_centroid_dist,
             mask_dist_cutoff=mask_dist_cutoff,
             num_rep=num_rep,
             method=method,
             reparametrize=reparametrize)
    
    cell_metadata = cmdargs.cell_metadata
    cell_boundary_polygons = cmdargs.cell_boundary_polygons
    detected_transcripts = cmdargs.detected_transcripts
    cell_by_gene_counts = cmdargs.cell_by_gene_counts
    
    m.import_data(cell_metadata=cell_metadata,
                  cell_boundary_polygons=cell_boundary_polygons,
                  detected_transcripts=detected_transcripts,
                  cell_by_gene_counts=cell_by_gene_counts)
    
    m.patchfy_data()
    
    prior_50_reassign_prob = cmdargs.prior_50_reassign_prob
    prior_5_reassign_prob = cmdargs.prior_5_reassign_prob
    
    m.initiate_parameters(prior_50_reassign_prob=prior_50_reassign_prob,
                          prior_5_reassign_prob=prior_5_reassign_prob)
    
    n_epochs = cmdargs.n_epochs
    
    m.training_loop(n_epochs=n_epochs,
                    verbose=False)
    
    criteria = {"soft": cmdargs.criteria}
    m.trial_reassign_tx(criteria=criteria)
    m.final_reassign_tx(selected_criterion="soft")
    
    # reclustering 
    
    
    # saving model 
    m.save_model(path=cmdargs.path)
    sys.exit(0)


if __name__ == "__main__":
    cmdargs = parser.parse_args()
    main(cmdargs=cmdargs)