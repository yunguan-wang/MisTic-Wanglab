import numpy as np
import doubletdetection
import scanpy as sc
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

sc.settings.n_jobs=8
sc.set_figure_params()

# Set up arguments
parser = argparse.ArgumentParser()
parser.add_argument("sample_name", help="sample name")
parser.add_argument("input_dir", help="path to Cell Ranger output, ending with the 'filtered_feature_bc_matrix' directory")
parser.add_argument("output_dir", help="path to directory in which to create output directories")
parser.add_argument("algorithm", help="clustering algorithm, options are phenograph, louvain, and leiden")
parser.add_argument("threshold", type=float, help="voter threshold. suggested to start 0.5")
args = parser.parse_args()

# Set up output directory
output_dir=os.path.join(args.output_dir, args.sample_name)
os.makedirs(output_dir, exist_ok = True)

# Load count matrix
#adata = sc.read_10x_h5(
#    "pbmc_10k_v3_filtered_feature_bc_matrix.h5",
#    backup_url="https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5"
#)
adata = sc.read_10x_mtx(args.input_dir)
adata.var_names_make_unique()

# Remove "empty" genes
sc.pp.filter_genes(adata, min_cells=1)

# Run Doublet Detection
clf = doubletdetection.BoostClassifier(
    n_iters=10,
    clustering_algorithm=args.algorithm,
    standard_scaling=True,
    pseudocount=0.1,
    n_jobs=-1,
)
doublets = clf.fit(adata.X).predict(p_thresh=1e-16, voter_thresh=args.threshold)
doublet_score = clf.doublet_score()

adata.obs["doublet"] = doublets
adata.obs["doublet_score"] = doublet_score

# Save predicted doublet text files
np.savetxt(os.path.join(output_dir, "predicted_doublets.txt"), np.array(doublets))
np.savetxt(os.path.join(output_dir, "doublet_scores.txt"), np.array(doublet_score))

# Visualize Results
f = doubletdetection.plot.convergence(clf, show=True, p_thresh=1e-16, voter_thresh=0.5)
f.savefig(os.path.join(output_dir, "Convergence.pdf"))

# Doublets on auto generated UMAP
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

#umap
f1=sc.pl.umap(adata, color=["doublet", "doublet_score"], return_fig=True)
f1.savefig(os.path.join(output_dir, "UMAP.pdf"))

#violin
ax1=sc.pl.violin(adata, "doublet_score", show=False)
ax1.get_figure().savefig(os.path.join(output_dir, "Violin.pdf"))

# Heatmap for number of predicted doublets at different thresholds
f2 = doubletdetection.plot.threshold(clf, show=True, p_step=6)
f2.savefig(os.path.join(output_dir, "Heatmap.pdf"))

# Save data.frame of results
BCs = np.loadtxt(args.input_dir + '/barcodes.tsv.gz', dtype="U")
df = pd.DataFrame({'PredictedDoublets': doublets, 'DoubletScores': doublet_score}, BCs)
df.to_csv(os.path.join(output_dir, "results_df.csv"))