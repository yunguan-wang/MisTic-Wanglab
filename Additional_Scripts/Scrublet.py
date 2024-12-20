import scrublet as scr
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import pkg_resources

# check annoy package
print(pkg_resources.get_distribution('annoy'))

# Set up arguments
parser = argparse.ArgumentParser()
parser.add_argument("sample_name", help="sample name")
parser.add_argument("input_dir", help="path to Cell Ranger output, ending with the 'filtered_feature_bc_matrix' directory")
parser.add_argument("output_dir", help="path to directory in which to create output directories")
parser.add_argument("--threshold", type=float, default=-1, help="threshold for re-running Scrublet, based on original run barplot. default = automatic threshold")
parser.add_argument("--expected", type=float, default=0.06, help="expected doublet rate for Scrublet. default = 0.06")
args = parser.parse_args()

# Set up output directory
output_dir=os.path.join(args.output_dir, args.sample_name)
os.makedirs(output_dir, exist_ok = True)

# Load in data for sample
counts_matrix = scipy.io.mmread(args.input_dir + '/matrix.mtx.gz').T.tocsc()
genes = np.array(scr.load_genes(args.input_dir + '/features.tsv', delimiter='\t', column=1))

print('Counts matrix shape: {} rows, {} columns'.format(counts_matrix.shape[0], counts_matrix.shape[1]))
print('Number of genes in gene list: {}'.format(len(genes)))

# Initialize Scrublet object
scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=args.expected)

# Run default scrublet pipeline
doublet_scores, predicted_doublets = scrub.scrub_doublets(min_counts=2, 
                                                          min_cells=3, 
                                                          min_gene_variability_pctl=85, 
                                                          n_prin_comps=30)

if args.threshold != -1:
    # Adjust threshold
    scrub.call_doublets(threshold=args.threshold)
    fig=scrub.plot_histogram();
    fig2=fig[0]
    fig2.savefig(os.path.join(output_dir, "firstPlot_recall.pdf"))
else:
    # Plot doublet score histograms for observed transcriptomes and simulated doublets
    fig=scrub.plot_histogram();
    fig2=fig[0]
    fig2.savefig(os.path.join(output_dir, "firstPlot.pdf"))

# Save predicted doublet text files
np.savetxt(os.path.join(output_dir, "predicted_doublets.txt"), np.array(predicted_doublets))
np.savetxt(os.path.join(output_dir, "doublet_scores.txt"), np.array(doublet_scores))

# Plot doublet predictions on UMAP
scrub.set_embedding('UMAP', scr.get_umap(scrub.manifold_obs_, 10, min_dist=0.3))

fig=scrub.plot_embedding('UMAP', order_points=True);
fig2=fig[0]
fig2.savefig(os.path.join(output_dir, "secondPlot.pdf"))

# Save data.frame of results
BCs = np.loadtxt(args.input_dir + '/barcodes.tsv.gz', dtype="U")
df = pd.DataFrame({'PredictedDoublets': predicted_doublets, 'DoubletScores': doublet_scores}, BCs)
df.to_csv(os.path.join(output_dir, "results_df.csv"))