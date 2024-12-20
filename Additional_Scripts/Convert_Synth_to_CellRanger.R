######################
#                    #
# Convert Synth data #
#   to CellRanger    #
#  12 December 2024  #
#  Erica DePasquale  #
#                    #
######################

# Load libraries
library(DropletUtils)
library(Matrix)

# Turn files from Yunguan's pipeline into Cell Ranger output
A=read.table("/data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Synthetic_spatial_data/merscope_hcc1/synthetic_counts_with_doublets.csv", sep=",", header=T, row.names=1)
dir.create("/data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Synthetic_spatial_data/merscope_hcc1/filtered_feature_bc_matrix")
setwd("/data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Synthetic_spatial_data/merscope_hcc1/filtered_feature_bc_matrix")
A=t(A)
B <- as(A, "sparseMatrix") 
write10xCounts(path="/data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Synthetic_spatial_data/merscope_hcc1/filtered_feature_bc_matrix", x=B, version="3")