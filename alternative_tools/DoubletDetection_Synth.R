####################
#                  #
# DoubletDetection #
#  for Synth data  #
# 12 December 2024 #
# Erica DePasquale #
#                  #
####################

# Load libraries
library(Matrix)
library(plotROC)
library(pROC)

# Read in results and metadata
C=read.table("/data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Testing_tools/DoubletDetection/DoubletDetection_Synth/Synth/results_df.csv", sep=",", header=T)
D=read.table("/data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Synthetic_spatial_data/merscope_hcc1/synthetic_counts_tx_metadata.csv", sep=",", header=T)
#### NOTE: this will become a parquet file, so i will need a new tool to parse that in the future

# Combine predicted and real doublets
doublets=unique(D[,1])
E=cbind(C, GroundTruth=C$X %in% doublets)

# Make plot
i=which(E$GroundTruth==TRUE)
E$GroundTruth=rep(0, length(E$GroundTruth))
E$GroundTruth[i]<-1
basicplot <- ggplot(E, aes(d = GroundTruth, m = DoubletScores)) + geom_roc() # not predictive, lol

pdf(file = "/data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Testing_tools/DoubletDetection/DoubletDetection_Synth/DoubletDetection_Synth_curve.pdf", width = 10, height = 10)
par(mar=c(4, 4, 4, 4))
basicplot
dev.off()

# Calculate AUC
roc_object <- roc(E$GroundTruth, E$PredictedDoublets)
auc(roc_object) # Area under the curve: 0.4494

