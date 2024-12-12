########################
#                      #
# scds from CellRanger #
#    4 October 2024    #
#   Erica DePasquale   #
#                      #
########################

# Load libraries
library(SingleCellExperiment)
library(DropletUtils)
library(scds)
library(plotROC)
library(pROC)

# turn off scientific notation
options(scipen=999)

# Load in relevant data
sce_Synth=read10xCounts("/data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Synthetic_spatial_data/merscope_hcc1/filtered_feature_bc_matrix", col.names=T)

#- Annotate doublet using co-expression based doublet scoring:
sce_Synth = cxds(sce_Synth)

#- Annotate doublet using binary classification based doublet scoring:
sce_Synth = bcds(sce_Synth)

#- Combine both annotations into a hybrid annotation
sce_Synth = cxds_bcds_hybrid(sce_Synth)

#- Doublet scores are now available via colData:
CD  = colData(sce_Synth)
head(cbind(CD$cxds_score, CD$bcds_score, CD$hybrid_score))

# Plots for threshold determination
pdf(file = "/data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Testing_tools/Scds/scds_Synth_thresholds.pdf", width = 10, height = 7)
par(mar=c(4, 4, 4, 4))
  plot(sort(CD$cxds_score, decreasing=T))
  plot(sort(CD$bcds_score, decreasing=T))
  plot(sort(CD$hybrid_score, decreasing=T))
dev.off()

# Read in metadata
D=read.table("/data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Synthetic_spatial_data/merscope_hcc1/synthetic_counts_tx_metadata.csv", sep=",", header=T)
#### NOTE: this will become a parquet file, so i will need a new tool to parse that in the future

# Combine predicted and real doublets
doublets=unique(D[,1])
E=cbind(CD[,2:5], GroundTruth=C$X %in% doublets)

# Make plot
i=which(E$GroundTruth==TRUE)
E$GroundTruth=rep(0, length(E$GroundTruth))
E$GroundTruth[i]<-1
basicplot1 <- ggplot(E, aes(d = GroundTruth, m = cxds_score)) + geom_roc() # not really predictive
basicplot1

basicplot2 <- ggplot(E, aes(d = GroundTruth, m = bcds_score)) + geom_roc() # not really predictive
basicplot2

basicplot3 <- ggplot(E, aes(d = GroundTruth, m = hybrid_score)) + geom_roc() # not really predictive
basicplot3

pdf(file = "/data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Testing_tools/Scds/scds_Synth_curve.pdf", width = 10, height = 10)
par(mar=c(4, 4, 4, 4))
 basicplot1
 basicplot2
 basicplot3
dev.off()

# Calculate AUC 
#TODO: start with a hybrid threshold of 1, but can be changed based on the plots above
doublets_yes=which(E$hybrid_score>=1)
doublets_no=which(E$hybrid_score<1)
myPreds=E$hybrid_score
myPreds[doublets_yes]<-"1"
myPreds[doublets_no]<-"0"
roc_object <- roc(E$GroundTruth, as.numeric(myPreds))
auc(roc_object) # Area under the curve: 0.5086, not predictive at that threshold/algorithm combination

