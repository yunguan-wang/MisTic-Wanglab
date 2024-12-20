#BSUB -W 1:00
#BSUB -o DoubletDetection_Synth.out
#BSUB -J DoubletDetection
#BSUB -M 10000

module load bzip2 python3/3.11.3 git/2.22.0 jdk/20.0.1

source $HOME/DoubletDetection/bin/activate
python DoubletDetection.py Synth /data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Synthetic_spatial_data/merscope_hcc1/filtered_feature_bc_matrix DoubletDetection_Synth louvain 0.5
