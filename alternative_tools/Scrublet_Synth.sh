#BSUB -W 0:30
#BSUB -o Scrublet_Synth.out
#BSUB -J Scrublet_Synth
#BSUB -M 10000
#BSUB -m bmi-200m5-09

module load bzip2 python3/3.11.3 git/2.22.0 jdk/20.0.1

source $HOME/SCENICplus/bin/activate
python Scrublet.py Synth /data/GI-Informatics/DePasquale/Projects/SpatialDoublet/Synthetic_spatial_data/merscope_hcc1/filtered_feature_bc_matrix Scrublet_Synth --threshold 0.3
