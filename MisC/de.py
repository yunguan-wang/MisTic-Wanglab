# Data IO
import scanpy as sc
import pandas as pd
import numpy as np 
# Data manipulation 
from itertools import combinations
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats


def de_deseq2(adata,
              layer,
              num_rep: int=3,
              method: str="split",
              n_cpus: int=8):
    
    aug_sample = adata.to_df(layer).copy()
    aug_sample['leiden'] = adata.obs['leiden'].copy()
    aug_sample.reset_index(drop=False, names=['cell_id'],inplace=True)
    for l in adata.uns[layer+"_leiden"]:
        temp = adata.to_df(layer).loc[adata.obs['leiden']!=l, :].copy()
        temp['leiden'] = "cell_type_m_"+l
        temp.reset_index(drop=False, names=['cell_id'],inplace=True)
        aug_sample = pd.concat([aug_sample, temp], axis=0, join='inner', ignore_index=True)

    counts_list = []
    if method == "split":
        rep_sample = aug_sample.groupby(['leiden'],
                                        observed=True,
                                        as_index=True).apply(lambda x: x.sample(frac=1, replace=False))
        rep_sample.reset_index(drop=False, names=['leiden', 'cell_id'], inplace=True)
        rep_sample.drop(columns=['cell_id'], inplace=True)
        rep_counts = rep_sample.groupby("leiden", as_index=True).apply(lambda x: np.array_split(x, num_rep), include_groups=False)
        for l in rep_counts.index:
            for rep in range(num_rep):
                temp = rep_counts[l][rep].sum(axis=0).to_frame().T
                temp['leiden'] = l
                counts_list.append(temp)
    else:   
        for _ in range(num_rep):
            rep_sample = aug_sample.groupby(['leiden'],
                                        observed=True,
                                        as_index=True).apply(lambda x: x.sample(frac=1, replace=True))
            rep_sample.reset_index(drop=False, names=['leiden', 'cell_id'], inplace=True)
            rep_sample.drop(columns=['cell_id'], inplace=True)
            rep_counts = rep_sample.groupby(by=['leiden'], observed=True, as_index=False).sum()
            counts_list.append(rep_counts)
    counts_df = pd.concat(counts_list, axis=0).reset_index(drop=True)    
    counts_df["sample_id"] = "Sample"+counts_df.index.astype(str)
    metadata = counts_df[["sample_id", "leiden"]].set_index("sample_id", inplace=False)
    counts_df.drop(columns=['leiden'], inplace=True)
    counts_df.set_index('sample_id', inplace=True)
    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors="leiden",
        refit_cooks=True,
        inference=inference,
    )
    dds.deseq2()
    
    exp_1v1_list = []
    for neighbor_celltype, celltype in combinations(adata.uns[layer+"_leiden"], 2):
        stat_res = DeseqStats(dds, 
                              inference=inference, 
                              contrast=['leiden', neighbor_celltype, celltype],
                              quiet=True)
        stat_res.summary() 
        exp_1v1_list.append([celltype, neighbor_celltype, 
                         list(stat_res.results_df.index), 
                         list(stat_res.results_df['log2FoldChange']),
                         list(stat_res.results_df['padj'])])

    exp_1v1_df0 = pd.DataFrame(exp_1v1_list, columns=["cell_type", 'neighbor_celltype',
                                            'gene', 'log2FoldChange', 'padj']).reset_index(drop=True)
    exp_1v1_df0 = exp_1v1_df0.explode(column=['gene', 'log2FoldChange', 'padj'])
    exp_1v1_df0['padj'] = -np.log10(exp_1v1_df0['padj'].astype(float)+1e-7)
    exp_1v1_df0['log2FoldChange'] = exp_1v1_df0['log2FoldChange'].astype(float)
    
    exp_1v1_df1 = exp_1v1_df0.copy()
    exp_1v1_df1['cell_type'], exp_1v1_df1['neighbor_celltype'] = exp_1v1_df1['neighbor_celltype'], exp_1v1_df1['cell_type']
    exp_1v1_df1['log2FoldChange'] *= -1
    
    exp_1v1_df = pd.concat([exp_1v1_df0, exp_1v1_df1], axis=0).reset_index(drop=True) 
    
    exp_1vR_list = []
    for celltype in adata.uns[layer+"_leiden"]:
        stat_res = DeseqStats(dds, 
                              inference=inference, 
                              contrast=['leiden', "cell_type_m_"+celltype, celltype],
                              quiet=True)
        stat_res.summary() 
        exp_1vR_list.append([celltype, 
                                list(stat_res.results_df.index), 
                                list(stat_res.results_df['log2FoldChange']),
                                list(stat_res.results_df['padj'])]) 
    exp_1vR_df = pd.DataFrame(exp_1vR_list, columns=["cell_type", 'gene', 'log2FoldChange', 'padj']).reset_index(drop=True) 
    exp_1vR_df = exp_1vR_df.explode(column=['gene', 'log2FoldChange', 'padj'])
    exp_1vR_df['padj'] = -np.log10(exp_1vR_df['padj'].astype(float)+1e-7)
    exp_1vR_df['log2FoldChange'] = exp_1vR_df['log2FoldChange'].astype(float)
        
    return exp_1v1_df, exp_1vR_df



def de_wilcoxon(adata,
                layer):
    pass 


def de(adata, 
       layer,
       method,
       **kwargs):
    pass 