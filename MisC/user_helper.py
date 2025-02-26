# import os 
# import h5py
# import pandas as pd
# from geopandas import GeoDataFrame  
# from shapely import Polygon
# import numpy as np 
# from itertools import combinations
# from typing import Tuple 


# def mask_eval(l: list) -> Tuple[bool, list]:
#     """Given a list of coordinates, check if it contains only on polygon

#     Parameters
#     ----------
#     l : list
#         List of potentially vertices of different polygons

#     Returns
#     -------
#     Tuple[bool, list]
#         If the list defines only one polygon, and the coordinates of its vertices 
#     """
#     unique_l = []
#     for a,b in combinations(l,2):
#         if (len(unique_l) ==0) or (a not in np.array(unique_l)):
#             unique_l.append(a)
#         if np.array_equal(a, b):
#             continue
#         else:
#             if b not in np.array(unique_l):
#                 unique_l.append(b)
#     return len(unique_l) > 1, unique_l


# def assemble_cell_coords(cls, 
#                              input_path: str,
#                              output_path: str) -> None:
#         """Assemble the file containing the polygons of cell masks

#         Parameters
#         ----------
#         input_path : str
#             The input folder
#         output_path : str
#             The output folder
#         """
#         boundries_fn = os.listdir(input_path + '/cell_boundaries')
#         for bfn in boundries_fn:
#             cell_coords = pd.Series()
#             # print(bfn)
#             bfn = os.path.join(input_path, 'cell_boundaries', bfn)
#             f = h5py.File(bfn,'r')
#             diff_coords = []
#             for cell in list(f['featuredata']):
#                 coords = []
#                 for i in range(7):
#                     tmp = np.array((f['featuredata'][cell]['zIndex_'+str(i)]['p_0']['coordinates'][0]))
#                     coords.append(tmp)
#                 non_unique, unique_v = mask_eval(coords)
#                 if non_unique:
#                     print(cell)
#                 else: 
#                     cell_coords[cell] = unique_v
#             print(bfn)
#             f.close()
#             cell_coords = cell_coords.to_frame(name='coord')
#             cell_coords['X'] = cell_coords.coord.apply(lambda x: '_'.join(x[0][:,0].round(2).astype(str)))
#             cell_coords['Y'] = cell_coords.coord.apply(lambda x: '_'.join(x[0][:,1].round(2).astype(str)))
#             if os.path.exists(os.path.join(input_path, 'cell_coords.csv')):
#                 cell_coords.iloc[:,1:].to_csv(input_path + '/cell_coords.csv', mode='a', header=False)
#             else:
#                     cell_coords.iloc[:,1:].to_csv(input_path + '/cell_coords.csv')
        
#         cell_masks.index = ['cell_' + str(x+1) for x in cell_masks.index]
#         cell_masks = cell_masks.loc[spacia_meta.index]
#         cell_masks['n_polygon'] = cell_masks.X.apply(lambda x: len(x.split('_')))
#         cell_masks.X = cell_masks.X.str.split('_')
#         cell_masks.Y = cell_masks.Y.str.split('_')
#         cell_masks['polygon'] = cell_masks.apply(
#             lambda x: Polygon([(x.X[i], x.Y[i]) for i in range(x.n_polygon)]), axis=1)
#         # Save as geopandas parquet file
#         GeoDataFrame(
#             cell_masks[['polygon']],geometry='polygon'
#             ).to_parquet('cell_polygons.parquet', index=True)
# def final_reassign_tx(self,
#                         selected_criterion: str) -> None:
#     """Make final transcript reassignment 

#     Parameters
#     ----------
#     selected_criterion : str
#         The criterion name 
#     """
#     tx_to_reassign = self.tx_to_reassign_dict[self.current_layer+"_"+selected_criterion].copy()
#     self.adata = make_reassignment_adata(adata=self.adata, 
#                                             layer=self.current_layer,
#                                             tx_to_reassign=tx_to_reassign,
#                                             preprocess=self.import_data_par['preprocess'])
#     self.tx_metadata = make_reassignment_tx_metadata(tx_to_reassign=tx_to_reassign,
#                                                         tx_metadata=self.tx_metadata)
#     # Move the current layer one step further 
#     layer_num = extract_layer_num(self.current_layer)
#     self.current_layer = "counts_"+str(int(layer_num+1))
#     self.adata.uns['current_layer'] = self.current_layer

# def recluster(self,
#                 temperature: float=0.0,
#                 top_k: Optional[int]=None,
#                 new_layer: Optional[str]=None) -> None:
#     """Regenerate the clusters 

#     Parameters
#     ----------
#     temperature : float, optional
#         Temperature in sampling multinomial, by default 0.0
#     top_k : Optional[int], optional
#         Only the top k candidates will be sampled, by default None
#     new_layer : Optional[str], optional
#         Layer upon which the logits will be computed, by default None
#     """
#     if new_layer is None:
#         new_layer = self.current_layer
#     self.eval()
#     with torch.no_grad():
#         cell_by_gene_counts_chunks = even_split(array=self.adata.layers[new_layer],
#                                                 chunk_size=1000)
#         logits = torch.empty((0, self.adata.uns["n_leiden"]), dtype=torch.float32)
#         cell_type_predict = torch.empty((0, 1), dtype=torch.int64)
#         for cell_by_gene_counts_chunk in cell_by_gene_counts_chunks:
#             logits_chunk = self.cell_type_coefficients(torch.tensor(cell_by_gene_counts_chunk,
#                                                             dtype=torch.float32, 
#                                                             device=self.model_device)).cpu()
#             logits = torch.cat((logits, logits_chunk), dim=0)
#             # For top k, the -inf mask trick is used 
#             if top_k is not None:
#                 top_logits, _ = torch.topk(logits_chunk, top_k)
#                 min_val = top_logits[:, -1]
#                 logits_chunk = torch.where(
#                     logits_chunk < min_val,
#                     torch.tensor(float('-inf')).to(logits.device),
#                     logits_chunk
#                 )
#             # If temperature is not 0, we sample from multinomial 
#             if temperature > 0.0:
#                 logits_chunk = logits_chunk/temperature
#                 probs = torch.softmax(logits_chunk, dim=-1)
#                 cell_type_predict_chunk = torch.multinomial(probs, num_samples=1)
#             else:
#                 cell_type_predict_chunk = torch.argmax(logits_chunk, dim=-1, keepdim=True)
#             cell_type_predict = torch.cat((cell_type_predict, cell_type_predict_chunk), dim=0)
#     # Record the results 
#     cell_type_predict = cell_type_predict.numpy()
#     logits = logits.numpy()
#     probs = softmax(logits, axis=1)
#     # perplexity is also computed to see how uncertain the model is 
#     perplexity = np.exp(entropy(probs, axis=1, keepdims=True))
#     # To allow the user to recluster multiple times 
#     # by running the recluster method multiple times 
#     # The first time the user runs recluster, the index will be 0
#     # after that every time the user runs recluster, the index will 
#     # increment by 1
#     i=0
#     while True:
#         new_leiden_name = new_layer + "_leiden_" + str(i)
#         new_cell_type_name = new_layer + "_cell_type_" + str(i)
#         if new_leiden_name not in self.adata.obs.columns:
#             break 
#         i += 1
#     self.adata.obs[new_leiden_name] = cell_type_predict
#     self.adata.obs[new_leiden_name] = self.adata.obs[new_leiden_name].astype(str)
    
#     temp_df = self.adata.obs[[new_leiden_name]].merge(self.adata.uns['cell_type_leiden_map'],
#                                         how='left', left_on = new_leiden_name,
#                                         right_on = "cell_type_index")
#     self.adata.obs[new_cell_type_name] = temp_df['cell_type_name'].values.copy()
    
#     self.adata.obs[new_leiden_name+"_perplexity"] = perplexity
#     self.adata.obs['leiden'] = self.adata.obs[new_leiden_name].copy()