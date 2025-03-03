# import shapely
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import pandas as pd
# import seaborn as sns

# from typing import Optional


# def plot_boundaries(
#           cell_coords, cg_pairs, adata, bbox = [2000, 4700, 800, 800]):
#     '''
#     image windows in um, bot_left_x, top_left_y, x_length, y_length.
#     adata must be the raw object.
#     '''
#     if bbox is not None:
#         bounding_box = shapely.geometry.box(
#             minx=bbox[0], 
#             miny=bbox[1], 
#             maxx=bbox[0] + bbox[2], 
#             maxy=bbox[1] + bbox[3])
#         cell_polys = cell_coords[cell_coords['Geometry'].within(bounding_box)]
#     else:
#         cell_polys = cell_coords
#     # cmaps = ['Reds', 'Blues']
#     annotate_cells = []
#     annotate_colors = []
#     for cg in cg_pairs:
#         _cells = adata[adata.obs.leiden==cg[0]].obs_names
#         _cells = [x for x in _cells if x in cell_polys.index]
#         _cell_color = cell_polys.loc[_cells,'color'][0]
#         annotate_cells += _cells
#         _gene_v = adata.to_df().loc[_cells, cg[1]]
#         norm = mpl.colors.Normalize(vmin=-0.25, vmax=0.75, clip=True)
#         cmap = mpl.colors.LinearSegmentedColormap.from_list(
#             '', ['#ffffff', _cell_color])
#         mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
#         gene_colors = list(map(lambda x: mpl.colors.to_hex(x), mapper.to_rgba(_gene_v)))
#         annotate_colors += gene_colors
#     gene_colors = pd.DataFrame(annotate_colors, index = annotate_cells, columns=['fill_color'])
#     try:
#         cell_polys.drop('fill_color',axis=1, inplace=True)
#     except:
#         pass
#     cell_polys = cell_polys.merge(gene_colors, left_index=True, right_index=True, how='left')
#     cell_polys.fill_color.fillna('w', inplace=True)
#     plt.figure(figsize=(10, 10), facecolor="white") 
#     for _ , row in cell_polys.iterrows():
#             shape = row["Geometry"]
#             if shape.geom_type.startswith('Multi'):
#                 for geom in shape.geoms:
#                     plt.plot(*geom.exterior.xy, color=row["color"], linewidth=2)
#                     plt.fill(*geom.exterior.xy, color=row["fill_color"], alpha=0.75)
#     handles = []                
#     for g, row in color_dict.iterrows():
#         handles.append(mpl.patches.Patch(color=row.color, label=g))
#     plt.legend(handles=handles, loc = 'center left', bbox_to_anchor=(1,0.5))



# def plot_tx(cell_id, 
#             cell_coords, 
#             current_intf_tx,
#             title):
#     fig, axs = plt.subplots()
#     cell = cell_id
#     cell_geom = cell_coords.loc[cell, 'Geometry']
#     neighbor_geoms = current_intf_tx[current_intf_tx.cell_id==cell]['mask_geom'].unique()
#     txs = current_intf_tx[current_intf_tx.cell_id==cell].sort_values('tx_mask_distance')
#     txs = txs.groupby(txs.index).first()
#     tx = txs['tx_geom'].values
#     tx_dist = txs['tx_mask_distance'].values
#     tx_pct_diff = txs['pct_diff'].values
#     for geom in cell_geom.geoms:
#         xs, ys = geom.exterior.xy
#         axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')

#     for m in neighbor_geoms:
#         for geom in m.geoms:  
#             xs, ys = geom.exterior.xy
#             axs.fill(xs, ys, alpha=0.5, fc='b', ec='none')
#     xs = []
#     ys = []
#     for t in tx:
#         x, y = t.xy    
#         xs.append(x[0])
#         ys.append(y[0])
#     # sns.scatterplot(x = xs, y = ys, hue=tx_dist)
#     sns.scatterplot(x = xs, y = ys, hue=tx_pct_diff, palette='coolwarm').set_title(title)
    