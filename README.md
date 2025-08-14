![Logo](/assets/logo.png)

# MisTIC
> A probabilistic model for correcting mis-assigned transcripts due to cell segmentation error in imaging-based spatial transcriptomics. It builds on top of [PyTorch] and [scanpy].

![forthebadge](/assets/spatial-transcriptomics.svg)
![forthebadge](/assets/scanpy-pytorch.svg)

## Installation :computer:

### Create a virtual environment :snake:

So far, we have only tested the software on Python 3.9 and 3.10.

```shell
conda create -n mistic python=3.9
conda activate mistic 
```

or 

```shell
conda create -n mistic python=3.10
conda activate mistic 
```

### Build the package

#### pip 

For a stable version of MisTIC, you can download and install the package via 

```shell
pip install txMisTIC
```

#### Local 

The latest version of MisTIC will be hosted on GitHub where we constantly update features of the package. To use the latest version of MisTIC, you will need to build the package loacally.

First, you need to clone the repo to a local directory, say `./awesome_repos` and `cd` to that folder. 

Now, you should have a `MisTIC` folder under the `awesome_repos` directory. Run the following to build the package.

```shell 
cd ./MisTIC
python setup.py sdist bdist_wheel
```

You should see a `dist` folder now which contains the wheel file you will need for installing the package. 

Remember to change the `VERSION` to match the file you see in the `dist` folder.

```shell
cd ./dist
pip install ./txMisTIC-VERSION-py3-none-any.whl
```

Once you have installed the package, you can quickly test if the installation is successful via 

```shell
python -m MisTIC --version
```

### Dependencies 

With both package building strategies, the dependencies should be installed automatically. Here, we just list them out for your reference.  

- python>=3.9,<3.11
- torch==1.13
- shapely==2.0
- geopandas==1.0
- pydeseq2>=0.4.6,<0.5
- scanpy==1.10
- numpy>=1.24,<2.0
- anndata>=0.10,<0.11
- pyarrow==16.1
- polars==1.21
- jupyter
- ipywidgets

## Input format
The input for MisTIC consists of four pieces of data: 

1. `detected_transcripts`: A dataframe recording information on detected transcripts. It has to contain at least four columns: 
    1. `global_x`: The x-coordinates of the transcripts in microns. Note that some data will contain another column that appears to also be some x-coordinates. It might be the local x-coordinates in the field of view (FOV). We need the global coordinates. 
    2. `global_y`: The y-coordinates of the transcripts in microns. Note that some data will contain another column that appears to also be some y-coordinates. It might be the local y-coordinates in the field of view (FOV). We need the global coordinates. 
    3. `gene`: The gene types of the transcripts. 
    4. `cell_id`: The IDs of the assigned cells. 
    * Note that the column names in your dataset might not be exactly what we specified here, but you can inform `MisTIC` the names of your dataset that can be used as those four columns. 
    * An index column or a column for IDs for the transcripts are NOT necessary. MisTIC will internally generate one. 
2. `cell_metadata`: A dataframe containing meta information on the segmented cells. It has to contain at least three columns: 
    1. `cell_id`: IDs of segmented cells. MisTIC assumes that the first column of the dataframe will be the cell IDs. Make sure that the cell IDs correspond to those recorded in the `detected_transcript` dataframe.
    2. `center_x`: Computed x-coordinates of the centroids of the cells in microns. 
    3. `center_y`: Computed y-coordinates of the centroids of the cells in microns. 
    4. (Optional) `cell_type`: Annotated cell types. If the users do not provide such information, MisTIC will use the `leiden` algorithm for unsupervised clustering. We highly recommend the users to perform cell typing themselves instead of relying solely on the clustering algorithm.
    * Note that the column names in your dataset might not be exactly what we specified here, but you can inform `MisTIC` the names of your dataset that can be used as those four columns. 
3. `cel_boundary_polygons`: A dataframe recording the polygons of the segmented cells. It has to contain at least two columns: 
    1. `cell_id`: IDs of segmented cells. MisTIC assumes that the index column of the dataframe will be the cell IDs. Make sure that the cell IDs correspond to those recorded in the `detected_transcript` dataframe. 
    2. `polygon`: Polygons defining boundaries of cells. 
    * Note that this dataframe is usually in `.parquet` format. And the `polygon` column should be the `geometry` object in `GeoPandas`. 
    * MisTIC assume that the `geometry` is the `polygon`. Therefore, the specific column name does not really matter as long as the data type is correct. 
4. (Optional) `cell_by_gene_counts`: An optional dataframe of the cell-by-gene count matrix. 
    * The column names should be genes 
    * The first column is assumed to contain IDs of cells 
    * If it's not provided, MisTIC will generate a cell-by-gene matrix based on the `detected_transcript` dataframe. 


We have also included in this package a sample dataset containing only ~200 cells under `tests/test_data` for your reference. 
* Note that due to the large number of detected transcripts, it's NOT recommended to store such information in a `.csv` format as it will take a long time to load the data. A `.parquet` format is recommended. We used `.csv` just so that you can inspect the structure of the data easily. 
* Pay close attention to the unit of the coordinates in your data. MisTIC assumes that all coordinates are in microns. If pixels are recorded, you are responsible for transforming them into microns using the scaling factors provided by your SRT platforms. 

## Tutorial :fast_forward:

### Interactive Python 
This assumes that you are using Jupyter notebook to run MisTIC.

1. Object instantiation

```python
>>> from MisTIC.mistic_class import mistic
>>> # Check and specify the column names!
>>> m = mistic(cell_centroid_x_col='X-COORDINATE COLUMN IN CELLMETA',
                cell_centroid_y_col='Y-COORDINATE COLUMN IN CELLMETA',
                celltype_col="OPTIONAL CELL TYPE COLUMN IN CELLMETA",
                tx_x_col='X-COORDINATE COLUMN IN DETECTED TRANSCRIPTS',
                tx_y_col='Y-COORDINATE COLUMN IN DETECTED TRANSCRIPTS',
                gene_col='GENE TYPE COLUMN IN DETECTED TRANSCRIPTS',
                cell_col='CELL ID COLUMN IN DETECTED TRANSCRIPTS')
```

* At this stage, there are other arguments you can use to tweak the behavior of `MisTIC`. Although the default setting works well based on our experience, you are welcome to change their settings especially the following two arguments: 
    1. `leiden_res`: If cell type annotation is not provided, MisTIC will use the leiden algorithm to get cell type information. You can tweak this parameter to control its resolution.
    2. `nearest`: By default, the top 3 nearest neighbor cells of a transcript is considered. If you want more possibilities, you can increase this number or decrease it to speed up the training process. 


2. Data importing 

Once the object is instantiated, we are now ready to load the data into the object. 

```python
>>> # cell_by_gene_counts is optional
>>> cell_by_gene_counts = "PATH/TO/COUNTS"
>>> detected_transcripts = 'PATH/TO/TX'
>>> cell_metadata = 'PATH/TO/META'
>>> cell_boundary_polygons = "PATH/TO/POLYGONS"
>>> m.import_data(cell_by_gene_counts=cell_by_gene_counts,
                    cell_metadata=cell_metadata,
                    cell_boundary_polygons=cell_boundary_polygons,
                    detected_transcripts=detected_transcripts)
```

This will not only import the data into the `mistic` object, but also perform data curation and generate features necessary for model training and transcript reassignment. 

* Note that the function will create a `molecule_id` column for the `detected_transcripts` file. The first record will be `tx_0`, the second will be `tx_1`, etc..
* We recommend providing MisTIC with the paths to the files. This is fine if you already now the structure of your data. Otherwise, you can also first read in the data and import them into MisTIC. However, MisTIC internally creates deep copies of those data which will eat a huge chunk of memory. 

3. Model training

```python
>>> # Generate minibatches for SGD
>>> m.patchfy_data()
>>> # Modle training 
>>> m.initialize_parameters()
>>> m.training_loop(n_epochs=20)
```

* Note that the algorithm will not necessarily train `20` epochs due to the implementation of an early stopping mechanism. So no need to panic. 

4. Transcript misassignment correction

We first need to compute reassignment probability for each transcript: 

```python
>>> m.compute_reassign_probs()
```

For didactic decision, we allow users to specify two grids over which to search for the best combination of thresholds.

`reassign_threshold_grid` should be an array/list of numbers within [0, 1]. Transcripts with `reassign_probs` greater than the threshold will be reassigned. 

`remove_threshold_grid` should be an array/list of numbers within [0, 1]. For transcripts that are NOT reassigned, a 
separate threshold will be generated. For example, if the `reassign_threshold=0.3` and the `remove_threshold=1/3`, the actual threshold will be `0.3*(1-1/3)=0.2`. Transcripts with `reassign_probs` greater than the threshold will be removed. If removal is not desired, simply set `remove_threshold_grid=0`.

```python
>>> m.correct_tx(reassign_threshold_grid=np.arange(start=0.1, stop=0.6, step=0.1),
                remove_threshold_grid=np.arange(start=0, stop=0.4, step=0.1),
                choice_type="best")
```

* MisTIC allows three types of `choice_type`s
    1. `best`: This is the combination that gives the lowest loss.
    2. `aggressive`: This is the combination that removes the most transcripts and the loss is within 0.01 of the lowest loss. 
    3. `conservative`: This is the combination that does not remove any transcripts and yields the lowest loss. 
* More advanced: If you are want more customized strategy, instead of the `correct_tx` function, you will need the `_find_criteria` function. 

```python
>>> import polars as pl 
>>> adata_obs = pl.from_pandas(m.adata.obs["cell_type"], include_index=True)  
>>> m._find_criteria(adata_obs=adata_obs,
                    reassign_threshold_grid=reassign_threshold_grid,
                    remove_threshold_grid=remove_threshold_grid)
```

Once the search is done, you can inspect all results via 

```python
>>> m.criterion_df
```

This dataframe records the loss, thresholds, as well as the number of removed transcripts. Once you have decided on a certain combination or if you already know what criteria to use, you can supply MisTIC with single numbers. For example: 

```python
>>> m.correct_tx(reassign_threshold_grid=0.5,
                remove_threshold_grid=0)
```

5. Cell reclustering (optional)

Once the reassignment/removal threshold has been chosen, the users can "reclassify" the cells using the trained classification head which is just a linear logistic regression.

* Note that cell type classification is NOT the main purpose of MisTIC and the classification is only intended for providing some guidance to transcript reassignment during training. Therefore, we do not guarantee the accuracy of the classification head. 

Nevertheless, if the users are curious about how the cell types change before and after the correction, we do provide a function for this purpose: 

```python
>>> m.recluster()
```

More sophisticated behaviors can be specified by tweaking the parameters: 

* If you do not want deterministic results, you can use these parameters: 
    1. `temperature`: The higher the `temperature` the more uniform the sampling would be. The default 0.0 will also give the one with the highest logit.  
    2. `top_k`: If the value is not `None`, the `k` highest logit outputs will be retained whereas others will be set to `-Inf`
    * Combining these two allows sampling from `top_k` cell types. 
    * With this stochastic behavior, MisTIC allows the users to give multiple tries. Each run of `recluster` will add several columns into the `m.adata.obs` dataframe. If you want to overwrite previous runs, set `overwrite_previous_trials=True`. 
    * You can also update the original leiden results by setting `update_leiden=True`. Note that the original cell type annotation will not be changed. 


6. Model saving 

To save the model, simply do the following: 

```python
>>> m.save_model(dir_name="PATH/TO/DIRECTORY",
                    model_name="mistic",
                    save_correction_result=True)
```

This will save PyTorch model `mistic.pt` along with some meta information `mistic_meta.json`. In addition, by specifying `save_correction_result=True` the transcripts that will be reassigned/removed will be save as `mistic_tx_to_reassign.parquet`/`mistic_tx_to_remove.parquet`. 

This `.parquet` file contains a dataframe with four columns `molecule_id`, `from_cell_id`, `to_cell_id`, and `gene`. 

The ids contained in the `molecule_id` column correspond to the row numbers of the original `detected_transcripts` file. Therefore, `tx_0` corresponds to the first record in the `detected_transcripts` file.

`mistic_criterion_df.csv` contains the loss computed based on various combinations of thresholds. 

If you are also interested in the computed reassigning probabilities, you can assess them via  

```python
>>> m.tx_reassign_info
```

We do not provide a function to save this polars dataframe as it could be large. However, recomputing it would not take too long. 

7. Model loading 

Loading the saved model could be useful if you turned off the program but wanted to take a closer look at the results later on. As previously stated, since not all information is saved with the model, you will need to import the data again.

```python
>>> from MisTIC.mistic_class import mistic
>>> # Check and specify the column names!
>>> m = mistic("MAKE/SURE/TO/CHECK/COLUMN/NAMES!!!",
            model_device="cpu")
>>> # Load the model 
>>> m.load_model(dir_name="PATH/TO/DIRECTORY",
                    model_name="mistic",)
>>> # cell_by_gene_counts is optional
>>> cell_by_gene_counts = "PATH/TO/COUNTS"
>>> detected_transcripts = 'PATH/TO/TX'
>>> cell_metadata = 'PATH/TO/META'
>>> cell_boundary_polygons = "PATH/TO/POLYGONS"
>>> m.import_data(cell_by_gene_counts=cell_by_gene_counts,
                    cell_metadata=cell_metadata,
                    cell_boundary_polygons=cell_boundary_polygons,
                    detected_transcripts=detected_transcripts)
>>> m.patchfy_data()
>>> # Recompute the probabilities 
>>> m.compute_reassign_probs()
```

That's it~

### Command-Line Interface (CLI)

The arguments for CLI is almost identical to those in the interactive Python with only `cell_centroid_x_col` and `cell_centroid_y_col` being amalgamated into `--cell_centroid_x_y_col` and `tx_x_col` and `tx_y_col` into `--tx_x_y_col`.


```shell
python -m MisTIC --cell_centroid_x_y_col center_x center_y 
                --tx_x_y_col global_x global_y 
                --gene_col gene
                --celltype_col cell_type
                --cell_metadata PATH/TO/META
                --cell_boundary_polygons PATH/TO/POLYGONS
                --detected_transcripts PATH/TO/TX
                --cell_by_gene_counts PATH/TO/COUNTS
                --dir_name .
                --model_name mistic
```

* Note that the CLI mode of MisTIC DOES NOT allow the users to perform reclustering. 


## Citation :page_with_curl:

If you use MisTIC in your SRT data analysis workflow, citing [our paper](https://google.com) is appreciated:

```
@article{

}
```


[pytorch]: https://pytorch.org
[scanpy]: http://scanpy.readthedocs.io/
[readthedoc]: https://about.readthedocs.com/