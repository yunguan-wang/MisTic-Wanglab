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

```shell
cd ./dist
pip install ./txMisTIC-0.0.1-py3-none-any.whl
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

## Quick start :fast_forward:

The comprehensive documentation is hosted [here](https://google.com) with the support of [readthedoc]. 

### Interactive Python 
This assumes that you are using Jupyter notebook to run MisTIC.

1. Object instantiation and data importing 
```python
>>> from MisTIC.mistic_class import mistic
>>> # Check and specify the column names!
>>> m = mistic("MAKE/SURE/TO/CHECK/COLUMN/NAMES!!!",
            model_device="cpu")
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

Note that the function will create a `molecule_id` column for the `detected_transcripts` file. The first record will be `tx_0`, the second will be `tx_1`, etc..

2. Model training

```python
>>> # Generate minibatches for SGD
>>> m.patchfy_data()
>>> # Modle training 
>>> m.initialize_parameters()
>>> m.training_loop(n_epochs=20)
```

Note that the algorithm will not necessarily train `20` epochs due to the implementation of an early stopping mechanism. So no need to panic. 

3. Transcript correction

We allow users to specify two grids over which to search for the best combination of threshods.

`reassign_threshold_grid` should be an array/list of numbers within [0, 1]. Transcripts with `reassign_probs` greater than the threshold will be reassigned. 

`remove_threshold_grid` should be an array/list of numbers within [0, 1]. For transcripts that are NOT reassigned, a 
separate threshold will be generated. For example, if the `reassign_threshold=0.3` and the `remove_threshold=1/3`, the actual threshold will be `0.3*(1-1/3)=0.2`. Transcripts with `reassign_probs` greater than the threshold will be removed. If removal is not desired, simply set `remove_threshold_grid=0`.

```python
>>> m.compute_reassign_probs()
>>> m.correct_tx(reassign_threshold_grid=np.arange(start=0.1, stop=0.5, step=0.1),
                remove_threshold_grid=np.linspace(start=0, stop=1, num=10))
```

4. Cell reclustering (optional)

Once the reassignment/removal threshold has been chosen, the users can "reclassify" the cells using the trained classification head which is just a linear logistic regression.

Note that cell type classification is NOT the main purpose of MisTIC and the classification is only intended for providing some guidance to transcript reassignment during training. Therefore, we do not guarantee the accuracy of the classification head. 

Nevertheless, if the users are curious about how the cell types change before and after the correction, we do provide a function for this purpose: 

```python
>>> m.recluster()
```

5. Model saving 

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

6. Model loading 

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

By default, MisTIC will save the model to the current directory and the name of the model will be `mistic`.

```shell
python -m MisTIC --cell_centroid_x_y_col x y 
                --tx_x_y_col X Y 
                --cell_metadata PATH/TO/META
                --cell_boundary_polygons PATH/TO/POLYGONS
                --detected_transcripts PATH/TO/TX
                --cell_by_gene_counts PATH/TO/COUNTS
```

## Citation :page_with_curl:



[pytorch]: https://pytorch.org
[scanpy]: http://scanpy.readthedocs.io/
[readthedoc]: https://about.readthedocs.com/