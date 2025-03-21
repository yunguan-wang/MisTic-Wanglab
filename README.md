![Logo](/assets/logo.png)

# MisC
> A probabilistic model for correcting mis-assigned transcripts due to cell segmentation error in imaging-based spatial transcriptomics. It builds on top of [PyTorch] and [scanpy].

![forthebadge](/assets/spatial-transcriptomics.svg)
![forthebadge](/assets/scanpy-pytorch.svg)

## Installation :computer:

### Create a virtual environment :snake:

So far, we have only tested the software on Python 3.9 and 3.10.

```shell
conda create -n misc python=3.9
conda activate misc 
```

or 

```shell
conda create -n misc python=3.10
conda activate misc 
```

### Build the package

#### pip 

For a stable version of MisC, you can download and install the package via 

```shell
pip install txMisC
```

#### Local 

The latest version of MisC will be hosted on GitHub where we constantly update features of the package. To use the latest version of MisC, you will need to build the package loacally.

First, you need to clone the repo to a local directory, say `./awesome_repos` and `cd` to that folder. 

Now, you should have a `MisC` folder under the `awesome_repos` directory. Run the following to build the package.

```shell 
cd ./MisC
python setup.py sdist bdist_wheel
```

You should see a `dist` folder now which contains the wheel file you will need for installing the package. 

```shell
cd ./dist
pip install ./txMisC-0.0.1-py3-none-any.whl
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
This assumes that you are using Jupyter notebook to run MisC.

1. Object instantiation and data importing 
```python
>>> from MisC.misc_class import misc
>>> # Check and specify the column names!
>>> m = misc("MAKE/SURE/TO/CHECK/COLUMN/NAMES!!!",
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

This will not only import the data into the `misc` object, but also perform data curation and generate features necessary for model training and transcript reassignment. 

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

3. Transcript reassignment

We allow users to specify either a threshold within [0,1] or 'auto' for reassigning transcripts. The default is 0.5. 

```python
>>> m.compute_reassign_probs()
>>> m.reassign_tx(criterion=0.5)
>>> m.save_model(dir_name="PATH/TO/DIRECTORY",
                    model_name="misc",
                    save_reassigning_result=True)
```

This will save PyTorch model `misc.pt` along with some meta information `misc_meta.json`. In addition, by specifying `save_reassigning_result=True` the transcripts that will be reassigned according to the specified `threshold` will be save as `misc_tx_to_reassign.parquet`. 

This `.parquet` file contains a dataframe with four columns `molecule_id`, `from_cell_id`, `to_cell_id`, and `gene`. 

The ids contained in the `molecule_id` column correspond to the row numbers of the original `detected_transcripts` file. Therefore, `tx_0` corresponds to the first record in the `detected_transcripts` file.

If you are also interested in the computed reassigning probabilities, you can assess them via  

```python
>>> m.tx_reassign_info
```

We do not provide a function to save this polars dataframe as it could be large. However, recomputing it would not take too long. 

4. Model loading 

Loading the saved model could be useful if you turned off the program but wanted to take a closer look at the results later on. As previously stated, since not all information is saved with the model, you will need to import the data again.

```python
>>> from MisC.misc_class import misc
>>> # Check and specify the column names!
>>> m = misc("MAKE/SURE/TO/CHECK/COLUMN/NAMES!!!",
            model_device="cpu")
>>> # Load the model 
>>> m.load_model(dir_name="PATH/TO/DIRECTORY",
                    model_name="misc",)
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

By default, MisC will save the model to the current directory and the name of the model will be `misc`.

```shell
python -m MisC --cell_centroid_x_y_col x y 
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