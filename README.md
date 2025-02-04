![Logo](/assets/MisC.png)

# MisC
MisC is a probabilistic model for correcting mis-assigned transcripts due to cell segmentation error. It builds on top of [PyTorch] and [scanpy].

## Installation 

1. Create a virtual environment

We have only tested the software on Python 3.9 and 3.10.

```shell
conda create -n misc python=3.9
conda activate misc 
```

or 

```shell
conda create -n misc python=3.9
conda activate misc 
```

2. For pip (currently not working)

```shell
pip install txMisC
```

Alternatively, you can use conda/micromamba to install all dependencies. Then use pip to just install MisC with `no-deps` option.
```shell
micromamba install python=3.10 pytorch=1.13 geopandas notebook ipykernel tqdm ipywidgets pyarrow scanpy pydeseq2
pip install --no-deps .
```

3. Build locally 

To build the package loacally, you first need to clone the repo to a local directory, say `./awesome_repos` and `cd` to that folder. 

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

So far, we have only tested the package on python `3.9` and `3.10` with pytorch < `2.0`. 

Note that if you install pytorch >= `2.0`, it will throw an error. 

Also make sure the numpy version is < `2.0` and pydeseq2 is >=`0.4.6` and <`0.5`. 

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

## Quick start  

### Interactive Python 
This assumes that you are using Jupyter notebook to run MisC.

1. Instantiate the object and import data
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
>>> m.initialize_parameters()
>>> m.training_loop(n_epochs=5)
```

3. Transcript reassignment

We allow users to specify various criteria/cutoff for reassigning transcripts. The default is 0.5. 

```python
>>> m.compute_reassign_probs()
>>> m.trial_reassign_tx(criteria={"threshold": 0.5})
>>> m.save_model(dir_name="PATH/TO/DIRECTORY",
                    model_name="misc",
                    save_reassigning_result=True,
                    selected_criterion="threshold")
```

This will save PyTorch model `misc.pt` along with some meta information `misc_meta.json`. In addition, by specifying `save_reassigning_result=True` along with the `selected_criterion` the transcripts that will be reassigned according to the specified `threshold` will be save as `misc_tx_to_reassign.parquet`. 

This `.parquet` file contains a dataframe with four columns `molecule_id`, `original_cell_id`, `reassigned_cell_id`, and `gene`. 

The ids contained in the `molecule_id` column correspond to the row numbers of the original `detected_transcripts` file. Therefore, `tx_0` corresponds to the first record in the `detected_transcripts` file.

If you are also interested in the computed reassigning probabilities, you can assess them via 

```python
>>> m.intf_tx
```

We do not provide a function to save this polars dataframe as it could be large. However, recomputing it would not take too long. 


### Command-Line Interface (CLI)

Coming soon. 

## Documentation 
Writing software documentation is like cleaning your room—everyone agrees it’s important, but no one wants to do it until they can’t find something.

## Citation
Citing a paper is like sending a thank-you note—it’s polite, necessary, and half the time you’re just copying what someone else did.


[pytorch]: https://pytorch.org
[scanpy]: http://scanpy.readthedocs.io/