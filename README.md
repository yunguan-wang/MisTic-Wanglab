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

```python
>>> from MisC.misc_class import misc
>>> # Check and specify the column names!
>>> m = misc("MAKE/SURE/TO/CHECK/COLUMN/NAMES!!!")
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
>>> m.initialize_parameters()
>>> m.training_loop(n_epochs=5)
>>> m.trial_reassign_tx(criteria={"threshold": 0.5})
>>> m.save_model(dir_name="PATH/TO/DIRECTORY",
                    model_name="misc",
                    save_reassigning_result=True)
```

This will save PyTorch model `misc.pt` along with some meta information `misc_meta.json`. In addition, by specifying `save_reassigning_result=True`, the transcripts that will be reassigned according to the specified `threshold` will be save as `misc_tx_to_reassign_dict.json`. 

This `.json` file contains potentially multiple dataframes each corresponding to one of the criteria. Each dataframe will have three columns `molecule_id`, `cell_id`, and `neighbor_cell_id`. To actually reassign those transcripts, you can run 

```shell
>>> m.final_reassign_tx(selected_criterion='threshold')
```

or if you do not want to work with the object

```shell
>>> from MisC.utility import make_reassignment_tx_metadata, JSONEncoder
>>> # Read in your detected transcripts and curated it
>>> tx_to_reassign_dict = json.load(open("PATH/TO/DICT"))
>>> for k in tx_to_reassign_dict:
        tx_to_reassign_dict[k] = pd.read_json(tx_to_reassign_dict[k])
>>> updated_tx_meta = make_reassignment_tx_metadata(
                tx_to_reassign=tx_to_reassign["CRITERION_YOU_WANG"],
                tx_metadata=tx_metadata)
```

### Command-Line Interface (CLI)

Coming soon. 

## Documentation 
Writing software documentation is like cleaning your room—everyone agrees it’s important, but no one wants to do it until they can’t find something.

## Citation
Citing a paper is like sending a thank-you note—it’s polite, necessary, and half the time you’re just copying what someone else did.


[pytorch]: https://pytorch.org
[scanpy]: http://scanpy.readthedocs.io/