![Logo](/assets/MisC.png)

# MisC
> Reassign transcripts 

## Installation 

This is only for internal usage. We will publish the package on pypi later on. But for now, we will just build the package locally. 

1. Clone the repo to a local directory, say `./awesome_repos` and `cd` to that folder. 

2. Now, you should have a `MisC` folder under the `awesome_repos` directory. 
```shell 
cd ./MisC
```

3. Build the package 
```shell
python setup.py sdist bdist_wheel
```
You should see a `dist` folder now which contains the wheel file you will need for installing the package. 

4. To install the package
```shell
conda create -n misc python=3.9
conda activate misc 
cd ./dist
pip install ./txMisC-0.0.1-py3-none-any.whl
```
5. Alternatively, you can use conda/micromamba to install all dependencies. Then use pip to just install MisC with `no-deps` option.
```shell
micromamba install python=3.10 pytorch=1.13 geopandas notebook ipykernel tqdm ipywidgets pyarrow scanpy pydeseq2
pip install --no-deps .
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
- jupyter
- ipywidgets

## Examples 

### Interactive Python 
This assumes that you are using Jupyter notebook to run MisC

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
>>> m.training_loop(1)
>>> m.trial_reassign_tx(criteria={"threshold": 0.5})
>>> m.final_reassign_tx(selected_criterion="threshold")
>>> m.recluster()
```


### Command-Line Interface (CLI)


## Documentation 
Writing software documentation is like cleaning your room—everyone agrees it’s important, but no one wants to do it until they can’t find something.

## Citation
Citing a paper is like sending a thank-you note—it’s polite, necessary, and half the time you’re just copying what someone else did.
