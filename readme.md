# Omni
#### Code for 'Omni-geometry representation learning and Large Language Models for Geospatial Entity Resolution'

Omni: a deep neural framework that performs Geospatial Entity Resolution (ER) on diverse-geometry databases. 

This project presents the code for the Omni framework. For the LLM based approach proposed in the paper, please check this [repo](https://github.com/Kalana777/LLMs-for-geospatial-ER).


### Install required packages
```
pip install -r requirements.txt
```

### Download Data
Omni is tested with 4 geospatial ER datasets. Download the New Zealand Entity Resolution dataset (NZER) 
[here](https://figshare.com/s/e0e0481d62a3e411178b) and copy to data directory. 


The complex geometry enhanced, pre-processed versions of the third-party geospatial ER datasets: GeoD, SGN & GTMD, have 
been made available [here](https://figshare.com/s/7858aa81a88b2347d09d). Unzip the content into the data directory.

### Model Training

#### Command for training Omni or Omni-small model:
```
python main.py \
  -d NZER \
  -r norse \
  -m omni \
  --run_att_aff \
  --attributes name type \
  --save_best_model \
  --save_path saved_models \
  --log_wandb \
```

* ``-d``: Specify the dataset to train. Possible values: ``NZER``, ``GTMD``, ``SGN``, ``GEOD_OSM_FSQ``, ``GEOD_OSM_YELP``.
* ``-r``: City or Region (NZER: ``auck``, ``hope``, ``norse``, ``north``, ``palm``. GTMD: ``mel``, ``sea``, ``sin``, ``tor``. SGN: ``swiss``. GEOD_OSM_FSQ/GEOD_OSM_YELP: ``sin``, ``edi``, ``tor``, ``pit``).
* ``-m``: Model. Choose between the two variants: ``omni`` and ``omnismall``.
* ``--run_att_aff``: Set flag to run attribute affinity.
* ``--attributes``: List of attributes to compare with attribute affinity. For example, ``name type`` for nzer..
* ``--save_best_model``: Set flag to save best model.
* ``--save_path``: Directory to save best model. Default is ``saved_models``.
* ``--log_wandb``: Set flag to log training to wandb. Requires WandB account
<br/>

#### When using our code, NZER dataset or the geometry enhanced versions of the third-party datasets we provide, please cite our paper
```Paper under review at VLDB 2025```