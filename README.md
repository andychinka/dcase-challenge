# dcase-challenge
Code for experiencing challenges in DCASE across different years and models

## Getting Started
### Prerequisites

TODO

```
pip install TODO
```

### Installing/Running the Code

1. Preprocess Data

```bash
python preprocess.py -db_path=<folder> -feature_folder=<folder>
```

2. Build Model

```
python train.py -db_path=<folder> -feature_folder=<folder> -model_save_fp=<filepath> -task=<task name> -model=<model name>
```
Example: Run 2018 Task 1A with baseline model
```bash
python train.py -db_path=<folder> - feature_folder=<folder> -task=task1a-2018 -model=baseline
```



## Authors/Contact


* **Cheung Chin Ka**- MAY 2020 - XXXXX MASTERS


## Acknowledgments

* A multi-device dataset for urban acoustic scene classification https://arxiv.org/abs/1807.09840.

