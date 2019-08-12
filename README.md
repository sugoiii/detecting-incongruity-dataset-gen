# detecting-incongruity-dataset-gen
English dataset generation code for [detecting-incongruity](https://github.com/david-yoon/detecting-incongruity) based on [NELA2017 Dataset](https://github.com/BenjaminDHorne/NELA2017-Dataset-v1)

Generation method from following paper: 
**Detecting Incongruity Between News Headline and Body Text via a Deep Hierarchical Encoder**, AAAI-19, [paper](https://arxiv.org/abs/1811.07066)

### Requirements

- Python3
- Python libraries
  - Pandas
  - NLTK



### Usage

- Read directly from NELA2017 Dataset (unzipped)

```
$python3 dataset_creation.py --nela_path [PATH_TO_UNZIPPED_NELA_FOLDER] --output_dir ./output/ 
```

- Read arbitrary article file 
  - Article file must be csv formatted with headline and body at each row without header.

```
$python3 dataset_creation.py --input_path sample.csv --output_dir ./output/ 
```

