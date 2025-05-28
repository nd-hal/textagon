![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg) ![License: PSF](https://img.shields.io/badge/License-MIT-blue.svg)




# Textagon

Textagon is a powerful tool for text data analysis, providing a means to visualize parallel representations of your data and gain insight into the impact of various lexicons on two classes of text data. 
- **Parallel Representations**
- **Graph-based Feature Weighting**



# Installation


## Prereqs

### Installation 

- Package versions needed (execution will stop via a check; will add requirements.txt in the future):
    - wn 0.0.23

- For the spellchecker (which defaults to aspell):
    - MacOS: brew install enchant
    - Windows: pyenchant includes hunspell out of the box
    - Linux: install libenchant via package manager
    - For general notes, see: https://pyenchant.github.io/pyenchant/install.html


### Initial Setup
```
pip install textagon 
```

### Upgrading Textagon
```
pip install --upgrade textagon 
```


# Running Textagon 

1. Generate representations

```python
import pandas as pd
from textagon.textagon import Textagon
from textagon.tGBS import tGBS

### Test cases ###

df = pd.read_csv(
    './sample_data/dvd.txt', 
    sep='\t', 
    header=None, 
    names=["classLabels", "corpus"]
)

tgon = Textagon(
    inputFile=df, 
    outputFileName="dvd"
)

tgon.RunFeatureConstruction()
tgon.RunPostFeatureConstruction()

```

2. Unzip stored representations

```python
import zipfile
import os

# Specify the path to the zip file
zip_file_path = './output/distress_representations.zip'

# Specify the directory to extract files to
extract_to_directory = './output/distress_representations'

# Ensure the directory exists
os.makedirs(extract_to_directory, exist_ok=True)

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents
    zip_ref.extractall(extract_to_directory)

print(f"Files extracted to {extract_to_directory}")
```

3. Score and rank representations with tGBS.


```python
featuresFile = './output/distress_key.txt'
trainFile = './output/distress.csv'
weightFile = './output/distress_weights.txt'


ranker=tGBS(
	featuresFile=featuresFile,
	trainFile=trainFile,
	weightFile=weightFile
)

ranker.RankRepresentations()

```


# Citation

If you use Textagon in your work, please cite the paper (forthcoming in ACL 2025):

```
@inproceedings{lalorTextagon,
  title = {Textagon: Boosting Language Models with Theory-guided Parallel Representations},
  author = {Lalor, John P and Qin, Ruiyang and Dobolyi, David and Abbasi, Ahmed},
  booktitle = {Proceedings of the 2025 Annual Meeting of the Association for Computational Linguistics},
  date = {2025}
}
```

