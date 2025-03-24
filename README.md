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

```python
import pandas as pd
from textagon.textagon import Textagon
from textagon.AFRN import AFRN

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
