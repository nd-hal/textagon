![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg) ![License: PSF](https://img.shields.io/badge/License-MIT-blue.svg)




# Textagon

Textagon is a powerful tool for text data analysis, providing a means to visualize parallel representations of your data and gain insight into the impact of various lexicons on two classes of text data. 
- **Parallel Representations**
- **Graph-based Feature Weighting**

# Run DEMO

https://colab.research.google.com/drive/115P0Psl49CQIp9InJXJxQKMoN9NmfDvt


# Installation

### Initial Setup
```
pip install textagon 
```

### Upgrading Textagon
```
pip install --upgrade textagon 
```


## Additional Considerations

### Installation 

- Package versions needed (execution will stop via a check; will add requirements.txt in the future):
    - wn 0.0.23
- For SpaCy, run the following to get the 'en_core_web_sm' model ('en' in SpaCy 2.3.5):

```python
python -m spacy download en_core_web_sm
```

- For the spellchecker (which defaults to aspell):
    - MacOS: brew install enchant
    - Windows: pyenchant includes hunspell out of the box
    - Linux: install libenchant via package manager
    - For general notes, see: https://pyenchant.github.io/pyenchant/install.html


### Running Textagon 

```python
import pandas as pd
from textagon.textagon import Textagon
from textagon.AFRN import AFRN

### Test cases ###

df = pd.read_csv('../examples/dvd.txt', sep='\t', header=None, names=["classLabels", "corpus"])

tgon = Textagon(
    df, "dvd", 0, 0, 4, 3, "Lexicons_v5.zip", 
    1, 5, "bB", 0, 1, 0, 3, 1, 1, 1, 1, 1, "upload/exclusions.txt", "full",
    False
)

tgon.RunFeatureConstruction()
tgon.RunPostFeatureConstruction()
```
