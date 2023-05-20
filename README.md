![Python 3.6+](https://img.shields.io/badge/python-3.6%2B-blue.svg) ![License: PSF](https://img.shields.io/badge/License-PSF-blue.svg)

# Textagon

Textagon is a powerful tool for text data analysis, providing a means to visualize parallel representations of your data and gain insight into the impact of various lexicons on two classes of text data.

# Installation

### Initial Setup
```
pip install textagon # Install the package using pip
textagon_post_install # Run the post-installation script
```

### Upgrading Textagon
```
pip install --upgrade textagon # Upgrade the package using pip
textagon_post_install # Run the post-installation script
```

### Updating the Package for Development (Only for developer)
```
# On Windows
rmdir /s /q ./build ./dist ./textagon.egg-info

# On Linux
rm -rf ./build ./dist ./textagon.egg-info

# Then, run
python setup.py sdist bdist_wheel
twine upload dist/*
```

# Core Functions

- **ReadAllLexicons**: Processes the lexicon file.
- **SanityCheck**: Ensures the data is valid and well-organized.
- **ReadRawText**: Invokes the SanityCheck function to return pure text data and class labels.
- **setSpellChecking**: Sets up the spellchecker. Note: If you wish to use a custom exclusion file, you must specify your exclusion file in the downstream function call.
- **TextToFeatures**: [Add description here]
