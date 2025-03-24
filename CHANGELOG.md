# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [0.1.14] - 2021-03-30

- update WNAffect code to avoid Python 3.9.2 warnings regarding 'is' syntax
- made all misspelling representations parallelizable by ensuring MISSPELLING tags are repeated when replacements are multi-gram
- suppress warnings from BeautifulSoup4 when text items consist exclusively of a URL (although this is likely not good input data)
- fix an issue with misspellings using an NA replacement when no suitable replacement suggestion was found

## [0.1.13] - 2021-03-04

- add more timing output and improve formatting
- massive speedup of column sums calculation vs. v11
- increased CSV chunk size slightly

## [0.1.12] - 2021-03-04

- update code to latest dev version to ensure all v9 improvements were included (various v9 improvements were missing from v10 and v11 due to inconsistent sources)

## [0.1.11] - 2021-03-04

- add column sums to the FRN key output
- ensure the batch/command line examples are up-to-date

## [0.1.10] - 2021-03-03

- fixes the AFFECT representation output by computing the Penn Treebank tags via NLTK (as opposed to relying on the OntoNotes 5 version of the Penn Treebank tags provided by spaCy, which are incompatible)

## [0.1.9] - 2021-02-22

- improves max cores handling when using mapply
- improve MISPELLING feature consistency
- speed up spaCy usage
- improve NER handling (e.g., handle multiple words such as $1 billion) and fix output issue with non-NER words
- fix case consistency in certain features (e.g., Word_POS)
- add a boundaries feature (Boundaries): 'S' is a sentence boundary; 'D' is a document boundary; '-' is anything else; note that the document boundary sequence is '|||'
- improve batch/command line run examples
- minor code cleanup

## [0.1.8] - 2021-02-16

- fix a legomena bug introduced in v6 related to mapply

## [0.1.7] - 2021-02-15

- speed up the NZV removal
- fix the script.bat example

## [0.1.6] - 2021-02-15

- write pickle during all modes that work on 'feature'
- parallelize legomena tagging
- avoid writing out raw text during spellchecker logging
- display which feature vector is being worked on in the log and show time elapsed

## [0.1.5] - 2021-02-10

- fix lingering Eastern timezone issue
- add 'representation' mode to run feature and generate a representation zip file
- add 'featuretorep' mode to convert a 'feature' run to a representation zip file

## [0.1.4] - 2021-02-10

- report Enchant version
- improve raw data reading (re: newline characters)
- add debug info for raw data read errors

## [0.1.3] - 2021-02-09

- update for SpaCy 3.0.x compatibility (note: using 'en_core_web_sm' model in place of 'en' shortcut)
- detect base directory automatically instead of hardcoding
- report PyEnchant version
- infer local timezone instead of assuming EST (note: tzlocal package added as a new dependency)

## [0.1.2] - 2021-02-04

- use parallelized custom hashing approach for feature matrix deduping instead of slower NumPy approach

## [0.1.1]

- initial development version