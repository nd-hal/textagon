# ------------
# Installation 
# ------------

# Package versions needed (execution will stop via a check; will add requirements.txt in the future):
- wn 0.0.23

# For SpaCy, run the following to get the 'en_core_web_sm' model ('en' in SpaCy 2.3.5):
python -m spacy download en_core_web_sm

# For the spellchecker (which defaults to aspell):
- MacOS: brew install enchant
- Windows: pyenchant includes hunspell out of the box
- Linux: install libenchant via package manager
- For general notes, see: https://pyenchant.github.io/pyenchant/install.html

# For NLTK dependencies (e.g., list of vader_lexicon), use the provided script after installing NLTK:
- python install-dependencies.py

# ----------------
# Running Textagon 
# ----------------

- For a demo or to test the installation, use:
    python cgi-bin/processText-serverModel.py
- From the main folder, for a file called 'dvd.txt' use:
    python cgi-bin/processText-serverModel.py upload/dvd.txt output 0 0 4 3 upload/FRNLexicons_v4.zip 1 5 bB 0 1 0 3 1 1 1 1 upload/exclusions.txt full > logs/output.log
- Change the name of the output and log file from 'output' and 'output.log' to whatever you prefer, respectively.
- To see on-screen output, remove the pipe to the log (i.e., delete ' > logs/output.log').
- In addition, for a full run (feature + matrix), the final argument should be 'full' as shown above. To run just one stage, replace 'full' with either 'feature' or 'matrix' depending on what you need.
- Beyond 'full', 'feature' and 'matrix', you can also use 'representation' mode to run 'feature' and only the relevant parts of 'matrix' to produce a representation zip file. Alternatively, use 'featuretorep' after performing a 'feature' run, use 'representation' to generate the associated zip file.
- In terms of settings, the key aspects are in the first sequence after output: 0 0 4 3. The 4 here is the n-gram length, while 3 is the maximum number of cores to use.
- The format of the command arguments will be improved in the next release and a help command will be added.
- See script.sh for a working command example (make sure to use chmod +x to run it). Similarly, script.bat provides a Windows example, but also includes a way to do benchmarking via mprof (pip install memory-profiler).

# ----
# Misc
# ----

- In the examples, exclusions.txt is always used, which contains a number of drug names to be ignored by the spellchecker (failing to exclude these will lead to a variety of innaccurate spell checker "fixes").