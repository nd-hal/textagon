import pandas as pd
from textagon.textagon import Textagon

#from textagon.tGBS import tGBS
# WHere is the tGBS file???

import streamlit as st

st.title("Textagon - ACL System Demonstration")

textInput = st.text_area(
    "Please enter your sentence in the box below, and then press Ctrl+Enter...",
    value="Hypotension. Massive headaches, bp was still on the low side."
)

d = {"corpus":[textInput], "classLabels": [1]}
df = pd.DataFrame(data=d)

tgon = Textagon(
    inputFile=df, 
    outputFileName="aclSample"
)

tgon.RunFeatureConstruction()
tgon.RunPostFeatureConstruction()


import zipfile
import os

# Specify the path to the zip file
zip_file_path = './output/aclSample_representations.zip'

# Specify the directory to extract files to
extract_to_directory = './output/aclSample_representations'

# Ensure the directory exists
os.makedirs(extract_to_directory, exist_ok=True)

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents
    zip_ref.extractall(extract_to_directory)

print(f"Files extracted to {extract_to_directory}")


# Load the representations and present them to user in a table

repFiles = os.listdir(extract_to_directory)

lexica = []
outputs = []

for f in repFiles:
    with open(f"{extract_to_directory}/{f}", 'r') as infile:
        lexiconName = f.split("_")[2]
        lexiconName = lexiconName[:-4]
        repLines = infile.readlines()
        for l in repLines:
            lexica.append(lexiconName)
            outputs.append(l)

outputDF = {"Lexicon": lexica, "Representation": outputs}
outDF = pd.DataFrame(data=outputDF)

st.dataframe(outDF, hide_index=True)



