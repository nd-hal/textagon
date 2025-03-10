import pandas as pd
from textagon.textagon import Textagon

### Test cases ###

df = pd.read_csv('../examples/dvd.txt', sep='\t', header=None, names=["classLabels", "corpus"])


tgon = Textagon(
    df, "dvd", 0, 0, 4, 3, "Lexicons_v5.zip", 
    1, 5, "bB", 0, 1, 0, 3, 1, 1, 1, 1, 1, "upload/exclusions.txt", "full",
    False
)

tgon.RunFeatureConstruction()
tgon.RunPostFeatureConstruction()
