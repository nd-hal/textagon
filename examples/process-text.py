import pandas as pd
from textagon.textagon import Textagon
from textagon.AFRN import AFRN

### Test cases ###

df = pd.read_csv('../examples/dvd.txt', sep='\t', header=None, names=["classLabels", "corpus"])


#tgon = Textagon(
#    df, "dvd", 0, 0, 4, 3, "Lexicons_v5.zip", 
#    1, 5, "bB", 0, 1, 0, 3, 1, 1, 1, 1, 1, "upload/exclusions.txt", "full",
#    False
#)

#tgon.RunFeatureConstruction()
#tgon.RunPostFeatureConstruction()


featuresFile = './output/dvd_key.txt'
trainFile = './output/dvd.csv'
weightFile = './output/dvd_weights.txt'


afrn=AFRN(
	featuresFile=featuresFile,
	trainFile=trainFile,
	weightFile=weightFile
)

afrn.ReadFeatures()
afrn.ReadTrain()
afrn.ReadSentiScores()
afrn.ReadLex()
afrn.AssignTrainWeights()
afrn.AssignSemanticWeights()
afrn.RunSubsumptions()
afrn.RunCCSubsumptions()
afrn.outLogSub.close()
afrn.RunParallels()
afrn.outLogPar.close()
afrn.OutputRankings()

