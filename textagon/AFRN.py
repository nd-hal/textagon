import os
import sys
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
import re


featuresFile = sys.argv[1]
trainFile = sys.argv[2]
weightFile = sys.argv[3]
numFeat = 0
foundNACt = 0
foundNA = False
numLexFile = 0
numCat = 0
maxposword = 0
numInst = 0
numClass = 0
semMax = 50000
maxhash = 50000
numLex = 0
flipposwordCt = np.zeros((27,27),dtype=int)
sentiscoresCt = np.zeros((27,27),dtype=int)
sentiscores = np.zeros((27,27,1,3), dtype=object)
lexFile = ["0"]*50
cat = ['0']*50
catN = [0]*50
classLabels = [0]*20
lexSentiScores = np.zeros(semMax, dtype="float")
trainWeight = []
trainWeightC = []
instLabels = []
matrix = []
featureIndex = []
featureStr = []
outLogSub = open("AFRNSubsumptionLogPy.txt", "w")
outLogPar = open("AFRNParallelLogPy.txt", "w")
thresh = 0.0000000001
subThresh = 0.05
corrThresh = 0.95
runLogs = True
featureCatStr = None

#semantic hashes
lex = np.zeros((semMax,semMax), dtype=object)
lexCt = np.zeros((semMax),dtype=int)
hashlex = np.zeros((27,27,semMax), dtype=object)
hashlexClust = np.zeros((27,27,semMax),dtype=int)
hashlexCt = np.zeros((27,27),dtype=int)
lexTags = np.zeros((semMax), dtype=object)

#hash arrays

def ReadFeatures():
	global featuresFile, numFeat, foundNACt, foundNA, numLexFile, numCat, maxposword, flipposword, flipposwordCt, lexFile, cat, featureIndex, featureStr, featureCatStr
	print("Loading features")
	featuresData = open(featuresFile, "r")
	line = featuresData.readline()

	#for testing...
	lCount1 = 0

	n = 0
	while line:
		#for testing...
		#if lCount1%10 == 0 and lCount1<200102:

		tokens = line[:-1].split("\t")
		#print(tokens[0],tokens[1])
		if tokens[0] == "NA" and tokens[1] == "NA-NA":
			foundNA = True #means we have class label and last X are vader
			foundNACt = foundNACt + 1 #figure out how many
		else:
			numFeat = numFeat + 1


		tokens2 = tokens[1].split("-")
		if len(tokens2) == 2 and tokens2[0] !="NA":

			#get N value
			n = int(tokens2[0])

			#get category string
			if len(tokens2[1]) > 7 and "LEXICON" in tokens2[1]:
				catStr="LEXICON"
				LF = tokens2[1][7:len(tokens2[1])]
				LFexists = False;

				for v in range(0, numLexFile):
					if lexFile[v] == LF:
							LFexists = True

				if LFexists == False:
					lexFile[numLexFile] = LF
					numLexFile = numLexFile + 1

			else:
				catStr = tokens2[1]

		else:
			n = 1
			catStr = tokens2[0]

		catFound = False
		for x in range(0, numCat):
			if catStr == cat[x]:
				catFound = True
				if n > catN[x]:
					catN[x] = n #increase max n for category if larger value found
				break

		if catFound == False:
			#add new category and current max n value for category
			cat[numCat] = catStr
			catN[numCat] = n
			print(str(numCat).strip("\n") + " " + str(catStr).strip("\n"))
			numCat = numCat + 1

		#handle flipposword
		if tokens[1] == "1-WORD&POS":
			tokens2 = re.split(" |\\|_\\|",tokens[0])
			flipWord = ""
			if len(tokens2) >= 2:
				flipWord = tokens2[1] + " " + tokens2[0]
				for f in range(3, len(tokens2), 2):
					flipWord = flipWord + " "+ tokens2[f] + " " + tokens2[f-1]

			if len(flipWord) >= 2:
				index = HashLetters(flipWord)
				flipposwordCt[index[0]][index[1]] = flipposwordCt[index[0]][index[1]] + 1
				if flipposwordCt[index[0]][index[1]] > maxposword:
					maxposword = flipposwordCt[index[0]][index[1]]
		#for testing...
		#if tokens[0] != "NA": lCount1+=1
		line = featuresData.readline()

	#ending first run
	print("Total categories found = ",numCat)
	print("Total features found = ",numFeat)
	print("Total lexicons = ",numLexFile)

	lCount = 0
	#initialize feature array and update hash array sizes
	featureIndex = np.zeros((numFeat, 3), dtype="int32") #status, catNum, and n
	featureStr = [""]*numFeat
	featureCatStr = [""]*numFeat
	flipposword = np.zeros((27,27,maxposword))
	flipposwordCt = np.zeros((27,27))

	featuresData.close()
	featuresData = open(featuresFile, "r")
	line = featuresData.readline()

	#for testing
	#lxCount = 0

	while line:
		#for testing...
		#if lxCount%10==0 and lxCount<200102:
		#	lCount = int(lxCount/10)

		tokens = line[:-1].split("\t");
		if tokens[0] != "NA" and tokens[1] != "NA-NA":
			featureIndex[lCount][0] = 1 #status column, 0 = discarded, 1 = retained/active
			
			#set letter indexes to 0 for now
			featureStr[lCount] = tokens[0]
			featureCatStr[lCount] = tokens[1]
			
			#need to store categories and n-values in second pass
			tokens2 = tokens[1].split("-")

			n = 0
			catStr = ""

			if len(tokens2) == 2 and tokens2[0] != "NA":
				#get N value
				n = int(tokens2[0])

				#get category string
				if len(tokens2[1]) > 7 and tokens2[1][0:7] == "LEXICON":
					catStr="LEXICON"
				else:
					catStr = tokens2[1]
			else:
				n = 1
				catStr = tokens2[0]

			for x in range(0, numCat):
				if catStr == cat[x]:
					featureIndex[lCount][1] = x
					break

			featureIndex[lCount][2] = n

			if tokens[1] == "1-WORD&POS":
				#handle flip - populate hash array
				tokens2 = re.split(" |\\|_\\|",tokens[0])
				flipWord = ""
				if len(tokens2) >=2:
					flipWord= str(tokens2[1]) + " " + str(tokens2[0])

					for f in range(3, len(tokens2), 2):
						flipWord = flipWord + " " + tokens2[f] + " " + tokens2[f-1]

				if len(flipWord) >=2:
					index = HashLetters(flipWord)
					flipposword[index[0]][index[1]][flipposwordCt[index[0]][index[1]]] = flipWord
					flipposwordCt[index[0]][index[1]] = flipposwordCt[index[0]][index[1]] + 1

		#if tokens[0] != "NA": lxCount = lxCount + 1
			lCount+=1
		## read next line
		line = featuresData.readline()

	featuresData.close()

def ReadTrain():
	global matrix, trainFile, numFeat, foundNACt, foundNA, numLexFile, numCat, maxposword, flipposwordCt, lexFile, cat, numInst, classLabels, numClass,trainWeight, trainWeightC, instLabels
	print("Loading training data")
	trainData = open(trainFile, "r")
	line = trainData.readline()
	while line:
		tokens = line[:-1].split(",")
		if tokens[0] != "Class":
			numInst = numInst + 1
			#check to see if class label already added to label array
			isNew = True
			thisClass = int(tokens[0])
			for a in range(0, numClass):
				if thisClass == classLabels[a]:
					isNew = False
					break

			if isNew:
				classLabels[numClass] = thisClass
				numClass = numClass + 1

		line = trainData.readline()

	trainData.close()

	print("Classes=", numClass, classLabels[0], classLabels[numClass-1],"Num Instances = ",numInst)
	ftNum = 0
	if foundNA:
		ftNum = len(tokens) - foundNACt
	else:
		ftNum = len(tokens) - 1

	if ftNum != numFeat:
		print("Number of features in Features file and Train file are different!!!", ftNum, numFeat)

	#for testing...
	ftNum = numFeat
	matrix = np.zeros((ftNum, numInst),dtype=int)
	instLabels = np.zeros(numInst)
	trainWeight = np.zeros(ftNum,dtype="float")
	trainWeightC = np.zeros((ftNum, 2))

	trainData = open(trainFile, "r")
	line = trainData.readline()
	lCount = 0
	while line:
		tokens = line.split(",")
		if tokens[0] != "Class":
			cIndex = int(tokens[0])
			for c in range(0, numClass):
				if cIndex == classLabels[c]:
					instLabels[lCount] = (int)(c)
					break

			if foundNA:
				#for testing...
				#for a in range(2, numFeat+2):
				#	matrix[a-1][lCount] = int(tokens[a])
				#for testing samples of all types
				#for a in range(1, 200102):
				#	if (a-1)%10==0:
				#		matrix[int((a-1.)/10.)][lCount] = int(tokens[a])
				#original correct
				for a in range(1, len(tokens) - foundNACt + 1):
					matrix[a-1][lCount] = int(tokens[a])
			else:
				for a in range(1, len(tokens)):
					matrix[a-1][lCount] = int(tokens[a])

			lCount = lCount + 1

		line = trainData.readline()
	trainData.close()

def AssignTrainWeights():
	global trainFile, numFeat, foundNACt, foundNA, numLexFile, numCat, maxposword, flipposwordCt, lexFile, cat, numInst, classLabels, numClass,trainWeight, trainWeightC, instLabels
	global matrix
	print("Assigning training weights")
	bestScore = 0

	#numFeat = numFeat - 1
	#numFeat = 20
	for b in range(0, numFeat):
		sumc = np.zeros(numClass, dtype="float")
		pc = np.zeros(numClass, dtype="float")
		wc = np.zeros(numClass, dtype="float")
		wcc = np.zeros((numClass, numClass), dtype="float")
		totSum = 0
		for a in range(0, numInst):
			#print(a, b)
			sumc[int(instLabels[a])] = sumc[int(instLabels[a])] + matrix[b][a]
			totSum = totSum + matrix[b][a]

		#adjust for measurement error in 0.2% of feature strings
		if totSum==0:
			totSum = 1

		for a in range(0, numClass):
			pc[a] = float(sumc[a] / totSum)

		for a in range(0, numClass):
			for c in range(0, numClass):
				if a != c:
					if pc[a] > 0 and pc[c] > 0:
						wcc[a][c] = pc[a] * math.log( float(pc[a]) / pc[c])
					else:
						wcc[a][c] = float(sumc[a]*0.1)
					wc[a]= float(wc[a]+wcc[a][c])
			wc[a]=float(wc[a]/(numClass-1))

		#print(featureStr[a],pc[0],pc[1],sumc[0],sumc[1],totSum,wc[0],wc[1])

		#identify best score for the feature and its best class
		maxC = 0
		maxCC = 0
		maxVal = 0
		maxValC = 0
		for a in range(0, numClass):
			if wc[a] > maxVal:
				maxVal = wc[a]
				maxC = a + 1
				for c in range(0, numClass):
					val = float(wcc[a][c])
					if val > maxValC:
						maxCC = c + 1
						maxValC = val


		trainWeight[b] = maxVal
		#if b<20: print(b,featureStr[b],trainWeight[b])
		trainWeightC[b][0] = int(maxC)
		trainWeightC[b][1] = int(maxCC)

def ReadSentiScores():
	global sentiscoresCt, sentiscores
	#we know the max hash value from prior testing...
	sentiMax = 4763

	sentiscores = np.zeros((27,27,int(sentiMax),3), dtype=object)
	sentiscoresCt = np.zeros((27,27), dtype="int32")
	sentiScoresData = open("sentiscores.txt").readlines()
	
	print("Loading sentiment scores",sentiMax)
	for row in sentiScoresData:
		tokens = re.split(",",row[:-1])
		#print(row[:-1],tokens,tokens[0],tokens[1],tokens[2])
		if len(tokens[0]) >= 2 and len(tokens) == 3:
			index = HashLetters(tokens[0])
			sentiscores[index[0]][index[1]][sentiscoresCt[index[0]][index[1]]][0] = tokens[0]
			sentiscores[index[0]][index[1]][sentiscoresCt[index[0]][index[1]]][1] = tokens[1]
			sentiscores[index[0]][index[1]][sentiscoresCt[index[0]][index[1]]][2] = str(abs(float(tokens[2]))) + ""
			
			#print(str(sentiscores[index[0]][index[1]][sentiscoresCt[index[0]][index[1]]][0]),str(sentiscores[index[0]][index[1]][sentiscoresCt[index[0]][index[1]]][1]),str(sentiscores[index[0]][index[1]][sentiscoresCt[index[0]][index[1]]][2]))
			sentiscoresCt[index[0]][index[1]] = sentiscoresCt[index[0]][index[1]] + 1

	#for testing
	#for z in range(0,4763):
	#	print(str(sentiscores[2][14][z][0]),str(sentiscores[2][14][z][1]),str(sentiscores[2][14][z][2]))

def ReadLex():
	global numLexFile, lexSentiScores, numLex, lexTags, hashlex, hashlexCt, hashlexClust, lexCt, lex
	print("Loading lexicons...")
	#tag index number and quantity
	numLex = 0
	#number of total lex items across tags
	totlex = 0

	for v in range(0, numLexFile):
		print(str(lexFile[v]) + "...")
		lexData = open("Lexicons/"+lexFile[v]+".txt").readlines()
		for row in lexData:
			tokens = row[:-1].split("\t")
			lexTags[numLex] = tokens[0] # lex tag for index value
			tokens2 = tokens[1].split(",") # get words for that tag

			for t in range(0, len(tokens2)):
				if len(tokens2[t]) > 1:
					#print(tokens2[t])
					i = int(numLex)
					lex[i][lexCt[i]] = tokens2[t]
					lexCt[i] = lexCt[i] + 1
					index = HashLetters(tokens2[t])
					hashlex[index[0]][index[1]][hashlexCt[index[0]][index[1]]] = tokens2[t]
					hashlexClust[index[0]][index[1]][hashlexCt[index[0]][index[1]]] = i
					hashlexCt[index[0]][index[1]] = hashlexCt[index[0]][index[1]] + 1
			totlex+= lexCt[numLex]

			numLex = numLex + 1

	print("NumLex = ", numLex, "NumLexItems = ",totlex)
	for x in range(0, numLex):
		for y in range(0, lexCt[x]):
			index = HashLetters(lex[x][y])
			for z in range(0, sentiscoresCt[index[0]][index[1]]):
				if str(lex[x][y]).lower() == str(sentiscores[index[0]][index[1]][z][0]).lower():
					lexSentiScores[x] = lexSentiScores[x] + float(sentiscores[index[0]][index[1]][z][2])
					break
		#print(x,lexTags[x],lexSentiScores[x],lexCt[x])
		if lexCt[x] >0: lexSentiScores[x]= float(lexSentiScores[x])/float(lexCt[x])

def AssignSemanticWeights():
	global featureIndex, featureStr, trainWeight
	print("Adding semantic weights")
	# assigns semantic weights and appends these to train weights 
	for x in range(0, numFeat):
		if x % 10000 == 0:
			print(str(x) + "...")

		categ = cat[featureIndex[x][1]]
		if categ == "WORD" or categ == "LEGOMENA" or categ == "HYPERNYM" or categ == "AFFECT" or categ =="SENTIMENT":
			valueSemantic = NGramSemantic(featureStr[x])
			trainWeight[x]+= valueSemantic
			#if trainWeight[x]>3: print(x,featureStr[x],trainWeight[x],valueSemantic)
			#trainWeight[x] = trainWeight[x] + NGramSemantic(featureStr[x])
		elif categ == "POS":
			trainWeight[x] = trainWeight[x] + POSSemantic(featureStr[x])
		elif categ == "WORD&POS":
			trainWeight[x] = trainWeight[x] + POSWordSemantic(featureStr[x])
		elif categ == "LEXICON":
			valueSemantic = LEXSemantic(featureStr[x])
			trainWeight[x] += valueSemantic
			if trainWeight[x]>3: print(x,featureStr[x],trainWeight[x],valueSemantic)

def NGramSemantic(word):
	#global sentiscoresCt
	tokens = re.split("_|-| |\\|_\\|",word)
	tscores = np.zeros(len(tokens), dtype='float')
	score = 0.0

	for c in range(0, len(tokens)):
		## extract letter indices

		#for testing...
		#print(len(tokens),tokens[c])

		if len(tokens[c]) >= 2:
			index = HashLetters(tokens[c])
			#print("HashLetters",index[0],index[1],sentiscoresCt[index[0]][index[1]])
			numWords = 0
			## find potential matches for each word
			for x in range(0, sentiscoresCt[index[0]][index[1]]):
				if str(sentiscores[index[0]][index[1]][x][0]).lower() == str(tokens[c]).lower():
					tscores[c] = tscores[c] + float(sentiscores[index[0]][index[1]][x][2])
					numWords = numWords + 1

			if numWords == 0:
				numWords = 1

			tscores[c] = float(tscores[c] / numWords) # average for each token across senses
			score = float(score) + float(tscores[c])

	score = float(float(score) / float(len(tokens)))
	return score

def POSSemantic(word):
	#global flipposwordCt, sentiscoresCt

	#tokens = word.split(" |\\|_\\|")
	tokens = re.split(" |\\|_\\|",word)
	tscores = np.zeros(len(tokens), dtype='float')
	score = 0.0
	for c in range(0, len(tokens)):
		## extract letter indices
		if len(tokens[c]) >= 2:
			index = HashLetters(tokens[c])
			poswords = ["0"]*100000
			numpw = 0

			# get tag sense
			psense = "n"
			for d in range(0, len(tokens[c])-1):
				if tokens[c][d:d+2] == "JJ":
					psense = "a"
				if tokens[c][d:d+2] == "VB":
					psense = "v"
				if tokens[c][d:d+2] == "RB":
					psense = "r"
				if tokens[c][d:d+2] == "NN":
					psense = "n"

			# get all words containing that pos tag
			for x in range(0, int(flipposwordCt[int(index[0])][int(index[1])])):
				tokens2 = re.split(" |\\|_\\|",flipposword[int(index[0])][int(index[1])][x])
				if len(tokens2) >= 2:
					if tokens2[0] == tokens[c]:
						isNew = True
						for v in range(0, numpw):
							if tokens2[1] == poswords[v]:
								isNew = False
								break
						if isNew:
							if len(tokens2[1]) >= 2:
								poswords[numpw] = tokens2[1]
								numpw = numpw + 1

			numWords = 0
			for k in range(0, numpw):
				index = HashLetters(tokens[c])
				for x in range(0, sentiscoresCt[index[0]][index[1]]):
					if str(sentiscores[index[0]][index[1]][x][0]).lower() == str(poswords[k]).lower() and sentiscores[index[0]][index[1]][x][1] == psense:
						tscores[c] = tscores[c] + float(sentiscores[index[0]][index[1]][x][2])
						numWords = numWords + 1

			if numWords == 0:
				numWords = 1
			tscores[c] = tscores[c] / numWords
			score = score + tscores[c]

	score = float(score) / float(len(tokens))
	return score

def POSWordSemantic(word):
	#global flipposwordCt, sentiscoresCt

	#tokens = word.split(" |\\|_\\|")
	tokens = re.split(" |\\|_\\|",word)
	tscores = np.zeros(len(tokens), dtype='float')
	score = 0.0

	for c in range(1, len(tokens), 2):
		if len(tokens[c-1]) >= 2:
			index = HashLetters(tokens[c])
			numWords = 0
			psense = "null"

			# get POSword sense from tag
			for d in range(0, len(tokens[c])-1):
				if tokens[c][d:d+2] == "JJ":
					psense = "a"
				if tokens[c][d:d+2] == "VB":
					psense = "v"
				if tokens[c][d:d+2] == "RB":
					psense = "r"
				if tokens[c][d:d+2] == "NN":
					psense = "n"

			for x in range(0, sentiscoresCt[index[0]][index[1]]):
				if psense == "null":
					if str(sentiscores[index[0]][index[1]][x][0]).lower() == str(tokens[c-1]).lower():
						tscores[c] = tscores[c] + float(sentiscores[index[0]][index[1]][x][2])
						numWords = numWords + 1
				else:
					if str(sentiscores[index[0]][index[1]][x][0]).lower() == str(tokens[c-1]).lower() and sentiscores[index[0]][index[1]][x][1] == psense:
						tscores[c] = tscores[c] + float(sentiscores[index[0]][index[1]][x][2]);
						numWords = numWords + 1;

			if numWords == 0:
				numWords = 1
			tscores[c] = tscores[c] / numWords
			score = score + tscores[c]

	score = float(score) / (float(len(tokens))/2)
	return score

def LEXSemantic(word):
	#global flipposwordCt, sentiscoresCt, numLex, lexSentiScores

	#tokens = word.split(" |\\|_\\|")
	tokens = re.split(" |\\|_\\|",word)
	score = 0.0
	notLex = True

	for c in range(0, len(tokens)):
		for t in range(0, numLex):
			if tokens[c] == lexTags[t]:
				score = score + lexSentiScores[t]
				notLex = False
				break

		if notLex:
			score = score + NGramSemantic(tokens[c])

	score = float(score) / float(len(tokens))
	return score

def RunSubsumptions():
	#global numCat, cat, catN
	#global thresh, subThresh
	# this method runs within-category subsumptions
	print("\nRunning within-category subsumption relations")
	matches = []

	# begin with within-category subsumptions
	for c in range(0, numCat):
	#for c in range(0, 2):
		print("Subsuming category ", c+1, " of ", numCat, cat[c])

		#loop through n's within category
		for n in range(1, catN[c]):
			for m in range(n+1, catN[c] + 1):
				SubsumeCatN(c,c,n,m); #e.g., 4-3, 3-2, 2-1 when m=n-1, but also covers 4-2, 4-1, etc.

def SubsumeCatN(catVal,compVal,n1,n2):
	#global numFeat, thresh, trainWeight, featureStr, featureIndex
	#global thresh, subThresh
	global featureIndex, outLogSub
	ct = datetime.datetime.now() 
	print("Subsuming", n1, " versus ", n2, ct)
	LoadHash(compVal, n2, 1)
	matches = []

	for f in range(0, numFeat):
	#for f in range(0, 100):
		# low weight features' status changed to inactive
		if trainWeight[f] <= thresh:
			featureIndex[f][0] = 0

		#if runLogs:
		#	outLogSub.write("********SubsumeCatN"+"\t"+str(f) + "," + str(featureStr[f])+","+str(trainWeight[f])+","+str(featureIndex[f][0])+"\n");
		
		if featureIndex[f][1] == catVal and featureIndex[f][2] == n1 and featureIndex[f][0]==1:
			# only select category features with status set to "active"
			if cat[catVal] == "CHAR":
				matches, matchNum = MatchCharSubstrings(featureStr[f], catVal, compVal)
			else:
				matches, matchNum = MatchSubstrings(featureStr[f], catVal, compVal)

			SubsumeFeatures(f,matches, matchNum)

def HashLetters(strToken):
	vals = np.zeros((2),dtype=int)
	vals = [-1,-1]

	if len(strToken) >= 2:
		indexa = ord(str(strToken[0]).lower()) - ord('a')
		indexb = ord(str(strToken[1]).lower()) - ord('a')
		
		if indexa < 0 or indexa > 26:
			indexa = 26
		if indexb < 0 or indexb > 26:
			indexb = 26

		vals[0] = indexa
		vals[1] = indexb

	return vals

def LoadHash(c, n, fStatus):
	#global maxhash, featureIndex, numFeat, featureStr, ft, ftIndex, ftPosition, ftCt
	#global thresh, subThresh
	global ft, ftIndex, ftPosition, ftCt
	# initialize super hash arrays
	ft = np.zeros((27,27,maxhash), dtype=object)
	ftIndex = np.zeros((27,27,maxhash), dtype="int32")
	ftPosition = np.zeros((27,27,maxhash), dtype="int32")
	ftCt = np.zeros((27,27), dtype="int32")

	# add all category n2 variables with active status to super hash array
	for f in range(0, numFeat):
		if featureIndex[f][1] == c and featureIndex[f][2] == n and featureIndex[f][0] >= fStatus:
			tokens = re.split(" |\\|_\\|",featureStr[f])
			for t in range(0, len(tokens)):
				index = HashLetters(tokens[t])
				if index[0] >= 0 and index[1] >= 0 and ftCt[index[0]][index[1]] < maxhash: #only those with at least 2 chars...and storing upto maxhash limit only. WARNING: features beyond maxhash limit won't be considered!!!
						ft[index[0]][index[1]][ftCt[index[0]][index[1]]] = tokens[t]
						ftIndex[index[0]][index[1]][ftCt[index[0]][index[1]]] = f
						ftPosition[index[0]][index[1]][ftCt[index[0]][index[1]]] = t
						ftCt[index[0]][index[1]] = ftCt[index[0]][index[1]] + 1

def MatchCharSubstrings(worda, c1, c2):
	#global numFeat, ftCt, ft, ftIndex
	#global thresh, subThresh
	matchIndices = np.zeros(100000, dtype="int32")
	numMatch = 0

	matchScore = np.zeros(numFeat, dtype="int")

	if len(worda) >= 2:
		index = HashLetters(worda)
		for x in range(0, ftCt[index[0]][index[1]]):
			if len(ft[index[0]][index[1]][x]) >= len(worda):
				if ft[index[0]][index[1]][x][0:len(worda)] == worda:
					matchIndices[numMatch] = ftIndex[index[0]][index[1]][x]
					numMatch = numMatch + 1

	return matchIndices, numMatch

def MatchSubstrings(worda, c1, c2):
	#global numFeat, ftCt, ft, ftIndex, featureStr, thresh, subThresh
	matchIndices = np.zeros(100000, dtype="int32")
	numMatch = 0

	matchScore = np.zeros(numFeat, dtype="int")
	tokens = re.split(" |\\|_\\|",worda)
	numToke = len(tokens)
	for t in range(0, numToke):
		index = HashLetters(tokens[t])

		# compare with hash array
		if index[0] >= 0 and index[1] >= 0:
			for x in range(0, ftCt[index[0]][index[1]]):
				if ft[index[0]][index[1]][x] == tokens[t]:
					matchScore[ftIndex[index[0]][index[1]][x]] = matchScore[ftIndex[index[0]][index[1]][x]] + 1
					#if worda=="absence":
					#	print (worda,featureStr[ftIndex[index[0]][index[1]][x]],matchScore[ftIndex[index[0]][index[1]][x]])

	for y in range(0, numFeat):
		foundMatch = False
		if matchScore[y] == numToke:
			if numToke > 1:
				tokens2 = re.split(" |\\|_\\|",featureStr[y])
				for z in range(0, len(tokens2)):
					if tokens[0] == tokens2[z]:
						if len(tokens2)-z-1 >= numToke-1 and (c1 == c2 or cat[c2] != "WORD&POS"):
							if numToke == 2:
								if tokens[1] == tokens2[z+1]:
									foundMatch = True
							elif numToke == 3:
								if tokens[1] == tokens2[z+1] and tokens[2] == tokens2[z+2]:
									foundMatch = True
							elif numToke == 4:
								if tokens[1] == tokens2[z+1] and tokens[2] == tokens2[z+2] and tokens[3] == tokens2[z+3]:
									foundMatch = True
							elif numToke == 5:
								if tokens[1] == tokens2[z+1] and tokens[2] == tokens2[z+2] and tokens[3] == tokens2[z+3] and tokens[4] == tokens2[z+4]:
									foundMatch = True
					elif len(tokens2)-z-1 >= 2*(numToke-1) and (cat[c1] == "WORD" or cat[c1] == "POS") and cat[c2] == "WORD&POS":
						if numToke == 2:
							if tokens[1] == tokens2[z+2]:
								foundMatch = True
						elif numToke == 3:
							if tokens[1] == tokens2[z+2] and tokens[2] == tokens2[z+4]:
								foundMatch = True
						elif numToke == 4:
							if tokens[1] == tokens2[z+2] and tokens[2] == tokens2[z+4] and tokens[3] == tokens2[z+6]:
								foundMatch = True
						elif numToke == 5:
							if tokens[1] == tokens2[z+2] and tokens[2] == tokens2[z+4] and tokens[3] == tokens2[z+6] and tokens[4] == tokens2[z+8]:
								foundMatch = True
			else:
				foundMatch = True

		if foundMatch:
			matchIndices[numMatch] = y
			numMatch = numMatch + 1

	return matchIndices, numMatch

def SubsumeFeatures(indexa, indexb, numM):
	#global thresh, trainWeight, subThresh, trainWeightC, featureIndex, runLogs, outLogSub
	#global thresh, subThresh
	global trainWeight, featureIndex, outLogSub
	for b in range(0, numM):
		#if indexb[b] == 0:
		#	break

		#outLogSub.write("********SubsumeFeatures"+"\t"+str(indexa) + "," + str(featureStr[indexa])+","+str(trainWeight[indexa])+","+str(indexb[b])+"\n");

		if (trainWeight[indexb[b]] - subThresh) <= trainWeight[indexa] and trainWeight[indexb[b]] > thresh and trainWeightC[indexb[b]][0] == trainWeightC[indexa][0] and trainWeightC[indexb[b]][1] == trainWeightC[indexa][1]:
			trainWeight[indexb[b]] = thresh
			featureIndex[indexb[b]][0] = 0 #deactivate subsumed feature

			if runLogs:
				outLogSub.write(str(indexa)+","+str(featureStr[indexa]) + "," + str(trainWeight[indexa]) + "  \t" + str(indexb[b])+","+str(featureStr[indexb[b]])  + "," + str(trainWeight[indexb[b]]) +"\n")

def RunCCSubsumptions():
	#global numCat, cat, catN
	print("Running cross-category subsumption relations")
	matches = []
	wordC = 0
	POSC = 0
	charC = 0
	for c in range(0, numCat):
		if cat[c] == "WORD":
			wordC = c
		if cat[c] == "POS":
			POSC = c
		if cat[c] == "CHAR":
			charC = c

	for c in range(0, numCat):
		#run Word against hapax, PosWord, lexicons, hypermyn, sentiment, affect, and CharTri
		if cat[c] == "LEGOMENA" or cat[c] == "LEXICON" or cat[c] == "WORD&SENSE" or cat[c] == "SENTIMENT" or cat[c] == "AFFECT" or cat[c] == "HYPERNYM":
			#loop through n's within category for wordC
			for n in range(1, catN[wordC]):
				for m in range(n+1, catN[c] + 1):
					SubsumeCatN(wordC,c,n,m); # e.g., 4-3, 3-2, 2-1 when m=n-1, but also covers 4-2, 4-1, etc.
		if cat[c] == "WORD&POS":
			for n in range(1, catN[wordC] + 1):
				SubsumeCatN(wordC,c,n,n) # e.g., 4-4, 3-3, 2-2
				SubsumeCatN(POSC,c,n,n)
		if cat[c] == "CHAR":
			for n in range(1, catN[charC] + 1):
				SubsumeCatN(c,wordC,n,1); # e.g., charbi-word, chartri-word

def RunParallels():
	#global numCat, catN, cat
	print("Running parallel relations")
	lexC = 0
	posC = 0
	hyperC = 0
	affectC = 0
	sentiC = 0
	wordsenseC = 0
	nerC = 0
	misC = 0
	for c in range(0, numCat):
		if cat[c] == "LEXICON":
			lexC = c
		if cat[c] == "POS":
			posC = c
		if cat[c] == "AFFECT":
			affectC = c
		if cat[c] == "SENTIMENT":
			sentiC = c
		if cat[c] == "HYPERNYM":
			hyperC = c
		if cat[c] == "WORD&SENSE":
			wordsenseC = c
		if cat[c] == "NER":
			nerC = c
		if cat[c] == "MISSPELLING":
			misC = c

	# go through categories
	for c in range(0, numCat):
		if cat[c] == "WORD&POS":
			for n in range(1, catN[c] + 1):
				ParallelCatN(c,lexC,n,n); # e.g., 1-1, 2-2, 3-3, etc.
		elif cat[c] == "WORD":
			for n in range(1, catN[c] + 1):
				ParallelCatN(c,posC,n,n); #WORD and POS
				ParallelCatN(c,lexC,n,n);  #WORD and LEXICON
				ParallelCatN(c,hyperC,n,n);  #WORD and HYPERNYM
				ParallelCatN(c,sentiC,n,n);  #WORD and SENTIMENT
				ParallelCatN(c,nerC,n,n);  #WORD and NER
				ParallelCatN(c,affectC,n,n);  #WORD and AFFECT
				ParallelCatN(c,wordsenseC,n,n);  #WORD and WORD&SENSE
				ParallelCatN(c,misC,n,n);  #WORD and MISSPELLING

def ParallelCatN(catVal, compVal, n1, n2):
	#global numCat, cat, trainWeight, featureIndex, thresh, numFeat
	ct = datetime.datetime.now() 
	print("Parallelizing", cat[catVal], cat[compVal], n1, " versus ", n2, ct)
	if cat[compVal] == "POS":
		posWordC = 0
		for c in range(0, numCat):
			if cat[c] == "WORD&POS":
				posWordC = c
		LoadHash(posWordC, n1, 0)
	else:
		LoadHash(compVal,n2,1)

	for f in range(0, numFeat):
		if trainWeight[f] <= thresh:
			featureIndex[f][0] = 0

		if featureIndex[f][1] == catVal and featureIndex[f][2] == n1 and featureIndex[f][0] == 1:
			if cat[catVal] == "WORD" and cat[compVal] == "LEXICON":
				ParaLex(featureStr[f], f, catVal, compVal)
			elif cat[catVal] == "WORD&POS" and cat[compVal] == "LEXICON":
				tokens = re.split(" |\\|_\\|",featureStr[f])
				ftr = tokens[0]
				if len(tokens) > 2:
					for x in range(2, len(tokens)):
						if x % 2 == 0:
							ftr = ftr + " " + tokens[x]
				ParaLex(ftr, f, catVal, compVal)
			elif cat[compVal] == "POS":
				ParaPOS(featureStr[f], f, catVal, compVal, n2)
			elif cat[compVal] == "AFFECT" or cat[compVal] == "SENTIMENT" or cat[compVal] == "HYPERNYM" or cat[compVal] == "WORD&SENSE" or cat[compVal] == "NER" or cat[compVal] == "MISSPELLING":
				matches, numResp = MatchSubstrings(featureStr[f], catVal, compVal)
				if numResp > 0:
					Correlation(f,matches,catVal,compVal)

def ParaLex(worda, f, c1, c2):
	#global hashlexCt, lexTags, hashlexClust, hashlex
	matchIndices = np.zeros(100000, dtype="int32")
	# parallel relations: compare word tokens against lexicons
	tokens = re.split(" |\\|_\\|",worda)
	numToke = len(tokens)
	tokLex = np.zeros(numToke, dtype=object)
	tokeLimit = np.zeros(numToke, dtype="int32")
	numPot = 0
	numlex = 0
	potQueries = []
	for t in range(0, numToke):
		if len(tokens[t]) >= 2:
			index = HashLetters(tokens[t])
			for x in range(0, hashlexCt[index[0]][index[1]]):
				if hashlex[index[0]][index[1]][x] == tokens[t]: #cluster number for a given token
					tokLex[t] = lexTags[hashlexClust[index[0]][index[1]][x]] #lex tag set for a given word token
					tokeLimit[t] = tokeLimit[t] + 1 #increment Limit? for that token???
					numlex = numlex + 1 #total number of lex matches
					break

	# generate potential query strings
	numPot = int(math.pow(2, numlex))
	potQueries = np.zeros(numPot, dtype=object)
	potSCt = np.zeros(numPot, dtype="int32")
	for a in range(0, numPot):
		potQueries[a] = ""

	pCt = 0
	for t in range(0, numToke):
		aCt = 0
		if t == 0 and tokeLimit[t] > 0:
			potQueries[aCt] = tokLex[t] #add "SYN"
			potSCt[aCt] = potSCt[aCt] + 1 #increment sem counter for string
			aCt = aCt + 1
		elif t > 0 and tokeLimit[t] > 0:
			for a in range(pCt, pCt+pCt):
				potQueries[a] = potQueries[a-pCt]+" "+tokLex[t] #need to double array size with SYN additions
			potSCt[aCt] = potSCt[aCt] + 1

		if t == 0:
			potQueries[aCt] = tokens[t] #add token in 0 or 1 slot
			aCt = aCt + 1
		else:
			for a in range(0, pCt):
				potQueries[a] = potQueries[a] + tokens[t]

		pCt = pCt + aCt

	for v in range(0, pCt):
		if len(potQueries[v]) >= 2 and potSCt[v] > 0:
			matchIndices, numResp = MatchSubstrings(potQueries[v],c1,c2);
			if numResp > 0:
				Correlation(f,matchIndices,c1,c2) #if not empty, send to correlation analyzer

def Correlation( indexa, comp, cat1, cat2):
	#global numInst, matrix, thresh, corrThresh, trainWeight, featureIndex, outLogPar
	global trainWeight, featureIndex, outLogPar
	vect1 = np.zeros(numInst, dtype="int32")
	for f in range(0, numInst):
		vect1[f] = matrix[indexa][f]
	for z in range(0, len(comp)):
		if comp[z] == 0:
			break
		if featureIndex[comp[z]][0] == 1: # check feature status
			vect2 = np.zeros(numInst, dtype="int32")
			for f in range(0, numInst):
				vect2[f]= matrix[comp[z]][f]

				corrcoff = 0
				mean1 = 0
				mean2 = 0
				cov = 0
				sum1 = 0
				sumsq1 = 0
				sum2 = 0
				sumsq2 = 0
				stdev1 = 0
				stdev2 = 0 

				for a in range(0, numInst):
					sum1 = sum1 + vect1[a]
					sumsq1 = sumsq1 + math.pow(vect1[a], 2)
				mean1 = float(sum1) / float(numInst)

				for a in range(0, numInst):
					sum2 = sum2 + vect2[a]
					sumsq2 = sumsq2 + math.pow(vect2[a], 2)
				mean2 = float(sum2) / float(numInst)

				#compute covariance
				for a in range(0, numInst):
					cov = cov + ( float(vect1[a]) - float(mean1)) * ( float(vect2[a]) - float(mean2))
				cov = cov / (numInst - 1)

				#compute stdev for vect 1 and 2
				stdev1 = ( float(numInst * sumsq1) - math.pow(sum1, 2)) / ( float(numInst) * (numInst - 1));
				stdev2 = ( float(numInst * sumsq2) - math.pow(sum2, 2)) / ( float(numInst) * (numInst - 1));

				stdev1 = math.pow(stdev1, 0.5)
				stdev2 = math.pow(stdev2, 0.5)

				if stdev1>0 and stdev2>0: corrcoff = float(cov) / (float(stdev1) * float(stdev2))
				else: corrcoff = 0

				if corrcoff >= corrThresh:
					trainWeight[comp[z]] = thresh
					featureIndex[comp[z]][0] = 0 #disable feature from future analysis

					if runLogs:
						outLogPar.write(str(cat[cat1])+","+str(featureStr[indexa]) + "," + str(trainWeight[indexa]) + "  \t" + str(cat[cat2])+","+str(featureStr[comp[z]]) + "," + str(trainWeight[comp[z]]) +"\t"+str(corrcoff)+"\n")

def ParaPOS(worda, f, c1, c2, n):
	#global numCat, cat, featureIndex
	wordPOSWordIndices = np.zeros(100000, dtype="int32")
	matchIndices = np.zeros(100000, dtype="int32")
	numMatch = 0

	# parallel relations: compare word tokens against POS
	# need to get POSWord equivalents, first

	posWordC = 0
	for c in range(0, numCat):
		if cat[c] == "WORD&POS":
			posWordC = c

	wordPOSWordIndices, numResp = MatchSubstrings(featureStr[f], c1, posWordC)
	# next, loop through this set and extract POS tag strings
	for k in range(0, numResp):
		tag = ""
		pw = featureStr[wordPOSWordIndices[k]]
		tokens = re.split(" |\\|_\\|",pw)
		for t in range(0, len(tokens)):
			if t % 2 == 1:
				tag = tokens[t]
			else:
				tag = tag + " " + tokens[t]

		#gives tag string devoid of words (POS only) which can be found
		for z in range(0, numFeat):
			if featureIndex[z][1] == c2 and featureIndex[z][2] == n and featureIndex[z][0] == 1:
				if featureStr[z] == tag:
					matchIndices[numMatch] = z
					numMatch = numMatch + 1
					break

	if numMatch>0:
		Correlation(f,matchIndices,c1,c2)

def OutputRankings():
	#global numFeat, featureCatStr, featureStr, trainWeight
	outFile = open(weightFile, "w")
	for b in range(0, numFeat):
		outFile.write(str(b+1)+"\t"+str(featureStr[b])+"\t"+str(featureCatStr[b]).strip("\n")+"\t"+str(trainWeight[b])+"\n")


## complete
ct = datetime.datetime.now() 
print("current time:-", ct) 

ReadFeatures()
ReadTrain()
ReadSentiScores()
ReadLex()
AssignTrainWeights()
AssignSemanticWeights()
RunSubsumptions()
RunCCSubsumptions()
outLogSub.close()
RunParallels()
outLogPar.close()
OutputRankings()