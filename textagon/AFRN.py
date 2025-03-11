import os
import sys
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
import re





class AFRN:
	def __init__(
			self,
			featuresFile,
			trainFile, 
			weightFile, 
			numFeat = 0,
			foundNACt = 0,
			foundNA = False,
			numLexFile = 0,
			numCat = 0,
			maxposword = 0,
			numInst = 0,
			numClass = 0,
			semMax = 50000,
			maxhash = 50000,
			numLex = 0,
			flipposwordCt = np.zeros((27,27),dtype=int),
			sentiscoresCt = np.zeros((27,27),dtype=int),
			sentiscores = np.zeros((27,27,1,3), dtype=object),
			lexFile = ["0"]*50,
			cat = ['0']*50,
			catN = [0]*50,
			classLabels = [0]*20,
			trainWeight = [],
			trainWeightC = [],
			instLabels = [],
			matrix = [],
			featureIndex = [],
			featureStr = [],
			outLogSub = "AFRNSubsumptionLogPy.txt",
			outLogPar = "AFRNParallelLogPy.txt",
			thresh = 0.0000000001,
			subThresh = 0.05,
			corrThresh = 0.95,
			runLogs = True,
			featureCatStr = None
	):
		self.featuresFile = featuresFile
		self.trainFile = trainFile
		self.weightFile = weightFile
		self.numFeat = numFeat
		self.foundNACt = foundNACt
		self.foundNA = foundNA
		self.numLexFile = numLexFile
		self.numCat = numCat
		self.maxposword = maxposword
		self.numInst = numInst
		self.numClass = numClass
		self.semMax = semMax
		self.maxhash = maxhash
		self.numLex = numLex
		self.flipposwordCt = flipposwordCt
		self.sentiscoresCt = sentiscoresCt
		self.sentiscores = sentiscores
		self.lexFile = lexFile
		self.cat = cat
		self.catN = catN
		self.classLabels = classLabels
		self.lexSentiScores = np.zeros(self.semMax, dtype="float"),
		self.trainWeight = trainWeight
		self.trainWeightC = trainWeightC
		self.instLabels = instLabels
		self.matrix = matrix
		self.featureIndex = featureIndex
		self.featureStr = featureStr
		self.outLogSub = open(outLogSub, "w")
		self.outLogPar = open(outLogPar, "w")
		self.thresh = thresh
		self.subThresh = subThresh
		self.corrThresh = corrThresh
		self.runLogs = runLogs
		self.featureCatStr = featureCatStr

		#semantic hashes
		self.lex = np.zeros((self.semMax,self.semMax), dtype=object)
		self.lexCt = np.zeros((self.semMax),dtype=int)
		self.hashlex = np.zeros((27,27,self.semMax), dtype=object)
		self.hashlexClust = np.zeros((27,27,self.semMax),dtype=int)
		self.hashlexCt = np.zeros((27,27),dtype=int)
		self.lexTags = np.zeros((self.semMax), dtype=object)


	def ReadFeatures(self):
		print("Loading features")
		featuresData = open(self.featuresFile, "r")
		line = featuresData.readline()

		#for testing...
		lCount1 = 0

		n = 0
		while line:

			tokens = line[:-1].split("\t")
			if tokens[0] == "NA" and tokens[1] == "NA-NA":
				self.foundNA = True #means we have class label and last X are vader
				self.foundNACt = self.foundNACt + 1 #figure out how many
			else:
				self.numFeat = self.numFeat + 1


			tokens2 = tokens[1].split("-")
			if len(tokens2) == 2 and tokens2[0] !="NA":

				#get N value
				n = int(tokens2[0])

				#get category string
				if len(tokens2[1]) > 7 and "LEXICON" in tokens2[1]:
					catStr="LEXICON"
					LF = tokens2[1][7:len(tokens2[1])]
					LFexists = False;

					for v in range(0, self.numLexFile):
						if self.lexFile[v] == LF:
								LFexists = True

					if LFexists == False:
						self.lexFile[self.numLexFile] = LF
						self.numLexFile = self.numLexFile + 1

				else:
					catStr = tokens2[1]

			else:
				n = 1
				catStr = tokens2[0]

			catFound = False
			for x in range(0, self.numCat):
				if catStr == self.cat[x]:
					catFound = True
					if n > self.catN[x]:
						self.catN[x] = n #increase max n for category if larger value found
					break

			if catFound == False:
				#add new category and current max n value for category
				self.cat[self.numCat] = catStr
				self.catN[self.numCat] = n
				print(str(self.numCat).strip("\n") + " " + str(catStr).strip("\n"))
				self.numCat = self.numCat + 1

			#handle flipposword
			if tokens[1] == "1-WORD&POS":
				tokens2 = re.split(" |\\|_\\|",tokens[0])
				flipWord = ""
				if len(tokens2) >= 2:
					flipWord = tokens2[1] + " " + tokens2[0]
					for f in range(3, len(tokens2), 2):
						flipWord = flipWord + " "+ tokens2[f] + " " + tokens2[f-1]

				if len(flipWord) >= 2:
					index = self.HashLetters(flipWord)
					self.flipposwordCt[index[0]][index[1]] = self.flipposwordCt[index[0]][index[1]] + 1
					if self.flipposwordCt[index[0]][index[1]] > self.maxposword:
						self.maxposword = self.flipposwordCt[index[0]][index[1]]
			#for testing...
			#if tokens[0] != "NA": lCount1+=1
			line = featuresData.readline()

		#ending first run
		print("Total categories found = ",self.numCat)
		print("Total features found = ",self.numFeat)
		print("Total lexicons = ",self.numLexFile)

		lCount = 0
		#initialize feature array and update hash array sizes
		self.featureIndex = np.zeros((self.numFeat, 3), dtype="int32") #status, catNum, and n
		self.featureStr = [""]*self.numFeat
		self.featureCatStr = [""]*self.numFeat
		self.flipposword = np.zeros((27,27,self.maxposword))
		self.flipposwordCt = np.zeros((27,27))

		featuresData.close()
		featuresData = open(self.featuresFile, "r")
		line = featuresData.readline()

		#for testing
		#lxCount = 0

		while line:
			#for testing...
			#if lxCount%10==0 and lxCount<200102:
			#	lCount = int(lxCount/10)

			tokens = line[:-1].split("\t");
			if tokens[0] != "NA" and tokens[1] != "NA-NA":
				self.featureIndex[lCount][0] = 1 #status column, 0 = discarded, 1 = retained/active
				
				#set letter indexes to 0 for now
				self.featureStr[lCount] = tokens[0]
				self.featureCatStr[lCount] = tokens[1]
				
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

				for x in range(0, self.numCat):
					if catStr == self.cat[x]:
						self.featureIndex[lCount][1] = x
						break

				self.featureIndex[lCount][2] = n

				if tokens[1] == "1-WORD&POS":
					#handle flip - populate hash array
					tokens2 = re.split(" |\\|_\\|",tokens[0])
					flipWord = ""
					if len(tokens2) >=2:
						flipWord= str(tokens2[1]) + " " + str(tokens2[0])

						for f in range(3, len(tokens2), 2):
							flipWord = flipWord + " " + tokens2[f] + " " + tokens2[f-1]

					if len(flipWord) >=2:
						index = self.HashLetters(flipWord)
						self.flipposword[index[0]][index[1]][self.flipposwordCt[index[0]][index[1]]] = flipWord
						self.flipposwordCt[index[0]][index[1]] = self.flipposwordCt[index[0]][index[1]] + 1

			#if tokens[0] != "NA": lxCount = lxCount + 1
				lCount+=1
			## read next line
			line = featuresData.readline()

		featuresData.close()

	def ReadTrain(self):
		print("Loading training data")
		trainData = open(self.trainFile, "r")
		line = trainData.readline()
		while line:
			tokens = line[:-1].split(",")
			if tokens[0] != "Class":
				self.numInst = self.numInst + 1
				#check to see if class label already added to label array
				isNew = True
				thisClass = int(tokens[0]) if tokens[0].isdigit() else 1
				for a in range(0, self.numClass):
					if thisClass == self.classLabels[a]:
						isNew = False
						break

				if isNew:
					self.classLabels[self.numClass] = thisClass
					self.numClass = self.numClass + 1

			line = trainData.readline()

		trainData.close()

		print("Classes=", self.numClass, self.classLabels[0], self.classLabels[self.numClass-1],"Num Instances = ", self.numInst)
		ftNum = 0
		if self.foundNA:
			ftNum = len(tokens) - self.foundNACt
		else:
			ftNum = len(tokens) - 1

		if ftNum != self.numFeat:
			print("Number of features in Features file and Train file are different!!!", ftNum, self.numFeat)

		#for testing...
		ftNum = self.numFeat
		self.matrix = np.zeros((ftNum, self.numInst),dtype=int)
		self.instLabels = np.zeros(self.numInst)
		self.trainWeight = np.zeros(ftNum,dtype="float")
		self.trainWeightC = np.zeros((ftNum, 2))

		trainData = open(self.trainFile, "r")
		line = trainData.readline()
		lCount = 0
		while line:
			tokens = line.split(",")
			if tokens[0] != "Class":
				cIndex = int(tokens[0]) if tokens[0].isdigit() else 1
				for c in range(0, self.numClass):
					if cIndex == self.classLabels[c]:
						self.instLabels[lCount] = (int)(c)
						break

				if self.foundNA:
					for a in range(1, len(tokens) - self.foundNACt + 1):
						self.matrix[a-1][lCount] = int(tokens[a]) if tokens[a].isdigit() else 1
				else:
					for a in range(1, len(tokens)):
						self.matrix[a-1][lCount] = int(tokens[a]) if tokens[a].isdigit() else 1

				lCount = lCount + 1

			line = trainData.readline()
		trainData.close()

	def AssignTrainWeights(self):
		print("Assigning training weights")
		bestScore = 0

		#numFeat = numFeat - 1
		#numFeat = 20
		for b in range(0, self.numFeat):
			sumc = np.zeros(self.numClass, dtype="float")
			pc = np.zeros(self.numClass, dtype="float")
			wc = np.zeros(self.numClass, dtype="float")
			wcc = np.zeros((self.numClass, self.numClass), dtype="float")
			totSum = 0
			for a in range(0, self.numInst):
				#print(a, b)
				sumc[int(self.instLabels[a])] = sumc[int(self.instLabels[a])] + self.matrix[b][a]
				totSum = totSum + self.matrix[b][a]

			#adjust for measurement error in 0.2% of feature strings
			if totSum==0:
				totSum = 1

			for a in range(0, self.numClass):
				pc[a] = float(sumc[a] / totSum)

			for a in range(0, self.numClass):
				for c in range(0, self.numClass):
					if a != c:
						if pc[a] > 0 and pc[c] > 0:
							wcc[a][c] = pc[a] * math.log( float(pc[a]) / pc[c])
						else:
							wcc[a][c] = float(sumc[a]*0.1)
						wc[a]= float(wc[a]+wcc[a][c])
				wc[a]=float(wc[a]/(self.numClass-1))

			#identify best score for the feature and its best class
			maxC = 0
			maxCC = 0
			maxVal = 0
			maxValC = 0
			for a in range(0, self.numClass):
				if wc[a] > maxVal:
					maxVal = wc[a]
					maxC = a + 1
					for c in range(0, self.numClass):
						val = float(wcc[a][c])
						if val > maxValC:
							maxCC = c + 1
							maxValC = val


			self.trainWeight[b] = maxVal
			#if b<20: print(b,featureStr[b],trainWeight[b])
			self.trainWeightC[b][0] = int(maxC)
			self.trainWeightC[b][1] = int(maxCC)

	def ReadSentiScores(self):
		#we know the max hash value from prior testing...
		sentiMax = 4763

		self.sentiscores = np.zeros((27,27,int(sentiMax),3), dtype=object)
		self.sentiscoresCt = np.zeros((27,27), dtype="int32")
		self.sentiScoresData = open("./data/sentiscores.txt").readlines()
		
		print("Loading sentiment scores",sentiMax)
		for row in self.sentiScoresData:
			tokens = re.split(",",row[:-1])
			#print(row[:-1],tokens,tokens[0],tokens[1],tokens[2])
			if len(tokens[0]) >= 2 and len(tokens) == 3:
				index = self.HashLetters(tokens[0])
				self.sentiscores[index[0]][index[1]][self.sentiscoresCt[index[0]][index[1]]][0] = tokens[0]
				self.sentiscores[index[0]][index[1]][self.sentiscoresCt[index[0]][index[1]]][1] = tokens[1]
				self.sentiscores[index[0]][index[1]][self.sentiscoresCt[index[0]][index[1]]][2] = str(abs(float(tokens[2]))) + ""
				
				#print(str(sentiscores[index[0]][index[1]][sentiscoresCt[index[0]][index[1]]][0]),str(sentiscores[index[0]][index[1]][sentiscoresCt[index[0]][index[1]]][1]),str(sentiscores[index[0]][index[1]][sentiscoresCt[index[0]][index[1]]][2]))
				self.sentiscoresCt[index[0]][index[1]] = self.sentiscoresCt[index[0]][index[1]] + 1


	def ReadLex(self):
		print("Loading lexicons...")
		#tag index number and quantity
		numLex = 0
		#number of total lex items across tags
		totlex = 0

		for v in range(0, self.numLexFile):
			print(str(self.lexFile[v]) + "...")
			lexData = open("Lexicons/"+self.lexFile[v]+".txt").readlines()
			for row in lexData:
				tokens = row[:-1].split("\t")
				self.lexTags[numLex] = tokens[0] # lex tag for index value
				tokens2 = tokens[1].split(",") # get words for that tag

				for t in range(0, len(tokens2)):
					if len(tokens2[t]) > 1:
						#print(tokens2[t])
						i = int(numLex)
						self.lex[i][self.lexCt[i]] = tokens2[t]
						self.lexCt[i] = self.lexCt[i] + 1
						index = self.HashLetters(tokens2[t])
						self.hashlex[index[0]][index[1]][self.hashlexCt[index[0]][index[1]]] = tokens2[t]
						self.hashlexClust[index[0]][index[1]][self.hashlexCt[index[0]][index[1]]] = i
						self.hashlexCt[index[0]][index[1]] = self.hashlexCt[index[0]][index[1]] + 1
				totlex+= self.lexCt[numLex]

				numLex = numLex + 1

		print("NumLex = ", numLex, "NumLexItems = ",totlex)
		for x in range(0, numLex):
			for y in range(0, self.lexCt[x]):
				index = self.HashLetters(self.lex[x][y])
				for z in range(0, self.sentiscoresCt[index[0]][index[1]]):
					if str(self.lex[x][y]).lower() == str(self.sentiscores[index[0]][index[1]][z][0]).lower():
						self.lexSentiScores[x] = self.lexSentiScores[x] + float(self.sentiscores[index[0]][index[1]][z][2])
						break
			#print(x,lexTags[x],lexSentiScores[x],lexCt[x])
			if self.lexCt[x] >0: self.lexSentiScores[x]= float(self.lexSentiScores[x])/float(self.lexCt[x])

	def AssignSemanticWeights(self):
		print("Adding semantic weights")
		# assigns semantic weights and appends these to train weights 
		for x in range(0, self.numFeat):
			if x % 10000 == 0:
				print(str(x) + "...")

			categ = self.cat[self.featureIndex[x][1]]
			if categ == "WORD" or categ == "LEGOMENA" or categ == "HYPERNYM" or categ == "AFFECT" or categ =="SENTIMENT":
				valueSemantic = self.NGramSemantic(self.featureStr[x])
				self.trainWeight[x]+= valueSemantic
				#if trainWeight[x]>3: print(x,featureStr[x],trainWeight[x],valueSemantic)
				#trainWeight[x] = trainWeight[x] + NGramSemantic(featureStr[x])
			elif categ == "POS":
				self.trainWeight[x] = self.trainWeight[x] + self.POSSemantic(self.featureStr[x])
			elif categ == "WORD&POS":
				self.trainWeight[x] = self.trainWeight[x] + self.POSWordSemantic(self.featureStr[x])
			elif categ == "LEXICON":
				valueSemantic = self.LEXSemantic(self.featureStr[x])
				self.trainWeight[x] += valueSemantic
				if self.trainWeight[x]>3: print(x,self.featureStr[x],self.trainWeight[x],valueSemantic)

	def NGramSemantic(self, word):
		#global sentiscoresCt
		tokens = re.split("_|-| |\\|_\\|",word)
		tscores = np.zeros(len(tokens), dtype='float')
		score = 0.0

		for c in range(0, len(tokens)):
			## extract letter indices

			if len(tokens[c]) >= 2:
				index = self.HashLetters(tokens[c])
				#print("HashLetters",index[0],index[1],sentiscoresCt[index[0]][index[1]])
				numWords = 0
				## find potential matches for each word
				for x in range(0, self.sentiscoresCt[index[0]][index[1]]):
					if str(self.sentiscores[index[0]][index[1]][x][0]).lower() == str(tokens[c]).lower():
						tscores[c] = tscores[c] + float(self.sentiscores[index[0]][index[1]][x][2])
						numWords = numWords + 1

				if numWords == 0:
					numWords = 1

				tscores[c] = float(tscores[c] / numWords) # average for each token across senses
				score = float(score) + float(tscores[c])

		score = float(float(score) / float(len(tokens)))
		return score

	def POSSemantic(self, word):

		#tokens = word.split(" |\\|_\\|")
		tokens = re.split(" |\\|_\\|",word)
		tscores = np.zeros(len(tokens), dtype='float')
		score = 0.0
		for c in range(0, len(tokens)):
			## extract letter indices
			if len(tokens[c]) >= 2:
				index = self.HashLetters(tokens[c])
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
				for x in range(0, int(self.flipposwordCt[int(index[0])][int(index[1])])):
					tokens2 = re.split(" |\\|_\\|",self.flipposword[int(index[0])][int(index[1])][x])
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
					index = self.HashLetters(tokens[c])
					for x in range(0, self.sentiscoresCt[index[0]][index[1]]):
						if str(self.sentiscores[index[0]][index[1]][x][0]).lower() == str(poswords[k]).lower() and self.sentiscores[index[0]][index[1]][x][1] == psense:
							tscores[c] = tscores[c] + float(self.sentiscores[index[0]][index[1]][x][2])
							numWords = numWords + 1

				if numWords == 0:
					numWords = 1
				tscores[c] = tscores[c] / numWords
				score = score + tscores[c]

		score = float(score) / float(len(tokens))
		return score

	def POSWordSemantic(self, word):

		#tokens = word.split(" |\\|_\\|")
		tokens = re.split(" |\\|_\\|",word)
		tscores = np.zeros(len(tokens), dtype='float')
		score = 0.0

		for c in range(1, len(tokens), 2):
			if len(tokens[c-1]) >= 2:
				index = self.HashLetters(tokens[c])
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

				for x in range(0, self.sentiscoresCt[index[0]][index[1]]):
					if psense == "null":
						if str(self.sentiscores[index[0]][index[1]][x][0]).lower() == str(tokens[c-1]).lower():
							tscores[c] = tscores[c] + float(self.sentiscores[index[0]][index[1]][x][2])
							numWords = numWords + 1
					else:
						if str(self.sentiscores[index[0]][index[1]][x][0]).lower() == str(tokens[c-1]).lower() and self.sentiscores[index[0]][index[1]][x][1] == psense:
							tscores[c] = tscores[c] + float(self.sentiscores[index[0]][index[1]][x][2]);
							numWords = numWords + 1;

				if numWords == 0:
					numWords = 1
				tscores[c] = tscores[c] / numWords
				score = score + tscores[c]

		score = float(score) / (float(len(tokens))/2)
		return score

	def LEXSemantic(self, word):

		#tokens = word.split(" |\\|_\\|")
		tokens = re.split(" |\\|_\\|",word)
		score = 0.0
		notLex = True

		for c in range(0, len(tokens)):
			for t in range(0, self.numLex):
				if tokens[c] == self.lexTags[t]:
					score = score + self.lexSentiScores[t]
					notLex = False
					break

			if notLex:
				score = score + self.NGramSemantic(tokens[c])

		score = float(score) / float(len(tokens))
		return score

	def RunSubsumptions(self):
		# this method runs within-category subsumptions
		print("\nRunning within-category subsumption relations")
		matches = []

		# begin with within-category subsumptions
		for c in range(0, self.numCat):
		#for c in range(0, 2):
			print("Subsuming category ", c+1, " of ", self.numCat, self.cat[c])

			#loop through n's within category
			for n in range(1, self.catN[c]):
				for m in range(n+1, self.catN[c] + 1):
					self.SubsumeCatN(c,c,n,m); #e.g., 4-3, 3-2, 2-1 when m=n-1, but also covers 4-2, 4-1, etc.

	def SubsumeCatN(self, catVal,compVal,n1,n2):

		ct = datetime.datetime.now() 
		print("Subsuming", n1, " versus ", n2, ct)
		self.LoadHash(compVal, n2, 1)
		matches = []

		for f in range(0, self.numFeat):
		#for f in range(0, 100):
			# low weight features' status changed to inactive
			if self.trainWeight[f] <= self.thresh:
				self.featureIndex[f][0] = 0

			#if runLogs:
			#	outLogSub.write("********SubsumeCatN"+"\t"+str(f) + "," + str(featureStr[f])+","+str(trainWeight[f])+","+str(featureIndex[f][0])+"\n");
			
			if self.featureIndex[f][1] == catVal and self.featureIndex[f][2] == n1 and self.featureIndex[f][0]==1:
				# only select category features with status set to "active"
				if self.cat[catVal] == "CHAR":
					matches, matchNum = self.MatchCharSubstrings(self.featureStr[f], catVal, compVal)
				else:
					matches, matchNum = self.MatchSubstrings(self.featureStr[f], catVal, compVal)

				self.SubsumeFeatures(f,matches, matchNum)

	def HashLetters(self, strToken):
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

	def LoadHash(self, c, n, fStatus):

		# initialize super hash arrays
		ft = np.zeros((27,27,self.maxhash), dtype=object)
		ftIndex = np.zeros((27,27,self.maxhash), dtype="int32")
		ftPosition = np.zeros((27,27,self.maxhash), dtype="int32")
		ftCt = np.zeros((27,27), dtype="int32")

		# add all category n2 variables with active status to super hash array
		for f in range(0, self.numFeat):
			if self.featureIndex[f][1] == c and self.featureIndex[f][2] == n and self.featureIndex[f][0] >= fStatus:
				tokens = re.split(" |\\|_\\|",self.featureStr[f])
				for t in range(0, len(tokens)):
					index = self.HashLetters(tokens[t])
					if index[0] >= 0 and index[1] >= 0 and ftCt[index[0]][index[1]] < self.maxhash: #only those with at least 2 chars...and storing upto maxhash limit only. WARNING: features beyond maxhash limit won't be considered!!!
							ft[index[0]][index[1]][ftCt[index[0]][index[1]]] = tokens[t]
							ftIndex[index[0]][index[1]][ftCt[index[0]][index[1]]] = f
							ftPosition[index[0]][index[1]][ftCt[index[0]][index[1]]] = t
							ftCt[index[0]][index[1]] = ftCt[index[0]][index[1]] + 1

	def MatchCharSubstrings(self, worda, c1, c2):

		matchIndices = np.zeros(100000, dtype="int32")
		numMatch = 0

		matchScore = np.zeros(self.numFeat, dtype="int")

		if len(worda) >= 2:
			index = self.HashLetters(worda)
			for x in range(0, self.ftCt[index[0]][index[1]]):
				if len(self.ft[index[0]][index[1]][x]) >= len(worda):
					if self.ft[index[0]][index[1]][x][0:len(worda)] == worda:
						matchIndices[numMatch] = self.ftIndex[index[0]][index[1]][x]
						numMatch = numMatch + 1

		return matchIndices, numMatch

	def MatchSubstrings(self, worda, c1, c2):

		matchIndices = np.zeros(100000, dtype="int32")
		numMatch = 0

		matchScore = np.zeros(self.numFeat, dtype="int")
		tokens = re.split(" |\\|_\\|",worda)
		numToke = len(tokens)
		for t in range(0, numToke):
			index = self.HashLetters(tokens[t])

			# compare with hash array
			if index[0] >= 0 and index[1] >= 0:
				for x in range(0, self.ftCt[index[0]][index[1]]):
					if self.ft[index[0]][index[1]][x] == tokens[t]:
						matchScore[self.ftIndex[index[0]][index[1]][x]] = matchScore[self.ftIndex[index[0]][index[1]][x]] + 1
						#if worda=="absence":
						#	print (worda,featureStr[ftIndex[index[0]][index[1]][x]],matchScore[ftIndex[index[0]][index[1]][x]])

		for y in range(0, self.numFeat):
			foundMatch = False
			if matchScore[y] == numToke:
				if numToke > 1:
					tokens2 = re.split(" |\\|_\\|",self.featureStr[y])
					for z in range(0, len(tokens2)):
						if tokens[0] == tokens2[z]:
							if len(tokens2)-z-1 >= numToke-1 and (c1 == c2 or self.cat[c2] != "WORD&POS"):
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
						elif len(tokens2)-z-1 >= 2*(numToke-1) and (self.cat[c1] == "WORD" or self.cat[c1] == "POS") and self.cat[c2] == "WORD&POS":
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

	def SubsumeFeatures(self, indexa, indexb, numM):

		for b in range(0, numM):
			#if indexb[b] == 0:
			#	break

			#outLogSub.write("********SubsumeFeatures"+"\t"+str(indexa) + "," + str(featureStr[indexa])+","+str(trainWeight[indexa])+","+str(indexb[b])+"\n");

			if (self.trainWeight[indexb[b]] - self.subThresh) <= self.trainWeight[indexa] and self.trainWeight[indexb[b]] > self.thresh and self.trainWeightC[indexb[b]][0] == self.trainWeightC[indexa][0] and self.trainWeightC[indexb[b]][1] == self.trainWeightC[indexa][1]:
				self.trainWeight[indexb[b]] = self.thresh
				self.featureIndex[indexb[b]][0] = 0 #deactivate subsumed feature

				if self.runLogs:
					self.outLogSub.write(str(indexa)+","+str(self.featureStr[indexa]) + "," + str(self.trainWeight[indexa]) + "  \t" + str(indexb[b])+","+str(self.featureStr[indexb[b]])  + "," + str(self.trainWeight[indexb[b]]) +"\n")

	def RunCCSubsumptions(self):

		print("Running cross-category subsumption relations")
		matches = []
		wordC = 0
		POSC = 0
		charC = 0
		for c in range(0, self.numCat):
			if self.cat[c] == "WORD":
				wordC = c
			if self.cat[c] == "POS":
				POSC = c
			if self.cat[c] == "CHAR":
				charC = c

		for c in range(0, self.numCat):
			#run Word against hapax, PosWord, lexicons, hypermyn, sentiment, affect, and CharTri
			if self.cat[c] == "LEGOMENA" or self.cat[c] == "LEXICON" or self.cat[c] == "WORD&SENSE" or self.cat[c] == "SENTIMENT" or self.cat[c] == "AFFECT" or self.cat[c] == "HYPERNYM":
				#loop through n's within category for wordC
				for n in range(1, self.catN[wordC]):
					for m in range(n+1, self.catN[c] + 1):
						self.SubsumeCatN(wordC,c,n,m); # e.g., 4-3, 3-2, 2-1 when m=n-1, but also covers 4-2, 4-1, etc.
			if self.cat[c] == "WORD&POS":
				for n in range(1, self.catN[wordC] + 1):
					self.SubsumeCatN(wordC,c,n,n) # e.g., 4-4, 3-3, 2-2
					self.SubsumeCatN(POSC,c,n,n)
			if self.cat[c] == "CHAR":
				for n in range(1, self.catN[charC] + 1):
					self.SubsumeCatN(c,wordC,n,1); # e.g., charbi-word, chartri-word

	def RunParallels(self):

		print("Running parallel relations")
		lexC = 0
		posC = 0
		hyperC = 0
		affectC = 0
		sentiC = 0
		wordsenseC = 0
		nerC = 0
		misC = 0
		for c in range(0, self.numCat):
			if self.cat[c] == "LEXICON":
				lexC = c
			if self.cat[c] == "POS":
				posC = c
			if self.cat[c] == "AFFECT":
				affectC = c
			if self.cat[c] == "SENTIMENT":
				sentiC = c
			if self.cat[c] == "HYPERNYM":
				hyperC = c
			if self.cat[c] == "WORD&SENSE":
				wordsenseC = c
			if self.cat[c] == "NER":
				nerC = c
			if self.cat[c] == "MISSPELLING":
				misC = c

		# go through categories
		for c in range(0, self.numCat):
			if self.cat[c] == "WORD&POS":
				for n in range(1, self.catN[c] + 1):
					self.ParallelCatN(c,lexC,n,n); # e.g., 1-1, 2-2, 3-3, etc.
			elif self.cat[c] == "WORD":
				for n in range(1, self.catN[c] + 1):
					self.ParallelCatN(c,posC,n,n); #WORD and POS
					self.ParallelCatN(c,lexC,n,n);  #WORD and LEXICON
					self.ParallelCatN(c,hyperC,n,n);  #WORD and HYPERNYM
					self.ParallelCatN(c,sentiC,n,n);  #WORD and SENTIMENT
					self.ParallelCatN(c,nerC,n,n);  #WORD and NER
					self.ParallelCatN(c,affectC,n,n);  #WORD and AFFECT
					self.ParallelCatN(c,wordsenseC,n,n);  #WORD and WORD&SENSE
					self.ParallelCatN(c,misC,n,n);  #WORD and MISSPELLING

	def ParallelCatN(self, catVal, compVal, n1, n2):

		ct = datetime.datetime.now() 
		print("Parallelizing", self.cat[catVal], self.cat[compVal], n1, " versus ", n2, ct)
		if self.cat[compVal] == "POS":
			posWordC = 0
			for c in range(0, self.numCat):
				if self.cat[c] == "WORD&POS":
					posWordC = c
			self.LoadHash(posWordC, n1, 0)
		else:
			self.LoadHash(compVal,n2,1)

		for f in range(0, self.numFeat):
			if self.trainWeight[f] <= self.thresh:
				self.featureIndex[f][0] = 0

			if self.featureIndex[f][1] == catVal and self.featureIndex[f][2] == n1 and self.featureIndex[f][0] == 1:
				if self.cat[catVal] == "WORD" and self.cat[compVal] == "LEXICON":
					self.ParaLex(self.featureStr[f], f, catVal, compVal)
				elif self.cat[catVal] == "WORD&POS" and self.cat[compVal] == "LEXICON":
					tokens = re.split(" |\\|_\\|",self.featureStr[f])
					ftr = tokens[0]
					if len(tokens) > 2:
						for x in range(2, len(tokens)):
							if x % 2 == 0:
								ftr = ftr + " " + tokens[x]
					self.ParaLex(ftr, f, catVal, compVal)
				elif self.cat[compVal] == "POS":
					self.ParaPOS(self.featureStr[f], f, catVal, compVal, n2)
				elif self.cat[compVal] == "AFFECT" or self.cat[compVal] == "SENTIMENT" or self.cat[compVal] == "HYPERNYM" or self.cat[compVal] == "WORD&SENSE" or self.cat[compVal] == "NER" or self.cat[compVal] == "MISSPELLING":
					matches, numResp = self.MatchSubstrings(self.featureStr[f], catVal, compVal)
					if numResp > 0:
						self.Correlation(f,matches,catVal,compVal)

	def ParaLex(self, worda, f, c1, c2):

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
				index = self.HashLetters(tokens[t])
				for x in range(0, self.hashlexCt[index[0]][index[1]]):
					if self.hashlex[index[0]][index[1]][x] == tokens[t]: #cluster number for a given token
						tokLex[t] = self.lexTags[self.hashlexClust[index[0]][index[1]][x]] #lex tag set for a given word token
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
				matchIndices, numResp = self.MatchSubstrings(potQueries[v],c1,c2);
				if numResp > 0:
					self.Correlation(f,matchIndices,c1,c2) #if not empty, send to correlation analyzer

	def Correlation(self, indexa, comp, cat1, cat2):

		vect1 = np.zeros(self.numInst, dtype="int32")
		for f in range(0, self.numInst):
			vect1[f] = self.matrix[indexa][f]
		for z in range(0, len(comp)):
			if comp[z] == 0:
				break
			if self.featureIndex[comp[z]][0] == 1: # check feature status
				vect2 = np.zeros(self.numInst, dtype="int32")
				for f in range(0, self.numInst):
					vect2[f]= self.matrix[comp[z]][f]

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

					for a in range(0, self.numInst):
						sum1 = sum1 + vect1[a]
						sumsq1 = sumsq1 + math.pow(vect1[a], 2)
					mean1 = float(sum1) / float(self.numInst)

					for a in range(0, self.numInst):
						sum2 = sum2 + vect2[a]
						sumsq2 = sumsq2 + math.pow(vect2[a], 2)
					mean2 = float(sum2) / float(self.numInst)

					#compute covariance
					for a in range(0, self.numInst):
						cov = cov + ( float(vect1[a]) - float(mean1)) * ( float(vect2[a]) - float(mean2))
					cov = cov / (self.numInst - 1)

					#compute stdev for vect 1 and 2
					stdev1 = ( float(self.numInst * sumsq1) - math.pow(sum1, 2)) / ( float(self.numInst) * (self.numInst - 1));
					stdev2 = ( float(self.numInst * sumsq2) - math.pow(sum2, 2)) / ( float(self.numInst) * (self.numInst - 1));

					stdev1 = math.pow(stdev1, 0.5)
					stdev2 = math.pow(stdev2, 0.5)

					if stdev1>0 and stdev2>0: corrcoff = float(cov) / (float(stdev1) * float(stdev2))
					else: corrcoff = 0

					if corrcoff >= self.corrThresh:
						self.trainWeight[comp[z]] = self.thresh
						self.featureIndex[comp[z]][0] = 0 #disable feature from future analysis

						if self.runLogs:
							self.outLogPar.write(str(self.cat[cat1])+","+str(self.featureStr[indexa]) + "," + str(self.trainWeight[indexa]) + "  \t" + str(self.cat[cat2])+","+str(self.featureStr[comp[z]]) + "," + str(self.trainWeight[comp[z]]) +"\t"+str(corrcoff)+"\n")

	def ParaPOS(self, worda, f, c1, c2, n):

		wordPOSWordIndices = np.zeros(100000, dtype="int32")
		matchIndices = np.zeros(100000, dtype="int32")
		numMatch = 0

		# parallel relations: compare word tokens against POS
		# need to get POSWord equivalents, first

		posWordC = 0
		for c in range(0, self.numCat):
			if self.cat[c] == "WORD&POS":
				posWordC = c

		wordPOSWordIndices, numResp = self.MatchSubstrings(self.featureStr[f], c1, posWordC)
		# next, loop through this set and extract POS tag strings
		for k in range(0, numResp):
			tag = ""
			pw = self.featureStr[wordPOSWordIndices[k]]
			tokens = re.split(" |\\|_\\|",pw)
			for t in range(0, len(tokens)):
				if t % 2 == 1:
					tag = tokens[t]
				else:
					tag = tag + " " + tokens[t]

			#gives tag string devoid of words (POS only) which can be found
			for z in range(0, self.numFeat):
				if self.featureIndex[z][1] == c2 and self.featureIndex[z][2] == n and self.featureIndex[z][0] == 1:
					if self.featureStr[z] == tag:
						matchIndices[numMatch] = z
						numMatch = numMatch + 1
						break

		if numMatch>0:
			self.Correlation(f,matchIndices,c1,c2)

	def OutputRankings(self):

		outFile = open(weightFile, "w")
		for b in range(0, self.numFeat):
			outFile.write(str(b+1)+"\t"+str(self.featureStr[b])+"\t"+str(self.featureCatStr[b]).strip("\n")+"\t"+str(self.trainWeight[b])+"\n")


## complete
ct = datetime.datetime.now() 
print("current time:-", ct) 




featuresFile = f'../examples/output/dvd_key.txt'
trainFile = f'../examples/output/dvd.csv'
weightFile = f'../examples/output/dvd_weights.txt'
"""#Extra reload"""
#lexFile = ['./Lexicons/lexicon_v5/' + file for file in os.listdir('./Lexicons/lexicon_v5/')]
#numLexFile = len(lexFile)


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