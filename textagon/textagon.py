import nltk

import os
import sys
import re
import fnmatch
from time import strftime
import csv
import gc
import psutil
import subprocess
import pkg_resources

from collections import OrderedDict

import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd
import itertools
import numpy as np

from bs4 import BeautifulSoup as BS

import zipfile as zf
import unicodedata
from datetime import datetime
from pytz import timezone
from tzlocal import get_localzone

import enchant
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter

import random
random.seed(1000)

import pickle
import mapply
import multiprocess.context as ctx
ctx._force_start_method('spawn')
import collections
from typing import Tuple, List
import pkg_resources


import warnings
warnings.filterwarnings('ignore', message = '.*looks like a URL.*', category = UserWarning, module = 'bs4')

# time display settings
fmt = '%Y-%m-%d %H:%M %p %Z'
start_time = datetime.now(get_localzone())
start_time_str = str(start_time.strftime(fmt))

# initialize mapply
useCores = mp.cpu_count()
mapply.init(
	n_workers = useCores) # chunk_size = 5

### Setup NLP Tools ###

# SentiWN #
from nltk.corpus import sentiwordnet as swn
swn.ensure_loaded()

import pkg_resources

wnaffect_path = pkg_resources.resource_filename('textagon', 'external/extracted/WNAffect-master')
wordnet_path = pkg_resources.resource_filename('textagon', 'external/extracted/wordnet-1.6')
wn_domains_path = pkg_resources.resource_filename('textagon', 'external/extracted/wn-domains')

sys.path.append(wnaffect_path)
from wnaffect import WNAffect
from emotion import Emotion
wna = WNAffect(wordnet_path, wn_domains_path)

# VADER #
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# spaCy #
import spacy
nlp = spacy.load('en_core_web_sm', exclude = ['lemmatizer'])
nlp.max_length = 10 ** 10

def spaCyTOK (sentence):
	"""_summary_

	:param sentence: _description_
	:type sentence: _type_
	"""

	doc = nlp.tokenizer(sentence)
	tokens = []
	for token in doc:
		tokens.append(token.text)
	return(tokens)

def splitWS (sentence):
	"""_summary_

	:param sentence: _description_
	:type sentence: _type_
	"""
	return(sentence.split(' '))

def vector_hasher(x):
	"""_summary_

	:param x: _description_
	:type x: _type_
	:return: _description_
	:rtype: _type_
	"""
	return hash(tuple(x))

pkg_resources.require('wn==0.0.23') # for pywsd

class SuppressStdErr:
	def __enter__ (self):
		self._original_stderr = sys.stderr
		sys.stderr = open(os.devnull, 'w')

	def __exit__ (self, exc_type, exc_val, exc_tb):
		sys.stderr.close()
		sys.stderr = self._original_stderr

with SuppressStdErr():
	import pywsd
	from pywsd import disambiguate
	from pywsd.lesk import adapted_lesk

### Below is for Parallel Representations ###

def say_hello():
	print("Hello, World!")

def setSpellChecking(exclusionsFileFullPath='None'):
	"""_summary_

	:param exclusionsFileFullPath: _description_, defaults to 'None'
	:type exclusionsFileFullPath: str, optional
	:return: _description_
	:rtype: _type_
	"""
	
	b = enchant.Broker()
	spellcheckerLibrary = 'en'
	b.set_ordering(spellcheckerLibrary, 'aspell')

	if exclusionsFileFullPath == 'None':
		# Use the default exclusions file
		exclusionsFileFullPath = pkg_resources.resource_filename('textagon', 'external/lexicons/exclusions.txt')
	elif not os.path.isfile(exclusionsFileFullPath):
		print('Provided exclusions file does not exist. Switching to default exclusions file.')
		# Switch to the default exclusions file if the provided one does not exist
		exclusionsFileFullPath = pkg_resources.resource_filename('textagon', 'external/lexicons/exclusions.txt')

	try:
		spellchecker = enchant.DictWithPWL(spellcheckerLibrary, pwl = exclusionsFileFullPath, broker = b)

		exclusionsFile = open(exclusionsFileFullPath, 'r')
		exclusionsLength = len(exclusionsFile.readlines())
		exclusionsFile.close()

		print('# Spellchecker Details #')
		print('Provider:', spellchecker.provider)
		print('Enchant Version:', enchant.get_enchant_version())
		print('Dictionary Tag:', spellchecker.tag)
		print('Dictionary Location:', spellchecker.provider.file)
		print('Total Exclusions: ' + str(exclusionsLength))

		# Return the values
		return spellcheckerLibrary, exclusionsFileFullPath, exclusionsLength, spellchecker
	except Exception as e:
		print(f'Error opening or reading file {exclusionsFileFullPath}: {e}')
		return None, None, 0, None

def ReadAllLexicons(lexiconFileFullPath=None):
	"""_summary_

	:param lexiconFileFullPath: _description_, defaults to None
	:type lexiconFileFullPath: _type_, optional
	:return: _description_
	:rtype: _type_
	"""

	# Get the path of the default lexicon file
	if lexiconFileFullPath is None:
		print('No lexicon file provided. Using default lexicon file.')
		lexiconFileFullPath = pkg_resources.resource_filename('textagon', 'external/lexicons/Lexicons_v5.zip')

	def is_valid_zip_file(file_path):
		"""_summary_

		:param file_path: _description_
		:type file_path: _type_
		:return: _description_
		:rtype: _type_
		"""
		# Check if file exists
		if not os.path.exists(file_path):
			print(f"Error: The file '{file_path}' does not exist.")
			print("Please ensure that you provide the full path to the file.")
			print('Example path looks like: C:/paht/to/zip/file/lexicon.zip')
			return False
		# Check if file is a zip file
		if not zf.is_zipfile(file_path):
			print(f"Error: The file '{file_path}' is not a valid zip file.")
			return False
		return True

	if not is_valid_zip_file(lexiconFileFullPath):
		return False
	
	customLexicons = {}

	def BuildLexicon (L, customLexicons):
		"""_summary_

		:param L: _description_
		:type L: _type_
		:param customLexicons: _description_
		:type customLexicons: _type_
		"""

		tagTokenPairs = list(filter(None, L.split('\n')))

		#print(tagTokenPairs)

		for i, tagTokenPair in enumerate(tagTokenPairs):
			elements = tagTokenPair.split('\t')
			tag = elements[0].strip().upper()
			#print(tag)
			#print(elements)
			tokens = elements[1].lower().split(',')
			tokens = [x.strip() for x in tokens]

			# add every lexicon word to spell checker (not used)
			'''
			for each in tokens:
				spellchecker.add(each)
			'''

			if i == 0:
				customLexicons[os.path.splitext(os.path.basename(file))[0].upper() ] = {tag: tokens}
			else:
				customLexicons[os.path.splitext(os.path.basename(file))[0].upper() ][tag] = tokens

		return(customLexicons)

	zipFile = zf.ZipFile(lexiconFileFullPath, 'r')
	for file in sorted(zipFile.namelist()):
		if fnmatch.fnmatch(file, '*.txt'):
			L = zipFile.read(file).decode('utf-8').encode('utf-8').decode('unicode-escape')
			customLexicons = BuildLexicon(L, customLexicons)
	print('# Custom Lexicons Imported:', len(customLexicons), '#')

	# sort lexicon names alphabetically
	customLexicons = OrderedDict(sorted(customLexicons.items()))

	if len(customLexicons) != 0:
		for key, value in customLexicons.items():

			# sort lexicon tags alphabetically
			customLexicons[key] = OrderedDict(sorted(value.items()))

			print('-', key, '(' + str(len(value)) + ' Tags)')
	print('\r')

	return(customLexicons)

def SanityCheck(dataPath: str = None, override_original_file: bool = False) -> Tuple[int, dict, List[Tuple[str, str]]]:
	"""_summary_

	:param dataPath: _description_, defaults to None
	:type dataPath: str, optional
	:param override_original_file: _description_, defaults to False
	:type override_original_file: bool, optional
	:return: _description_
	:rtype: Tuple[int, dict, List[Tuple[str, str]]]
	"""
	print("Sanity check started...")
	def is_valid_file(file_path: str) -> bool:
		if not os.path.exists(file_path):
			print(f"Error: The file '{file_path}' does not exist. Please ensure that you provide the full path to the file.")
			return False
		return True
	
	if not is_valid_file(dataPath):
		return -1, {}, []

	spellchecker = enchant.Dict("en_US")
	classes_counter = collections.Counter()
	raw_data = []

	with open(dataPath, 'r') as f:
		lines = f.readlines()

	for line in lines:
		parts = line.strip().split('\t')
		if len(parts) != 2:
			print(f"Error: Invalid format on line '{line.strip()}', each line should contain exactly two parts separated by a tab.")
			print('Please make sure your data have labels in first column and text in second column.')
			break

		label, text = parts
		classes_counter[label] += 1

		words = re.findall(r'\b\w+\b', text)
		checked_text = " ".join(word for word in words if spellchecker.check(word))

		raw_data.append((label, checked_text))

	num_classes = len(classes_counter)
	samples_per_class = dict(classes_counter)

	if override_original_file:
		with open(dataPath, 'w') as f:
			for label, text in raw_data:
				f.write(f"{label}\t{text}\n")
	print(f"Sanity check completed, found {num_classes} classes and {len(raw_data)} samples.")

	ret_dict = {
		'num_classes': num_classes,
		'samples_per_class': samples_per_class,
		'raw_data': raw_data
	}

	return ret_dict

def ReadRawText (path: str = None):
	"""_summary_

	:param path: _description_, defaults to None
	:type path: str, optional
	"""
	print("Reading raw text...")
	pure_chunck = SanityCheck(dataPath=path, override_original_file=False)
	classLabels = list(pure_chunck['samples_per_class'].keys())
	raw = [x[1] for x in pure_chunck['raw_data']]
	print("Reading raw text completed.")
	return({'corpus': raw, 'classLabels': classLabels})

def TextToFeatures (textData, debug = False, lexicons = None, wnaReturnLevel = 5, useSpellChecker = True, provideMisspellingDetailed = True, useCores = 1):
	"""_summary_

	:param textData: _description_
	:type textData: _type_
	:param debug: _description_, defaults to False
	:type debug: bool, optional
	:param lexicons: _description_, defaults to None
	:type lexicons: _type_, optional
	:param wnaReturnLevel: _description_, defaults to 5
	:type wnaReturnLevel: int, optional
	:param useSpellChecker: _description_, defaults to True
	:type useSpellChecker: bool, optional
	:param provideMisspellingDetailed: _description_, defaults to True
	:type provideMisspellingDetailed: bool, optional
	:param useCores: _description_, defaults to 1
	:type useCores: int, optional
	:return: _description_
	:rtype: _type_
	"""

	if lexicons == None:
		lexicons = ReadAllLexicons()
	else:
		if isinstance(lexicons, str):
			if os.path.exists(lexicons):
				lexicons = ReadAllLexicons(lexiconFileFullPath=lexicons)
		elif isinstance(lexicons, collections.OrderedDict):
			pass
		else:
			print('Lexicons must be a path to lexicon zip file or an well-done lexicon OrderedDict.')
			return None

	spellcheckerLibrary, exclusionsFileFullPath, exclusionsLength, spellchecker = setSpellChecking(exclusionsFileFullPath=exclusionsFileFullPath)

	textData = pd.DataFrame({
		'InitialSentence': textData
		})

	def BasicTextCleanup (sentence, debug = False):
		"""_summary_

		:param sentence: _description_
		:type sentence: _type_
		:param debug: _description_, defaults to False
		:type debug: bool, optional
		"""

		if debug:
			print('\nInitial Sentence:', sentence)

		# note: need to add exception handler (e.g., non-English issues)

		# Basic Cleaning
		initialSentenceLength = len(sentence)

		# Strip html
		sentence = BS(sentence, 'html.parser').get_text()
		htmlStripLength = initialSentenceLength - len(sentence)

		# Strip all excessive whitespace (after html to ensure no additional spaces result from html stripping)
		sentence = ' '.join(sentence.split())
		whitespaceStripLength = initialSentenceLength - htmlStripLength - len(sentence)

		# Spellchecking
		spellingCorrectionDetailsSentences = []
		spellingCorrectionDetailsWords = []
		spellingCorrectionDetailsSuggestions = []
		spellingCorrectionDetailsChosenSuggestion = []
		spellingCorrectionDetailsChangesWord = []
		spellingCorrectionDetailsReplacementLength = []
		spellingCorrectionCount = 0

		spellchecker = enchant.DictWithPWL(spellcheckerLibrary, pwl = exclusionsFileFullPath, broker = b)
		chkr = SpellChecker(spellchecker, sentence, filters = [EmailFilter, URLFilter])

		collectMisspellingDetails = {
			'Word': [], 
			'Substitution': [], 
			'SubstitutionText': []
			}

		for err in chkr:

			#print('\nSpellcheck Word:', err.word)
			matchedWord = False

			word = err.word

			if lexicons is not None and provideMisspellingDetailed:

				appendLexiconLabel = ''

				for lexicon, tagTokenPairs in lexicons.items():

					lexiconName = '|_|' + lexicon.upper() + '&'

					matchedWord = False  # note: we want to capture in multiple lexicons (but only once per lexicon)

					for tag, tokens in tagTokenPairs.items():

						if matchedWord:
							break

						elif any('*' in s for s in tokens):
							# regex mode
							nonmatching = [s for s in tokens if not s.endswith('*')]
							if word.lower() in nonmatching:
								appendLexiconLabel += lexiconName + tag.upper()
								matchedWord = True
							else:
								matching = [s for s in tokens if s.endswith('*')]
								for eachToken in matching:
									startString = eachToken[:-1]
									startStringUnique = set(startString)
									if startStringUnique != set('*'):
										if word.lower().startswith(startString):

											appendLexiconLabel += lexiconName + tag.upper()
											matchedWord = True
									else:
										if eachToken == word.lower():

											appendLexiconLabel += lexiconName + tag.upper()
											matchedWord = True

						elif word.lower() in tokens:

							appendLexiconLabel += lexiconName + tag.upper()
							matchedWord = True

				collectMisspellingDetails['SubstitutionText'].append('MISSPELLING' + appendLexiconLabel)

			#print(appendLexiconLabel)
			collectMisspellingDetails['Word'].append(err.word)
			collectMisspellingDetails['Substitution'].append('ABCMISSPELLING' + str(len(collectMisspellingDetails['Word'])) + 'XYZ')

			if (len(err.suggest()) == 0):
				spellingCorrectionDetailsSentences.append(sentence)
				spellingCorrectionDetailsChangesWord.append('True')
				spellingCorrectionDetailsWords.append(err.word)
				spellingCorrectionDetailsSuggestions.append(' | '.join(err.suggest()))
				spellingCorrectionDetailsChosenSuggestion.append('NA')
				spellingCorrectionDetailsReplacementLength.append('NA')
			else: # no need to count case corrections (e.g., i'm = I'm), but go ahead and perform them
				spellingCorrectionDetailsSentences.append(sentence)
				spellingCorrectionDetailsWords.append(err.word)
				spellingCorrectionDetailsSuggestions.append(' | '.join(err.suggest()))
				if err.word.lower() != err.suggest()[0].lower():
					spellingCorrectionDetailsChangesWord.append('True')
					spellingCorrectionCount += 1
				else:
					spellingCorrectionDetailsChangesWord.append('False')

				finalSuggestions = err.suggest()

				err.replace(finalSuggestions[0])
				spellingCorrectionDetailsChosenSuggestion.append(finalSuggestions[0])
				spellingCorrectionDetailsReplacementLength.append(len(finalSuggestions[0].split()))

		sentenceMisspelling = sentence
		#print('\nRaw:', sentenceMisspelling)

		for i, word in enumerate(collectMisspellingDetails['Word']):

			replacementLength = spellingCorrectionDetailsReplacementLength[i]
			# if there is no suggested replacement
			if replacementLength == 'NA':
				replacementLength = 1

			sentenceMisspelling = re.sub('(?<=[^a-zA-Z0-9])' + word + '(?![a-zA-Z0-9])', ' '.join([collectMisspellingDetails['Substitution'][i]] * replacementLength), sentenceMisspelling, count = 1)

		MisspellingRaw = ' '.join(spaCyTOK(sentenceMisspelling)).lower()

		Misspelling = re.sub('ABCMISSPELLING[0-9]+XYZ'.lower(), 'MISSPELLING', MisspellingRaw)

		if provideMisspellingDetailed == True:

			MisspellingDetailed = MisspellingRaw

			for i, word in enumerate(collectMisspellingDetails['Word']):

				replacementLength = spellingCorrectionDetailsReplacementLength[i]
				# if there is no suggested replacement
				if replacementLength == 'NA':
					replacementLength = 1

				MisspellingDetailed = MisspellingDetailed.replace(collectMisspellingDetails['Substitution'][i].lower(), collectMisspellingDetails['SubstitutionText'][i], replacementLength)


			MisspellingDetailed = MisspellingDetailed

		#print('\nMISSPELLING Representation:', Misspelling)
		#print('\nMISSPELLINGDETAILED Representation:', MisspellingDetailed)

		if useSpellChecker:
			sentence = chkr.get_text()
			correctedSentence = sentence
		else:
			correctedSentence = chkr.get_text()

		#print('\nCorrected Sentence:', correctedSentence)

		checkLength = [
			len(spellingCorrectionDetailsSentences),
			len(spellingCorrectionDetailsWords),
			len(spellingCorrectionDetailsChangesWord),
			len(spellingCorrectionDetailsReplacementLength),
			len(spellingCorrectionDetailsSuggestions),
			len(spellingCorrectionDetailsChosenSuggestion)
			]

		if debug:
			print('correctionDF:', checkLength)

		if not all(x == checkLength[0] for x in checkLength):
			print('\nProblem detected with the following text (spellchecker):', '\n')
			print(sentence)
			print(spellingCorrectionDetailsSuggestions)
			print(spellingCorrectionDetailsChosenSuggestion)
			print(spellingCorrectionDetailsReplacementLength)

		correctionDF = pd.DataFrame({
			#'RawInput': spellingCorrectionDetailsSentences,
			'RawWord': spellingCorrectionDetailsWords,
			'ChangesWord': spellingCorrectionDetailsChangesWord,
			'ReplacementLength': spellingCorrectionDetailsReplacementLength,
			'Suggestions': spellingCorrectionDetailsSuggestions,
			'ChosenSuggestion': spellingCorrectionDetailsChosenSuggestion
			})

		if debug:
			print('CorrectedSentence:', correctedSentence)
			print('CountStrippedWhitespaceChars:', whitespaceStripLength)
			print('CountStrippedHTMLChars:', htmlStripLength)
			print('CountSpellingCorrections', spellingCorrectionCount)
			print(correctionDF)

		resReturn = pd.DataFrame({
			'Sentence': sentence, 
			'Feature_Misspelling': Misspelling,
			'Spellchecker_CorrectedSentence': correctedSentence,
			'Spellchecker_CountStrippedWhitespaceChars': whitespaceStripLength,
			'Spellchecker_CountStrippedHTMLChars': htmlStripLength,
			'Spellchecker_CountSpellingCorrections': spellingCorrectionCount
			}, index = [0])

		if provideMisspellingDetailed:
			resReturn['Feature_MisspellingDetailed'] = MisspellingDetailed

		return([resReturn, correctionDF])

	# Basic Text Cleanup
	print('# Performing Basic Text Cleanup #\n')
	res = textData['InitialSentence'].mapply(BasicTextCleanup, debug = debug)
	resZip = list(zip(*res))

	textData = pd.concat([textData, pd.concat(resZip[0], ignore_index = True)], axis = 1)
	corrections = pd.concat(resZip[1], ignore_index = True)
	
	# Process Text with spaCy
	print('\n# Processing Text Representations #\n')
	def ProcessText (doc, debug = debug):
		"""_summary_

		:param doc: _description_
		:type doc: _type_
		:param debug: _description_, defaults to debug
		:type debug: _type_, optional
		"""

		doc = nlp(doc)

		all_word = []
		all_word_lower = []
		all_pos = []
		all_word_pos = []
		all_ner = []
		all_word_ner = []
		all_bounds = []

		for token in doc:

			word = token.text
			pos = token.pos_

			all_word.append(word)
			all_word_lower.append(token.lower_)
			all_pos.append(pos)
			all_word_pos.append(token.lower_ + '|_|' + pos)

			if token.ent_iob_ == "O":
				ner = token.lower_
				all_word_ner.append(token.lower_)
			else:
				ner = token.ent_type_
				all_word_ner.append(token.lower_ + '|_|' + token.ent_type_)

			all_ner.append(ner)

		sents = doc.sents

		for eachSent in sents:
			sentBounds = ['-'] * len([token.text for token in eachSent])
			sentBounds[-1] = 'S'
			all_bounds += sentBounds

		all_bounds = np.array(all_bounds)
		all_bounds[np.where(np.array(all_word) == '|||')] = 'D'

		# Vars
		Word        = all_word_lower
		POS         = all_pos
		Word_POS    = all_word_pos
		NER         = all_ner
		Word_NER    = all_word_ner
		Boundaries  = all_bounds

		# Word Sense Disambiguation
		tempWS = disambiguate(' '.join(all_word), algorithm = adapted_lesk, tokenizer = splitWS)
		tempWSRaw = [x[1] for x in tempWS]

		# Hypernym, Sentiment, Affect
		Hypernym = []
		Sentiment = []
		Affect = []
		Word_Sense = []

		# for WNAffect
		POSTreeBank = nltk.pos_tag(all_word)

		for i, each in enumerate(Word):

			try:
				wnaRes = str(wna.get_emotion(Word[i], POSTreeBank[i][1]).get_level(wnaReturnLevel))
				Affect.append(wnaRes.upper())
			except:
				Affect.append(Word[i])

			if (str(tempWSRaw[i]) != 'None'):

				Word_Sense.append(Word[i] + '|_|' + tempWS[i][1].name().split('.')[-1:][0])

				hypernyms = tempWS[i][1].hypernyms()

				if len(hypernyms) > 0:
					Hypernym.append(hypernyms[0].name().split('.')[0].upper())
				else:
					Hypernym.append(Word[i])

				swnScores = swn.senti_synset(tempWS[i][1].name())

				wordSentiment = ''

				if swnScores.pos_score() > 2/3:
					wordSentiment += 'HPOS'
				elif swnScores.pos_score() > 1/3:
					wordSentiment += 'MPOS'
				else:
					wordSentiment += 'LPOS'

				if swnScores.neg_score() > 2/3:
					wordSentiment += 'HNEG'
				elif swnScores.neg_score() > 1/3:
					wordSentiment += 'MNEG'
				else:
					wordSentiment += 'LNEG'

				Sentiment.append(wordSentiment)

			else:
				Word_Sense.append(Word[i])
				Hypernym.append(Word[i])
				Sentiment.append(Word[i])

		res = {
			'Feature_Word': all_word_lower,
			'Feature_POS': all_pos,
			'Feature_Word&POS': all_word_pos,
			'Feature_NER': all_ner,
			'Feature_Word&NER': all_word_ner,
			'Feature_Boundaries': all_bounds,
			'Feature_Affect': Affect,
			'Feature_Word&Sense': Word_Sense,
			'Feature_Hypernym': Hypernym,
			'Feature_Sentiment': Sentiment,
			}
		
		# Generate separate lexicon features (if available)
		LexiconFeatures = {}

		if lexicons is not None:

			for lexicon, tagTokenPairs in lexicons.items():

				lexiconName = 'Feature_Lexicon' + lexicon.upper()
				LexiconFeatures[lexiconName] = []

				for i, word in enumerate(Word):

					LexiconFeatures[lexiconName].append(word)
					wordReplaced = False

					for tag, tokens in tagTokenPairs.items():
						if wordReplaced:
							break
						elif any('*' in s for s in tokens):
							# regex mode
							nonmatching = [s for s in tokens if not s.endswith('*')]
							if word.lower() in nonmatching:
								LexiconFeatures[lexiconName][i] = tag.upper()
								wordReplaced = True
							else:
								matching = [s for s in tokens if s.endswith('*')]
								for eachToken in matching:
									startString = eachToken[:-1]
									startStringUnique = set(startString)
									if startStringUnique != set('*'):
										if word.lower().startswith(startString):
											LexiconFeatures[lexiconName][i] = tag.upper()
											matchedWord = True
									else:
										if eachToken == word.lower():
											LexiconFeatures[lexiconName][i] = tag.upper()
											matchedWord = True

						elif word.lower() in tokens:

							LexiconFeatures[lexiconName][i] = tag.upper()
							wordReplaced = True

		if lexicons is not None:
			res.update(LexiconFeatures)

		checkLength = [len(res[each]) for each in res]

		if len(set(checkLength)) != 1:
			print('Check Length:', checkLength)
			print('Problem detected with the following text:')
			print(sentence)

		# Rejoin features
		for each in res.keys():
			res[each] = ' '.join(res[each])

		return(res)

	res = textData['Sentence'].mapply(ProcessText)
	textData = pd.concat([textData, pd.DataFrame(res.values.tolist())], axis = 1)

	return([textData, corrections])

def TextToFeaturesReader (sentenceList, debug = False, inputLimit = False, lexicons = None, maxCores = False, wnaReturnLevel = 5, useSpellChecker = False, provideMisspellingDetailed = False, outputFileName = 'output', output_path = None):
	"""_summary_

	:param sentenceList: _description_
	:type sentenceList: _type_
	:param debug: _description_, defaults to False
	:type debug: bool, optional
	:param inputLimit: _description_, defaults to False
	:type inputLimit: bool, optional
	:param lexicons: _description_, defaults to None
	:type lexicons: _type_, optional
	:param maxCores: _description_, defaults to False
	:type maxCores: bool, optional
	:param wnaReturnLevel: _description_, defaults to 5
	:type wnaReturnLevel: int, optional
	:param useSpellChecker: _description_, defaults to False
	:type useSpellChecker: bool, optional
	:param provideMisspellingDetailed: _description_, defaults to False
	:type provideMisspellingDetailed: bool, optional
	:param outputFileName: _description_, defaults to 'output'
	:type outputFileName: str, optional
	:param output_path: _description_, defaults to None
	:type output_path: _type_, optional
	"""

	# if output_path is None, set it to the current working directory
	if output_path is None:
		output_path = os.getcwd()
	else:
		# Check if the provided path exists
		if not os.path.isdir(output_path):
			print(f"The provided path {output_path} is not valid. Saving the file in the current directory instead.")
			output_path = os.getcwd()
		else:
			print(f"Saving the file in {output_path}")

	if (inputLimit == 0):
		inputLimit = len(sentenceList)
	elif (inputLimit > len(sentenceList)):
		inputLimit = len(sentenceList)
		
	if (len(sentenceList) == 0):
		print("No rows in input data! Terminating...", '\n')
		quit()

	processRows = min(len(sentenceList), inputLimit)
	print('Items to Process:', processRows, '\n')

	print('# Now Processing Text Items #', '\n')

	start = datetime.now(get_localzone())

	processedText, corrections = TextToFeatures(sentenceList[:inputLimit], debug = False, lexicons = lexicons, wnaReturnLevel = wnaReturnLevel, useSpellChecker = useSpellChecker, provideMisspellingDetailed = provideMisspellingDetailed, useCores = useCores)

	print('\nItems Processed: ' + str(len(processedText)) + ' (Time Elapsed: {})\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

	with open(os.path.join(output_path, outputFileName + '_raw_representations.pickle'), "wb") as f:
		pickle.dump({'ProcessedText': processedText, 'Corrections': corrections}, f, pickle.HIGHEST_PROTOCOL)

	return({'ProcessedText': processedText, 'Corrections': corrections})

def RunFeatureConstruction (fullinputpath = None, inputLimit = False, outputpath = None, outputFileName = 'output', maxCores = False, lexiconFileFullPath = None, wnaReturnLevel = 5, useSpellChecker = False, provideMisspellingDetailed = False):
	"""_summary_

	:param fullinputpath: _description_, defaults to None
	:type fullinputpath: _type_, optional
	:param inputLimit: _description_, defaults to False
	:type inputLimit: bool, optional
	:param outputpath: _description_, defaults to None
	:type outputpath: _type_, optional
	:param outputFileName: _description_, defaults to 'output'
	:type outputFileName: str, optional
	:param maxCores: _description_, defaults to False
	:type maxCores: bool, optional
	:param lexiconFileFullPath: _description_, defaults to None
	:type lexiconFileFullPath: _type_, optional
	:param wnaReturnLevel: _description_, defaults to 5
	:type wnaReturnLevel: int, optional
	:param useSpellChecker: _description_, defaults to False
	:type useSpellChecker: bool, optional
	:param provideMisspellingDetailed: _description_, defaults to False
	:type provideMisspellingDetailed: bool, optional
	"""

	lexicons = ReadAllLexicons(lexiconFileFullPath)

	print('# Now Reading Raw Data #')

	res = ReadRawText(fullinputpath)
	rawTextData = res['corpus']
	classLabels = res['classLabels']

	# Run a single test on a specific row:
	# TextToFeatures(raw[4-1], debug = True, lexicons = lexicons); quit() # 1357 in dvd.txt includes spanish; 4 in modified dvd_issue.txt

	output = TextToFeaturesReader(rawTextData, inputLimit = inputLimit, lexicons = lexicons, maxCores = maxCores, wnaReturnLevel = wnaReturnLevel, useSpellChecker = useSpellChecker, provideMisspellingDetailed = provideMisspellingDetailed)

	end_time = datetime.now(get_localzone())
	end_time_str = str(end_time.strftime(fmt))
	print('### Stage execution finished at ' + end_time_str + ' (Time Elapsed: {})'.format(pd.to_timedelta(end_time - start_time).round('1s')) + ' ###\n')

def ConstructLegomena (corpus, debug = False):
	"""_summary_

	:param corpus: _description_
	:type corpus: _type_
	:param debug: _description_, defaults to False
	:type debug: bool, optional
	"""

	vectorizerLegomenaHapax = CountVectorizer(
		ngram_range = (1, 1),
		analyzer = 'word',
		tokenizer = None,
		preprocessor = None,
		stop_words = None,
		token_pattern = r'\S+',
		max_features = None,
		lowercase = False,
		min_df = 1,
		max_df = 1,
		dtype = np.uint8)

	vectorizerLegomenaDis = CountVectorizer(
		ngram_range = (1, 1),
		analyzer = 'word',
		tokenizer = None,
		preprocessor = None,
		stop_words = None,
		token_pattern = r'\S+',
		max_features = None,
		lowercase = False,
		min_df = 2,
		max_df = 2,
		dtype = np.uint8)

	legomenaVocab = {'HAPAX': [], 'DIS': []}

	for label, vectorizer in {'HAPAX': vectorizerLegomenaHapax, 'DIS': vectorizerLegomenaDis}.items():

		try:
			train_data_features = vectorizer.fit_transform(corpus)
			train_data_features = train_data_features.toarray()
			vocab = vectorizer.get_feature_names()
			legomenaVocab[label] = vocab
		except:
			print('# Warning: No ' + label.lower() + ' legomena were found. #', '\n')

	legomenaDF = pd.DataFrame(corpus)

	def word_subber (item):
		legomena = []
		for word in item[0].split(' '):
			if word in legomenaVocab['HAPAX']:
				legomena.append('HAPAX')
			elif word in legomenaVocab['DIS']:
				legomena.append('DIS')
			else:
				legomena.append(word)
		return(' '.join(legomena))

	legomenaDF = legomenaDF.mapply(word_subber, axis = 1).to_frame(name = 'Legomena')

	return(legomenaDF)

def BuildFeatureVector (data, vectorizer, vectorizerName, feature, debug = False):
	"""_summary_

	:param data: _description_
	:type data: _type_
	:param vectorizer: _description_
	:type vectorizer: _type_
	:param vectorizerName: _description_
	:type vectorizerName: _type_
	:param feature: _description_
	:type feature: _type_
	:param debug: _description_, defaults to False
	:type debug: bool, optional
	"""

	# Using standard scikit vectorizers. For custom analyzer, see http://stackoverflow.com/questions/26907309/create-ngrams-only-for-words-on-the-same-line-disregarding-line-breaks-with-sc

	train_data_features = vectorizer.fit_transform( data )
	#train_data_features = train_data_features.toarray()

	names = vectorizer.get_feature_names()

	#debug = True

	if feature == 'Misspelling' and debug == True:
		print('### ' + feature + ' ###')
		print(vectorizerName)
		print(data, '\n')
		#print(names, '\n')
		#print(vectorizer.vocabulary_)

		vocab = vectorizer.get_feature_names()
		print(vocab)

		# Sum up the counts of each vocabulary word
		dist = np.sum(train_data_features, axis=0)

		# For each, print the vocabulary word and the number of times it
		# appears in the training set
		for tag, count in zip(vocab, dist):
			print(count, tag)

	for i, name in enumerate(names):
		names[i] = vectorizerName.upper() + '|~|' + re.sub(' ', '', feature.upper()) + '|~|' + re.sub(' ', '|-|', name)

	#df = pd.DataFrame(train_data_features, columns = names)
	df = pd.DataFrame.sparse.from_spmatrix(train_data_features, columns = names)

	if debug:
		print(df)

	return(df)

def VectorProcessor (data, maxNgram = 3, vader = False, maxFeatures = None, buildVectors = 'b', removeZeroVariance = True, combineFeatures = False, minDF = 5, removeDupColumns = False, classLabels = False, runLegomena = True, additionalCols = False, writeRepresentations = False, justRepresentations = False, outputpath = None, debug = False, outputFileName = 'output'):
	"""_summary_

	:param data: _description_
	:type data: _type_
	:param maxNgram: _description_, defaults to 3
	:type maxNgram: int, optional
	:param vader: _description_, defaults to False
	:type vader: bool, optional
	:param maxFeatures: _description_, defaults to None
	:type maxFeatures: _type_, optional
	:param buildVectors: _description_, defaults to 'b'
	:type buildVectors: str, optional
	:param removeZeroVariance: _description_, defaults to True
	:type removeZeroVariance: bool, optional
	:param combineFeatures: _description_, defaults to False
	:type combineFeatures: bool, optional
	:param minDF: _description_, defaults to 5
	:type minDF: int, optional
	:param removeDupColumns: _description_, defaults to False
	:type removeDupColumns: bool, optional
	:param classLabels: _description_, defaults to False
	:type classLabels: bool, optional
	:param runLegomena: _description_, defaults to True
	:type runLegomena: bool, optional
	:param additionalCols: _description_, defaults to False
	:type additionalCols: bool, optional
	:param writeRepresentations: _description_, defaults to False
	:type writeRepresentations: bool, optional
	:param justRepresentations: _description_, defaults to False
	:type justRepresentations: bool, optional
	:param outputpath: _description_, defaults to None
	:type outputpath: _type_, optional
	:param debug: _description_, defaults to False
	:type debug: bool, optional
	:param outputFileName: _description_, defaults to 'output'
	:type outputFileName: str, optional
	"""

	if outputpath is None or not os.path.isdir(outputpath):
		print("Output path not provided or invalid. Using current directory instead.")
		outputpath = os.getcwd()
	else:
		# Convert outputpath to absolute path
		outputpath = os.path.abspath(outputpath)

	dataRows = len(data)

	print ('# Settings #')

	if maxFeatures == 0:
		maxFeatures = None
		min_df = minDF
	else:
		min_df = minDF

	if min_df > dataRows:
		print('Warning: minDF setting was lower than the number of items. Set to 0.0!')
		min_df = 0.0
	else:
		print('Minimum Term Frequency:', min_df)

	if (dataRows == 1):
		print('Warning: The data consist of a single row, so Legomena, Remove Zero Variance, and Remove Duplicate Columns were disabled!')
		removeZeroVariance = False
		removeDupColumns = False
		runLegomena = False

	print('N-grams:', maxNgram)

	vectorizerTfidf = TfidfVectorizer(
		ngram_range = (1, maxNgram),
		sublinear_tf=True,
		analyzer = 'word',
		tokenizer = None,
		preprocessor = None,
		stop_words = None,
		token_pattern = r'\S+',
		max_features = maxFeatures,
		lowercase = False,
		min_df = min_df,
		dtype = np.float64) # maybe use float32?

	vectorizerCount = CountVectorizer(
		ngram_range = (1, maxNgram),
		analyzer = 'word',
		tokenizer = None,
		preprocessor = None,
		stop_words = None,
		token_pattern = r'\S+',
		max_features = maxFeatures,
		lowercase = False,
		min_df = min_df,
		dtype = np.uint32)

	vectorizerCharCount = CountVectorizer(
		ngram_range = (1, maxNgram),
		analyzer = 'char_wb',
		tokenizer = None,
		preprocessor = None,
		stop_words = None,
		#token_pattern = r'\S+',
		max_features = maxFeatures,
		lowercase = False,
		min_df = min_df,
		dtype = np.uint32)

	vectorizerBinary = CountVectorizer(
		ngram_range = (1, maxNgram),
		analyzer = 'word',
		tokenizer = None,
		preprocessor = None,
		stop_words = None,
		token_pattern = r'\S+',
		max_features = maxFeatures,
		lowercase = False,
		min_df = min_df,
		binary = True,
		dtype = np.uint8)

	vectorizerCharBinary = CountVectorizer(
		ngram_range = (1, maxNgram),
		analyzer = 'char_wb',
		tokenizer = None,
		preprocessor = None,
		stop_words = None,
		#token_pattern = r'\S+',
		max_features = maxFeatures,
		lowercase = False,
		min_df = min_df,
		binary = True,
		dtype = np.uint8)

	buildVectors = list(buildVectors)

	chosenVectorizers = {'vectorizers': [], 'names': []}

	for option in buildVectors:
		if option == 't':
			chosenVectorizers['vectorizers'].append(vectorizerTfidf)
			chosenVectorizers['names'].append('tfidf')
		elif option == 'c':
			chosenVectorizers['vectorizers'].append(vectorizerCount)
			chosenVectorizers['names'].append('count')
		elif option == 'b':
			chosenVectorizers['vectorizers'].append(vectorizerBinary)
			chosenVectorizers['names'].append('binary')
		elif option == 'C':
			chosenVectorizers['vectorizers'].append(vectorizerCharCount)
			chosenVectorizers['names'].append('charcount')
		elif option == 'B':
			chosenVectorizers['vectorizers'].append(vectorizerCharBinary)
			chosenVectorizers['names'].append('charbinary')

	print('Requested Feature Vectors:', chosenVectorizers['names'])

	# Build additional features that can only be done after basic feature generation (right now just legomena)
	legomena = []

	if runLegomena:
		print('\n# Adding Legomena Feature #\n')
		try:
			legomena = ConstructLegomena(data['Word'], debug = False)
			data = pd.concat([data, legomena], axis = 1)
		except:
			print('Warning: There was an error generating legomena features...')
		
		print('\n')

	# Combine parallel features if needed (CHECK ME OR REMOVE!)
	combos = []
	'''
	if combineFeatures:

		combos = FeatureCombiner(data)
		#print(len(data))
		#print(len(combos))
		data = {**data, **combos}
		#print(len(data))

	###
	'''

	# Evaluate final set of features
	processedFeatures = data.sort_index(axis = 1)
	print('# Final Set of Feature Representations (' + str(len(processedFeatures)) + ' Total) #')
	print(processedFeatures.columns.tolist(), '\n')

	# Write representations to disk (if requested)
	if writeRepresentations:
		print('# Now Writing Representations to Disk #')

		start = datetime.now(get_localzone())

		# Compress features
		repArchive = os.path.join(outputpath, outputFileName + '_representations.zip')
		try:
			os.remove(repArchive)
		except OSError:
			pass
		z = zf.ZipFile(repArchive, 'a')

		for feature in processedFeatures:
			repFile = os.path.join(outputpath, outputFileName + '_representation_' + feature + '.txt')

			sentenceWriter = open(repFile, 'w', encoding = 'utf-8')
			for each in processedFeatures[feature]:
				sentenceWriter.write(each + '\n')
			sentenceWriter.close()
			z.write(repFile, os.path.basename(repFile))
			os.remove(repFile)

		z.close()

		print('- Time Elapsed: {}\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

	if justRepresentations:

		end_time = datetime.now(get_localzone())
		end_time_str = str(end_time.strftime(fmt))
		print('### Stage execution finished at ' + end_time_str + ' (Time Elapsed: {})'.format(pd.to_timedelta(end_time - start_time).round('1s')) + ' ###\n')
		quit()

	else:
	
		print('# Now Generating Feature Matrices #', '\n')

		featureFiles = []
		
		for i, vectorizer in enumerate(chosenVectorizers['vectorizers']):

			# only run character n-grams on Word feature
			if 'char' in chosenVectorizers['names'][i].lower():
				start = datetime.now(get_localzone())

				print('\n# Adding Character N-grams (' + chosenVectorizers['names'][i].lower() + '-' + 'Word' + ') #')

				tempDF = BuildFeatureVector(data['Word'], chosenVectorizers['vectorizers'][i], chosenVectorizers['names'][i], 'Word', False)

				#tempDF = tempDF.loc[:, ~tempDF.mapply(vector_hasher).duplicated()]
				#tempDF = tempDF.loc[:, ~(tempDF.mapply(np.var) == 0)]

				fileLoc = os.path.join(outputpath, outputFileName + '_' + re.sub('&', '_', chosenVectorizers['names'][i] + '_' + 'Word') + '_feature_matrix.pickle')
				with open(fileLoc, "wb") as f:
					pickle.dump(tempDF, f, pickle.HIGHEST_PROTOCOL)
				featureFiles.append(fileLoc)

				print('Features: ' + str(len(tempDF.columns)) + ' (Time Elapsed: {})'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))
				
				del(tempDF)
			else:
				for j, feature in enumerate(processedFeatures):
					start = datetime.now(get_localzone())

					print('---\n' + feature)

					tempDF = BuildFeatureVector(data[feature], chosenVectorizers['vectorizers'][i], chosenVectorizers['names'][i], feature, False)

					#tempDF = tempDF.loc[:, ~tempDF.mapply(vector_hasher).duplicated()]
					#tempDF = tempDF.loc[:, ~(tempDF.mapply(np.var) == 0)]

					fileLoc = os.path.join(outputpath, outputFileName + '_' + re.sub('&', '_', chosenVectorizers['names'][i] + '_' + feature) + '_feature_matrix.pickle')
					with open(fileLoc, "wb") as f:
						pickle.dump(tempDF, f, pickle.HIGHEST_PROTOCOL)

					# place Word feature at the front
					if feature == 'Word':
						featureFiles.insert(0, fileLoc)
					else:
						featureFiles.append(fileLoc)

					print('Features: ' + str(len(tempDF.columns)) + ' (Time Elapsed: {})'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

					del(tempDF)

		# clean up memory
		del data
		del processedFeatures
		del legomena
		del combos

		gc.collect()

		print('\n# Now Joining Feature Matrices #', '\n')

		# join df from individual feature matrix files
		for i, eachFile in enumerate(featureFiles):
			start = datetime.now(get_localzone())
			if i == 0:
				with open(eachFile, "rb") as f:
					df = pickle.load(f)
			else:
				with open(eachFile, "rb") as f:
					df = pd.concat([df, pickle.load(f)], axis = 1)
				#df = df.loc[:, ~df.mapply(vector_hasher).duplicated()]
			print('Processed ' + os.path.splitext(os.path.basename(eachFile))[0] + ' (Time Elapsed: {})'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

		print('\nNumber of Features Produced:', len(df.columns), '\n')

		# Remove zero variance
		if removeZeroVariance:

			start = datetime.now(get_localzone())

			lenPreRemoveZV = len(df.columns)

			df = df.loc[:, ~(df.mapply(np.var) == 0)]

			removedCols = lenPreRemoveZV - len(df.columns)

			print('Number of Zero Variance Features Removed: ' + str(removedCols) + ' (Time Elapsed: {})\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

		# Remove duplicate columns
		if removeDupColumns:

			start = datetime.now(get_localzone())
			
			dfStart = df.columns

			df = df.loc[:, ~df.mapply(vector_hasher).duplicated()]

			dfFinish = df.columns
			dups = dfStart.difference(dfFinish)

			print('Number of Duplicate Features Removed: ' + str(len(dups)) + ' (Time Elapsed: {})\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

		# Add class labels
		if type(classLabels) != bool:
			classLabels = pd.DataFrame({'Class': classLabels[:dataRows]})
			df = pd.concat([classLabels, df], axis = 1)

		# Add VADER
		if type(vader) != bool:
			df = pd.concat([df, vader], axis = 1)

		# Add additional columns
		if type(additionalCols) != bool:
			df = pd.concat([df, additionalCols], axis = 1)

		return(df)

def ResultWriter (df, outputpath = None, outputFileName = 'output', index = False, header = False, compression = None):
	"""_summary_

	:param df: _description_
	:type df: _type_
	:param outputpath: _description_, defaults to None
	:type outputpath: _type_, optional
	:param outputFileName: _description_, defaults to 'output'
	:type outputFileName: str, optional
	:param index: _description_, defaults to False
	:type index: bool, optional
	:param header: _description_, defaults to False
	:type header: bool, optional
	:param compression: _description_, defaults to None
	:type compression: _type_, optional
	"""

	if outputpath is None or not os.path.isdir(outputpath):
		print("Output path not provided or invalid. Using current directory instead.")
		outputpath = os.getcwd()
	else:
		# Convert outputpath to absolute path
		outputpath = os.path.abspath(outputpath)

	start = datetime.now(get_localzone())

	#print(df)
	if index:
		df.index += 1
		df.index.name = 'Index'

	# this is extremely slow and needs to be improved
	df.to_csv(os.path.join(outputpath, outputFileName + '.csv'), index = index, header = header, sep = ',', chunksize = 2000, compression = compression)

	print('- Time Elapsed: {}\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

def runVader (sentenceList, inputLimit):
	"""_summary_

	:param sentenceList: _description_
	:type sentenceList: _type_
	:param inputLimit: _description_
	:type inputLimit: _type_
	"""

	if (inputLimit == 0):
		inputLimit = len(sentenceList)

	if (len(sentenceList) == 0):
		print("No rows in input data! Terminating...", '\n')
		quit()

	sid = SentimentIntensityAnalyzer()

	processRows = min(len(sentenceList), inputLimit)

	neg = []
	pos = []
	neu = []
	compound = []

	for sentence in sentenceList[:processRows]:
		ss = sid.polarity_scores(sentence)
		neg.append(ss['neg'])
		pos.append(ss['pos'])
		neu.append(ss['neu'])
		compound.append(ss['compound'])

	vader = {'VaderNEG': neg, 'VaderPOS': pos, 'VaderNEU': neu, 'VaderCOMPOUND': compound}
	vaderDF = pd.DataFrame(vader, columns = list(dict.keys(vader)))

	return(vaderDF)

def GenerateColumnKey(df, outputpath = None, outputFileName = 'output'):
	"""_summary_

	:param df: _description_
	:type df: _type_
	:param outputpath: _description_, defaults to None
	:type outputpath: _type_, optional
	:param outputFileName: _description_, defaults to 'output'
	:type outputFileName: str, optional
	"""

	# |~| separates vectorizer, category, and feature (in that order); always 2 in label (e.g., BINARY|~|WORD|~|hello)
	# |-| replaces spaces within features from higher order n-grams, e.g., "the|-|cat|-|jumped" (3-gram); this also applies to character n-grams that include spaces, e.g., g|-|a == 'g a'
	# |_| indicates a composite feature was generated, e.g., WordPOS of cat|_|NN
	# |&| indicates a category is a two-way combo, e.g., POS|&|HYPERNYM
	# |+| indicates a combo composite feature was formed, e.g., NN|+|CANINE based on the Word 'dog'
	# _ can appear as part of a substitution (e.g., the hypernym for style, EXPRESSIVE_STYLE)
	# category names with spaces (e.g., from lexicon file names) will have their white space stripped
	# original words are in all lower case; substituted word tags are in all caps (e.g., POSITIVE, NEUTRAL), as are the latter half of word composites (e.g., dog_NN, dog_CANINE, keith_PERSON)

	start = datetime.now(get_localzone())

	if outputpath is None or not os.path.isdir(outputpath):
		print("Output path not provided or invalid. Using current directory instead.")
		outputpath = os.getcwd()
	else:
		# Convert outputpath to absolute path
		outputpath = os.path.abspath(outputpath)

	# calculate column sums for key output
	colSums = df.values.sum(axis = 0).astype('str')

	# full version (f1) and GBS (f2)
	f1 = open(os.path.join(outputpath, outputFileName + '_key.txt'), 'w', encoding = 'utf-8')
	f2 = open(os.path.join(outputpath, outputFileName + '_key_GBS.txt'), 'w', encoding = 'utf-8')

	for i, column in enumerate(df.columns):

		column = str(column)

		if column.startswith('Vader') or column.startswith('Count') or column == 'Class' or column == 'Index':
			f1.write(column + '\t' + 'NA' + '\t' + 'NA' + '\t' + 'NA' + '\t' + 'NA' + '\n')
			f2.write('NA' + '\t' + 'NA-NA' + '\t' + 'NA' + '\n')
		else:
			#print(column)
			colSplit = column.split('|~|')
			#print(colSplit)
			vectorizerName = colSplit[0]
			categoryName = colSplit[1]
			feature = colSplit[2]

			if 'CHAR' in vectorizerName.upper():
				feature = list(re.sub('\|-\|', ' ', feature))
				categoryName = 'CHAR' #vectorizerName
			else:
				feature = feature.split('|-|')

			#print(feature, len(feature))

			f1.write(column + '\t' + vectorizerName + '\t' + categoryName + '\t' + ' '.join(feature) + '\t' + str(len(feature)) + '-gram' + '\n')

			# modify GBS features to remove instances of |_| and replace with _
			feature = [re.sub('\|_\|', '_', x) for x in feature]

			f2.write(' '.join(feature) + '\t' + str(len(feature)) + '-' + categoryName + '\t' + colSums[i] + '\n')

	f1.close()
	f2.close()

	print('- Time Elapsed: {}\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

def RunPostFeatureConstruction (lexiconpath, fullinputpath, inputLimit, outputpath = None, outputFileName = 'output', maxCores = False, maxNgram = 3, lexiconFileFullPath = False, vader = False, wnaReturnLevel = 5, maxFeatures = 50, buildVectors = 'b', index = False, removeZeroVariance = True, combineFeatures = False, minDF = 5, removeDupColumns = False, useSpellChecker = False, provideMisspellingDetailed = False, additionalCols = False, writeRepresentations = False, justRepresentations = False):
	"""_summary_

	:param lexiconpath: _description_
	:type lexiconpath: _type_
	:param fullinputpath: _description_
	:type fullinputpath: _type_
	:param inputLimit: _description_
	:type inputLimit: _type_
	:param outputpath: _description_, defaults to None
	:type outputpath: _type_, optional
	:param outputFileName: _description_, defaults to 'output'
	:type outputFileName: str, optional
	:param maxCores: _description_, defaults to False
	:type maxCores: bool, optional
	:param maxNgram: _description_, defaults to 3
	:type maxNgram: int, optional
	:param lexiconFileFullPath: _description_, defaults to False
	:type lexiconFileFullPath: bool, optional
	:param vader: _description_, defaults to False
	:type vader: bool, optional
	:param wnaReturnLevel: _description_, defaults to 5
	:type wnaReturnLevel: int, optional
	:param maxFeatures: _description_, defaults to 50
	:type maxFeatures: int, optional
	:param buildVectors: _description_, defaults to 'b'
	:type buildVectors: str, optional
	:param index: _description_, defaults to False
	:type index: bool, optional
	:param removeZeroVariance: _description_, defaults to True
	:type removeZeroVariance: bool, optional
	:param combineFeatures: _description_, defaults to False
	:type combineFeatures: bool, optional
	:param minDF: _description_, defaults to 5
	:type minDF: int, optional
	:param removeDupColumns: _description_, defaults to False
	:type removeDupColumns: bool, optional
	:param useSpellChecker: _description_, defaults to False
	:type useSpellChecker: bool, optional
	:param provideMisspellingDetailed: _description_, defaults to False
	:type provideMisspellingDetailed: bool, optional
	:param additionalCols: _description_, defaults to False
	:type additionalCols: bool, optional
	:param writeRepresentations: _description_, defaults to False
	:type writeRepresentations: bool, optional
	:param justRepresentations: _description_, defaults to False
	:type justRepresentations: bool, optional
	"""

	#print(maxCores)
	if outputpath is None or not os.path.isdir(outputpath):
		print("Output path not provided or invalid. Using current directory instead.")
		outputpath = os.getcwd()
	else:
		# Convert outputpath to absolute path
		outputpath = os.path.abspath(outputpath)

	print('# Now Reading Raw Data #')

	res = ReadRawText(fullinputpath)
	rawTextData = res['corpus']
	classLabels = res['classLabels']

	print('\n# Now Reading Feature Data Pickle #')
	start = datetime.now(get_localzone())

	with open(os.path.join(outputpath, outputFileName + '_raw_representations.pickle'), "rb") as f:
		output = pickle.load(f)

	processedText = output['ProcessedText']
	corrections = output['Corrections']

	print('- Time Elapsed: {}\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

	print('# Now Writing Spellchecked Sentences to Disk #')
	start = datetime.now(get_localzone())

	sentenceWriter = open(os.path.join(outputpath, outputFileName + '_cleaned_sentences.txt'), 'w', encoding = 'utf-8')
	for i, cleanedSentence in enumerate(processedText['Spellchecker_CorrectedSentence']):
		sentenceWriter.write(classLabels[i] + '\t' + cleanedSentence + '\n')
	sentenceWriter.close()
	processedText = processedText.drop(columns = 'Spellchecker_CorrectedSentence')

	print('- Time Elapsed: {}\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

	print('# Now Writing Spelling Corrections to Disk #')
	ResultWriter(corrections, outputpath, outputFileName + '_spelling_corrections', index = False, header = True)

	if additionalCols:
		additionalCols = processedText.loc[:, processedText.columns.str.startswith('Spellchecker_')]
		additionalCols.columns = additionalCols.columns.str.lstrip('Spellchecker_')
	else:
		additionalCols = False

	if vader:
		print('# Now Generating VADER Scores #')
		start = datetime.now(get_localzone())
		vader = runVader(rawTextData, inputLimit)
		print('- Time Elapsed: {}\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))
	else:
		vader = False

	print('# Now Constructing Feature Vectors #', '\n')

	representations = processedText.loc[:, processedText.columns.str.startswith('Feature_')]
	representations.columns = representations.columns.str.lstrip('Feature_')
	df = VectorProcessor(representations, maxNgram = maxNgram, vader = vader, maxFeatures = maxFeatures, buildVectors = buildVectors, removeZeroVariance = removeZeroVariance, combineFeatures = combineFeatures, minDF = minDF, removeDupColumns = removeDupColumns, classLabels = classLabels, additionalCols = additionalCols, writeRepresentations = writeRepresentations, justRepresentations = justRepresentations)

	print('# Now Writing Results to Disk #')
	ResultWriter(df, outputpath, outputFileName, index = index, header = True)

	print('# Now Generating Column Key Files #')
	GenerateColumnKey(df, outputpath, outputFileName)

	end_time = datetime.now(get_localzone())
	end_time_str = str(end_time.strftime(fmt))
	print('Output Dimensions (Rows, Features):', df.shape, '\n\n### Execution finished at ' + end_time_str + ' (Time Elapsed: {})'.format(pd.to_timedelta(end_time - start_time).round('1s')) + ' ###\n')

###  Below is for ARFN  ###

import os
import sys
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
import re

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
thresh = 0.005
subThresh = 1.05
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

#semantic hashes
lex = np.zeros((50000,50000), dtype=object)
lexCt = np.zeros((50000),dtype=int)
hashlex = np.zeros((27,27,50000), dtype=object)
hashlexClust = np.zeros((27,27,50000),dtype=int)
hashlexCt = np.zeros((27,27),dtype=int)
lexTags = np.zeros((50000), dtype=object)


def HashLetters(strToken):
	"""_summary_

	:param strToken: _description_
	:type strToken: _type_
	:return: _description_
	:rtype: _type_
	"""

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

def ReadFeatures(featuresFile):
	"""_summary_

	:param featuresFile: _description_
	:type featuresFile: _type_
	"""

	global numFeat, foundNACt, foundNA, numLexFile, numCat, maxposword, flipposword, flipposwordCt, lexFile, cat, featureIndex, featureStr, featureCatStr
	if featuresFile is None or not os.path.isfile(featuresFile):
		print("Features file not provided or invalid. Terminating...", '\n')
		quit()
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

def ReadTrain(trainFile = None):
	"""_summary_

	:param trainFile: _description_, defaults to None
	:type trainFile: _type_, optional
	:return: _description_
	:rtype: _type_
	"""

	# global matrix, numFeat, foundNACt, foundNA, numLexFile, numCat, maxposword, flipposwordCt, lexFile, cat, numInst, classLabels, numClass,trainWeight, trainWeightC, instLabels
	global matrix, numFeat, foundNACt, foundNA, numLexFile, numCat, maxposword, flipposwordCt, lexFile, cat, numInst, classLabels, numClass,trainWeight, trainWeightC, instLabels
	if trainFile is None or not os.path.isfile(trainFile):
		print("Training file not provided or invalid. Using current directory instead.")
		return False
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

def AssignTrainWeights(trainFile = None):
	"""_summary_

	:param trainFile: _description_, defaults to None
	:type trainFile: _type_, optional
	"""

	global numFeat, foundNACt, foundNA, numLexFile, numCat, maxposword, flipposwordCt, lexFile, cat, numInst, classLabels, numClass,trainWeight, trainWeightC, instLabels
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
	"""_summary_
	"""

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
	"""_summary_
	"""

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

def NGramSemantic(word):
	"""_summary_

	:param word: _description_
	:type word: _type_
	:return: _description_
	:rtype: _type_
	"""

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

def AssignSemanticWeights():
	"""_summary_
	"""

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

def POSSemantic(word):
	"""_summary_

	:param word: _description_
	:type word: _type_
	:return: _description_
	:rtype: _type_
	"""

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
	"""_summary_

	:param word: _description_
	:type word: _type_
	:return: _description_
	:rtype: _type_
	"""

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
	"""_summary_

	:param word: _description_
	:type word: _type_
	:return: _description_
	:rtype: _type_
	"""

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

def MatchCharSubstrings(worda, c1, c2):
	"""_summary_

	:param worda: _description_
	:type worda: _type_
	:param c1: _description_
	:type c1: _type_
	:param c2: _description_
	:type c2: _type_
	:return: _description_
	:rtype: _type_
	"""

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
	"""_summary_

	:param worda: _description_
	:type worda: _type_
	:param c1: _description_
	:type c1: _type_
	:param c2: _description_
	:type c2: _type_
	:return: _description_
	:rtype: _type_
	"""

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

def SubsumeCatN(catVal,compVal,n1,n2):
	"""_summary_

	:param catVal: _description_
	:type catVal: _type_
	:param compVal: _description_
	:type compVal: _type_
	:param n1: _description_
	:type n1: _type_
	:param n2: _description_
	:type n2: _type_
	"""

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

def SubsumeFeatures(indexa, indexb, numM):
	"""_summary_

	:param indexa: _description_
	:type indexa: _type_
	:param indexb: _description_
	:type indexb: _type_
	:param numM: _description_
	:type numM: _type_
	"""

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

def RunSubsumptions():
	"""_summary_
	"""

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

def LoadHash(c, n, fStatus):
	"""_summary_

	:param c: _description_
	:type c: _type_
	:param n: _description_
	:type n: _type_
	:param fStatus: _description_
	:type fStatus: _type_
	"""

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

def RunCCSubsumptions():
	"""_summary_
	"""

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
	"""_summary_
	"""

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
	"""_summary_

	:param catVal: _description_
	:type catVal: _type_
	:param compVal: _description_
	:type compVal: _type_
	:param n1: _description_
	:type n1: _type_
	:param n2: _description_
	:type n2: _type_
	"""

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
	"""_summary_

	:param worda: _description_
	:type worda: _type_
	:param f: _description_
	:type f: _type_
	:param c1: _description_
	:type c1: _type_
	:param c2: _description_
	:type c2: _type_
	"""

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
	"""_summary_

	:param indexa: _description_
	:type indexa: _type_
	:param comp: _description_
	:type comp: _type_
	:param cat1: _description_
	:type cat1: _type_
	:param cat2: _description_
	:type cat2: _type_
	"""

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
	"""_summary_

	:param worda: _description_
	:type worda: _type_
	:param f: _description_
	:type f: _type_
	:param c1: _description_
	:type c1: _type_
	:param c2: _description_
	:type c2: _type_
	:param n: _description_
	:type n: _type_
	"""

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

def OutputRankings(weightFile):
	"""_summary_

	:param weightFile: _description_
	:type weightFile: _type_
	"""

	#global numFeat, featureCatStr, featureStr, trainWeight
	outFile = open(weightFile, "w")
	for b in range(0, numFeat):
		outFile.write(str(b+1)+"\t"+str(featureStr[b])+"\t"+str(featureCatStr[b]).strip("\n")+"\t"+str(trainWeight[b])+"\n")

def AFRN():
	"""_summary_
	"""
	
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




