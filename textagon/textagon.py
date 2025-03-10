# -*- coding: utf-8 -*-

###############
### IMPORTS ###
###############

#from externalFunctions import *

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

import warnings
warnings.filterwarnings('ignore', message = '.*looks like a URL.*', category = UserWarning, module = 'bs4')
warnings.filterwarnings('ignore', message = '.*The multilingual functions are not available with this Wordnet version*', category = UserWarning, module = 'nltk')


from utils import *


#####################
### CONFIGURATION ###
#####################


# time display settings
fmt = '%Y-%m-%d %H:%M %p %Z'
start_time = datetime.now(get_localzone())
start_time_str = str(start_time.strftime(fmt))

if __name__ == '__main__':
    print('### Execution started at ' + start_time_str + ' ###\n')


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

# SentiWN #
from nltk.corpus import sentiwordnet as swn
swn.ensure_loaded()


# lots of arguments here...
class Textagon:
    def __init__(
            self,
            inputFile,
            outputFileName, 
            inputLimit, 
            maxFeatures,
            maxNgram,
            maxCores,
            lexiconFileFullPath,
            vader,
            wnaReturnLevel,
            buildVectors,
            index,
            removeZeroVariance,
            combineFeatures,
            minDF,
            removeDupColumns,
            useSpellChecker,
            provideMisspellingDetailed,
            additionalCols,
            writeRepresentations,
            exclusionsFileFullPath,
            runType,
            justRepresentations
            ):
        self.inputFile = inputFile
        self.outputFileName = outputFileName
        self.inputLimit = inputLimit
        self.maxFeatures = maxFeatures
        self.maxNgram = maxNgram
        self.maxCores = maxCores
        self.lexiconFileFullPath = lexiconFileFullPath
        self.vader = vader
        self.wnaReturnLevel = wnaReturnLevel
        self.buildVectors = buildVectors
        self.index = index
        self.removeZeroVariance = removeZeroVariance
        self.combineFeatures = combineFeatures
        self.minDF = minDF
        if self.minDF >= 1:
            self.minDF = int(self.minDF)
        self.removeDupColumns = removeDupColumns
        self.useSpellChecker = useSpellChecker
        self.provideMisspellingDetailed = provideMisspellingDetailed
        self.additionalCols = additionalCols
        self.writeRepresentations = writeRepresentations
        self.exclusionsFileFullPath = exclusionsFileFullPath
        self.runType = runType
        self.justRepresentations = justRepresentations

        # set default values
        # inputFileFullPath = basepath + '/upload/dvd.txt'
        # outputFileName = 'output'
        # inputLimit = 20
        # maxFeatures = 0
        # maxNgram = 4
        # maxCores = 4
        # lexiconFileFullPath = basepath + '/external/lexicons/GloveWG.zip' # False will use folder read .txt file mode
        # vader = True
        # wnaReturnLevel = 5
        # buildVectors = 'bB'
        # index = False
        # removeZeroVariance = True
        # combineFeatures = False
        # minDF = 3
        # removeDupColumns = True
        # useSpellChecker = True
        # provideMisspellingDetailed = True
        # additionalCols = True
        # writeRepresentations = True
        # exclusionsFileFullPath = basepath + '/external/lexicons/exclusions.txt'
        # runType = 'full'

        # initialize mapply
        if maxCores:
            useCores = min(mp.cpu_count(), maxCores)
        else:
            useCores = mp.cpu_count()

        mapply.init(
            n_workers = useCores) # chunk_size = 5

        ### Setup Spellchecking ###

        b = enchant.Broker()
        #print(b.describe())
        spellcheckerLibrary = 'en'
        b.set_ordering(spellcheckerLibrary, 'aspell')

        if exclusionsFileFullPath != 'None':

            self.spellchecker = enchant.DictWithPWL(spellcheckerLibrary, pwl = exclusionsFileFullPath, broker = b)
            exclusionsFile = open(exclusionsFileFullPath, 'r')
            exclusionsLength = len(exclusionsFile.readlines())
            #exclusions = [x.lower() for x in exclusions]
            exclusionsFile.close()

            #print(vars(spellchecker))
            print('# Spellchecker Details #')
            print('Provider:', self.spellchecker.provider)
            print('Enchant Version:', enchant.get_enchant_version())
            print('Dictionary Tag:', self.spellchecker.tag)
            print('Dictionary Location:', self.spellchecker.provider.file)
            print('Total Exclusions: ' + str(exclusionsLength))
        else:
            self.spellchecker = enchant.DictWithPWL(spellcheckerLibrary, broker = b)
            print('# Spellchecker Details #')
            print('Provider:', self.spellchecker.provider)
            print('Enchant Version:', enchant.get_enchant_version())
            print('Dictionary Tag:', self.spellchecker.tag)
            print('Dictionary Location:', self.spellchecker.provider.file)
            print('Total Exclusions: 0 (No File Supplied)')

        
        ### Set Paths ###
        basepath = os.getcwd()
        self.lexiconpath = basepath + '/external/lexicons'
        self.outputpath = basepath + '/output'

        ### Setup NLP Tools ###

        

        # WordNet Affect (not on pip; see github) #
        sys.path.append(basepath + '/external/extracted/WNAffect-master')
        from wnaffect import WNAffect
        from emotion import Emotion
        self.wna = WNAffect(basepath + '/external/extracted/wordnet-1.6', basepath + '/external/extracted/wn-domains')

        # VADER #
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        # spaCy #
        import spacy
        self.nlp = spacy.load('en_core_web_sm', exclude = ['lemmatizer'])
        self.nlp.max_length = 10 ** 10

        print('\n# CPU Cores Detected and Initialized:', useCores, '#\n')
        print('# Python Details #')
        print(sys.version, '\n')

        print("# Package Versions #")
        print('SpaCy:', spacy.__version__)
        print('PyEnchant:', enchant.__version__)
        print('pywsd:', pywsd.__version__)
        print('NLTK:', nltk.__version__, '\n')

    def ProcessText (self, sentence, lexicons=None, debug = False):

        doc = self.nlp(sentence)

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
                wnaRes = str(self.wna.get_emotion(Word[i], POSTreeBank[i][1]).get_level(self.wnaReturnLevel))
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

    ### Process Sentence Function ###
    def TextToFeatures (self, textData, lexicons = None):

        textData = pd.DataFrame({
            'InitialSentence': textData
            })

        # Basic Text Cleanup
        print('# Performing Basic Text Cleanup #\n')
        res = textData['InitialSentence'].mapply(self.BasicTextCleanup, lexicons=lexicons)
        resZip = list(zip(*res))

        textData = pd.concat([textData, pd.concat(resZip[0], ignore_index = True)], axis = 1)
        corrections = pd.concat(resZip[1], ignore_index = True)
        
        res = textData['Sentence'].mapply(self.ProcessText, lexicons=lexicons)
        textData = pd.concat([textData, pd.DataFrame(res.values.tolist())], axis = 1)

        return([textData, corrections])



    def BasicTextCleanup (self, sentence, lexicons = None, debug = False):

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

            chkr = SpellChecker(self.spellchecker, sentence, filters = [EmailFilter, URLFilter])

            collectMisspellingDetails = {
                'Word': [], 
                'Substitution': [], 
                'SubstitutionText': []
                }

            for err in chkr:

                #print('\nSpellcheck Word:', err.word)
                matchedWord = False

                word = err.word

                if lexicons is not None and self.provideMisspellingDetailed:

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

            MisspellingRaw = ' '.join(spaCyTOK(sentenceMisspelling, self.nlp)).lower()

            Misspelling = re.sub('ABCMISSPELLING[0-9]+XYZ'.lower(), 'MISSPELLING', MisspellingRaw)

            if self.provideMisspellingDetailed == True:

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

            if self.useSpellChecker:
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

            if self.provideMisspellingDetailed:
                resReturn['Feature_MisspellingDetailed'] = MisspellingDetailed

            return([resReturn, correctionDF])


    ### Process Corpus Function ###
    def TextToFeaturesReader (self, sentenceList, lexicons = None):

        if (self.inputLimit == 0):
            inputLimit = len(sentenceList)
        elif (self.inputLimit > len(sentenceList)):
            inputLimit = len(sentenceList)
            
        if (len(sentenceList) == 0):
            print("No rows in input data! Terminating...", '\n')
            quit()

        processRows = min(len(sentenceList), inputLimit)
        print('Items to Process:', processRows, '\n')

        print('# Now Processing Text Items #', '\n')

        start = datetime.now(get_localzone())

        processedText, corrections = self.TextToFeatures(sentenceList[:inputLimit], lexicons = lexicons)

        print('\nItems Processed: ' + str(len(processedText)) + ' (Time Elapsed: {})\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

        with open(os.path.join(self.outputpath, self.outputFileName + '_raw_representations.pickle'), "wb") as f:
            pickle.dump({'ProcessedText': processedText, 'Corrections': corrections}, f, pickle.HIGHEST_PROTOCOL)

        return({'ProcessedText': processedText, 'Corrections': corrections})



    def RunFeatureConstruction (self):
        # defaults: lexiconpath, fullinputpath, inputLimit, outputpath, outputFileName, maxCores = False, lexiconFileFullPath = False, wnaReturnLevel = 5, useSpellChecker = False, provideMisspellingDetailed = False
        lexicons = ReadAllLexicons(self.lexiconpath, self.lexiconFileFullPath)

        print('# Now Reading Raw Data #')

        res = self.inputFile
        rawTextData = res['corpus']
        classLabels = res['classLabels']

        # Run a single test on a specific row:
        # TextToFeatures(raw[4-1], debug = True, lexicons = lexicons); quit() # 1357 in dvd.txt includes spanish; 4 in modified dvd_issue.txt

        output = self.TextToFeaturesReader(rawTextData, lexicons = lexicons)

        end_time = datetime.now(get_localzone())
        end_time_str = str(end_time.strftime(fmt))
        print('### Stage execution finished at ' + end_time_str + ' (Time Elapsed: {})'.format(pd.to_timedelta(end_time - start_time).round('1s')) + ' ###\n')

    def RunPostFeatureConstruction (self):
        # defaults: lexiconpath, fullinputpath, inputLimit, outputpath, outputFileName, maxCores = False, maxNgram = 3, lexiconFileFullPath = False, vader = False, wnaReturnLevel = 5, maxFeatures = 50, buildVectors = 'b', index = False, removeZeroVariance = True, combineFeatures = False, minDF = 5, removeDupColumns = False, useSpellChecker = False, provideMisspellingDetailed = False, additionalCols = False, writeRepresentations = False, justRepresentations = False
        #print(maxCores)

        print('# Now Reading Raw Data #')

        res = self.inputFile
        rawTextData = res['corpus']
        classLabels = res['classLabels']

        print('\n# Now Reading Feature Data Pickle #')
        start = datetime.now(get_localzone())

        with open(os.path.join(self.outputpath, self.outputFileName + '_raw_representations.pickle'), "rb") as f:
            output = pickle.load(f)

        processedText = output['ProcessedText']
        corrections = output['Corrections']

        print('- Time Elapsed: {}\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

        print('# Now Writing Spellchecked Sentences to Disk #')
        start = datetime.now(get_localzone())

        sentenceWriter = open(os.path.join(self.outputpath, self.outputFileName + '_cleaned_sentences.txt'), 'w', encoding = 'utf-8')
        for i, cleanedSentence in enumerate(processedText['Spellchecker_CorrectedSentence']):
            sentenceWriter.write(classLabels[i] + '\t' + cleanedSentence + '\n')
        sentenceWriter.close()
        processedText = processedText.drop(columns = 'Spellchecker_CorrectedSentence')

        print('- Time Elapsed: {}\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))

        print('# Now Writing Spelling Corrections to Disk #')
        ResultWriter(corrections, self.outputpath, self.outputFileName + '_spelling_corrections', index = False, header = True)

        if self.additionalCols:
            additionalCols = processedText.loc[:, processedText.columns.str.startswith('Spellchecker_')]
            additionalCols.columns = additionalCols.columns.str.lstrip('Spellchecker_')
        else:
            additionalCols = False

        if self.vader:
            print('# Now Generating VADER Scores #')
            start = datetime.now(get_localzone())
            vader = runVader(rawTextData, self.inputLimit)
            print('- Time Elapsed: {}\n'.format(pd.to_timedelta(datetime.now(get_localzone()) - start).round('1s')))
        else:
            vader = False

        print('# Now Constructing Feature Vectors #', '\n')

        representations = processedText.loc[:, processedText.columns.str.startswith('Feature_')]
        representations.columns = representations.columns.str.lstrip('Feature_')
        df = VectorProcessor(representations, maxNgram = self.maxNgram, vader = vader, maxFeatures = self.maxFeatures, buildVectors = self.buildVectors, removeZeroVariance = self.removeZeroVariance, combineFeatures = self.combineFeatures, minDF = self.minDF, removeDupColumns = self.removeDupColumns, classLabels = classLabels, additionalCols = additionalCols, writeRepresentations = self.writeRepresentations, justRepresentations = self.justRepresentations, outputpath = self.outputpath, outputFileName = self.outputFileName)

        print('# Now Writing Results to Disk #')
        ResultWriter(df, self.outputpath, self.outputFileName, index = self.index, header = True)

        print('# Now Generating Column Key Files #')
        GenerateColumnKey(df, self.outputpath, self.outputFileName)

        end_time = datetime.now(get_localzone())
        end_time_str = str(end_time.strftime(fmt))
        print('Output Dimensions (Rows, Features):', df.shape, '\n\n### Execution finished at ' + end_time_str + ' (Time Elapsed: {})'.format(pd.to_timedelta(end_time - start_time).round('1s')) + ' ###\n')

    def RunRepresentationConstructionOnly (self):
        # defaults from before: lexiconpath, fullinputpath, inputLimit, outputpath, outputFileName, maxCores = False, maxNgram = 3, lexiconFileFullPath = False, vader = False, wnaReturnLevel = 5, maxFeatures = 50, buildVectors = 'b', index = False, removeZeroVariance = True, combineFeatures = False, minDF = 5, removeDupColumns = False, useSpellChecker = False, provideMisspellingDetailed = False, additionalCols = False, writeRepresentations = False, justRepresentations = True
        #print('TEST', maxFeatures)

        print('# Now Reading Raw Data #')

        res = self.inputFile
        raw = res['corpus']
        classLabels = res['classLabels']

        with open(os.path.join(self.outputpath, self.outputFileName + '_raw_representations.pickle'), "rb") as f:
            output = pickle.load(f)

        representations = output['Representations']

        cleanedSentences = []

        for i, eachExtended in enumerate(output['AdditionalCols']):
            cleanedSentences.append(eachExtended['CleanedSentence'])

            if i == 0:
                correctionDFs = eachExtended['CorrectionDF']
                countDFs = eachExtended['CountDF']
            else:
                correctionDFs = pd.concat([correctionDFs, eachExtended['CorrectionDF']])
                countDFs = pd.concat([countDFs, eachExtended['CountDF']])

        print('# Now Writing Cleaned Sentences to Disk #', '\n')
        sentenceWriter = open(os.path.join(self.outputpath, self.outputFileName + '_cleaned_sentences.txt'), 'w', encoding = 'utf-8')
        for i, cleanedSentence in enumerate(cleanedSentences):
            sentenceWriter.write(classLabels[i] + '\t' + cleanedSentence + '\n')
        sentenceWriter.close()

        print('# Now Writing Spelling Corrections to Disk #', '\n')
        ResultWriter(correctionDFs, self.outputpath, self.outputFileName + '_spelling_corrections', index = False, header = True)

        if self.additionalCols:
            additionalCols = countDFs.reset_index(drop = True)
        else:
            additionalCols = False

        if vader:
            print('# Now Generating VADER Scores #', '\n')
            vader = runVader(raw, self.inputLimit)
        else:
            vader = False

        print('# Now Constructing Feature Vectors #', '\n')

        df = VectorProcessor(representations, maxNgram = self.maxNgram, vader = vader, maxFeatures = self.maxFeatures, buildVectors = self.buildVectors, removeZeroVariance = self.removeZeroVariance, combineFeatures = self.combineFeatures, minDF = self.minDF, removeDupColumns = self.removeDupColumns, classLabels = classLabels, additionalCols = additionalCols, writeRepresentations = self.writeRepresentations, justRepresentations = self.justRepresentations)


### Test cases ###

df = pd.read_csv('../examples/dvd.txt', sep='\t', header=None, names=["classLabels", "corpus"])


tgon = Textagon(
    df, "dvd", 0, 0, 4, 3, "upload/Lexicons_v5.zip", 
    1, 5, "bB", 0, 1, 0, 3, 1, 1, 1, 1, 1, "upload/exclusions.txt", "full",
    False
)

tgon.RunFeatureConstruction()
tgon.RunPostFeatureConstruction()


"""
if __name__ == '__main__':

    if runType.lower() == 'feature':
        RunFeatureConstruction(lexiconpath, inputFileFullPath, inputLimit, outputpath, outputFileName, maxCores = maxCores, lexiconFileFullPath = lexiconFileFullPath, wnaReturnLevel = wnaReturnLevel, useSpellChecker = useSpellChecker, provideMisspellingDetailed = provideMisspellingDetailed)
    elif runType.lower() == 'matrix':
        RunPostFeatureConstruction(lexiconpath, inputFileFullPath, inputLimit, outputpath, outputFileName, maxCores = maxCores, maxNgram = maxNgram, lexiconFileFullPath = lexiconFileFullPath, vader = vader, wnaReturnLevel = wnaReturnLevel, maxFeatures = maxFeatures, buildVectors = buildVectors, index = index, removeZeroVariance = removeZeroVariance, combineFeatures = combineFeatures, minDF = minDF, removeDupColumns = removeDupColumns, useSpellChecker = useSpellChecker, provideMisspellingDetailed = provideMisspellingDetailed, additionalCols = additionalCols, writeRepresentations = writeRepresentations, justRepresentations = False)
    elif runType.lower() == 'representation':
        RunFeatureConstruction(lexiconpath, inputFileFullPath, inputLimit, outputpath, outputFileName, maxCores = maxCores, lexiconFileFullPath = lexiconFileFullPath, wnaReturnLevel = wnaReturnLevel, useSpellChecker = useSpellChecker, provideMisspellingDetailed = provideMisspellingDetailed)
        RunPostFeatureConstruction(lexiconpath, inputFileFullPath, inputLimit, outputpath, outputFileName, maxCores = maxCores, maxNgram = maxNgram, lexiconFileFullPath = lexiconFileFullPath, vader = vader, wnaReturnLevel = wnaReturnLevel, maxFeatures = maxFeatures, buildVectors = buildVectors, index = index, removeZeroVariance = removeZeroVariance, combineFeatures = combineFeatures, minDF = minDF, removeDupColumns = removeDupColumns, useSpellChecker = useSpellChecker, provideMisspellingDetailed = provideMisspellingDetailed, additionalCols = additionalCols, writeRepresentations = writeRepresentations, justRepresentations = True)
    elif runType.lower() == 'featuretorep':
        RunRepresentationConstructionOnly(lexiconpath, inputFileFullPath, inputLimit, outputpath, outputFileName, maxCores = maxCores, maxNgram = maxNgram, lexiconFileFullPath = lexiconFileFullPath, vader = vader, wnaReturnLevel = wnaReturnLevel, maxFeatures = maxFeatures, buildVectors = buildVectors, index = index, removeZeroVariance = removeZeroVariance, combineFeatures = combineFeatures, minDF = minDF, removeDupColumns = removeDupColumns, useSpellChecker = useSpellChecker, provideMisspellingDetailed = provideMisspellingDetailed, additionalCols = additionalCols, writeRepresentations = writeRepresentations, justRepresentations = True)
    elif runType.lower() == 'full':
        RunFeatureConstruction(lexiconpath, inputFileFullPath, inputLimit, outputpath, outputFileName, maxCores = maxCores, lexiconFileFullPath = lexiconFileFullPath, wnaReturnLevel = wnaReturnLevel, useSpellChecker = useSpellChecker, provideMisspellingDetailed = provideMisspellingDetailed)
        RunPostFeatureConstruction(lexiconpath, inputFileFullPath, inputLimit, outputpath, outputFileName, maxCores = maxCores, maxNgram = maxNgram, lexiconFileFullPath = lexiconFileFullPath, vader = vader, wnaReturnLevel = wnaReturnLevel, maxFeatures = maxFeatures, buildVectors = buildVectors, index = index, removeZeroVariance = removeZeroVariance, combineFeatures = combineFeatures, minDF = minDF, removeDupColumns = removeDupColumns, useSpellChecker = useSpellChecker, provideMisspellingDetailed = provideMisspellingDetailed, additionalCols = additionalCols, writeRepresentations = writeRepresentations, justRepresentations = False)
    else:
        print('Check your command arguments! Aborting...')
        exit()
"""

