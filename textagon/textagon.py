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

    doc = nlp.tokenizer(sentence)
    tokens = []
    for token in doc:
        tokens.append(token.text)
    return(tokens)

def splitWS (sentence):
    return(sentence.split(' '))

def vector_hasher(x):
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

### Below are functions to run ###

def say_hello():
    print("Hello, World!")

def setSpellChecking(exclusionsFileFullPath='None'):
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

    # Get the path of the default lexicon file
    if lexiconFileFullPath is None:
        print('No lexicon file provided. Using default lexicon file.')
        lexiconFileFullPath = pkg_resources.resource_filename('textagon', 'external/lexicons/Lexicons_v5.zip')

    def is_valid_zip_file(file_path):
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
    print("Reading raw text...")
    pure_chunck = SanityCheck(dataPath=path, override_original_file=False)
    classLabels = list(pure_chunck['samples_per_class'].keys())
    raw = [x[1] for x in pure_chunck['raw_data']]
    print("Reading raw text completed.")
    return({'corpus': raw, 'classLabels': classLabels})

def TextToFeatures (textData, debug = False, lexicons = None, wnaReturnLevel = 5, useSpellChecker = True, provideMisspellingDetailed = True, useCores = 1):

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







