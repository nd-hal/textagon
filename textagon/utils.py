
import fnmatch
import gc
import itertools
import os
import pickle
import nltk
import numpy as np
import pandas as pd
import re
import zipfile as zf

from datetime import datetime
from pytz import timezone
from tzlocal import get_localzone


from bs4 import BeautifulSoup as BS
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import enchant
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from importlib.resources import files


def spaCyTOK (sentence, nlp):

    doc = nlp.tokenizer(sentence)
    tokens = []
    for token in doc:
        tokens.append(token.text)
    return(tokens)

def splitWS (sentence):
    return(sentence.split(' '))

def vector_hasher(x):
    return hash(tuple(x))


def BuildLexicon (L, customLexicons, file):

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
def ReadAllLexicons (lexiconpath, lexiconFileFullPath = False):
    """Read Custom Lexicons Function"""

    customLexicons = {}


    if lexiconFileFullPath:
        
        if lexiconFileFullPath != 'None':
            lexFile = files('textagon.data').joinpath(lexiconFileFullPath)

            zipFile = zf.ZipFile(lexFile, 'r')
            for file in sorted(zipFile.namelist()):
                if fnmatch.fnmatch(file, '*.txt'):
                    L = zipFile.read(file).decode('utf-8').encode('utf-8').decode('unicode-escape')
                    customLexicons = BuildLexicon(L, customLexicons, file)
    else:

        for (dir, dirs, pathFiles) in os.walk(lexiconpath):
            for file in pathFiles:
                if fnmatch.fnmatch(file, '*.txt'):
                    L = open(os.path.join(dir, file), "r").read().encode('utf-8').decode('unicode-escape')
                    #L = codecs.open(os.path.join(dir, file), "r", "utf-8").read()
                    customLexicons = BuildLexicon(L, customLexicons, file)

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




### Create Feature Combinations Function ###
def FeatureCombiner (data):

    nonParallelReps = [] #['Misspelling', 'MisspellingDetailed'] # include all non parallelizable reps here (will exclude from combos)

    processedFeatures = list(dict.keys(data))
    #print(processedFeatures)

    # remove unparallelizable representations
    processedFeatures = list(set(processedFeatures) - set(nonParallelReps))

    #print(processedFeatures)
    #quit()

    combos = list(itertools.combinations(processedFeatures, 2))

    # TO-DO: Remove combos involving: Word&POS_POS, Word_Word&POS, etc.
    # ...

    print('Feature Combinations to Build:', len(combos), '\n')

    #print('Combos:', combos, '\n')

    comboReturn = {}

    for combo in combos:

        comboName = '|&|'.join(combo)

        #print(comboName)

        comboReturn[comboName] = []

        for j, item in enumerate(data[combo[0]]):

            comboTextMaxLength = 0

            comboText = list(zip(data[combo[0]][j], data[combo[1]][j]))

            #print(comboText)

            for i, text in enumerate(comboText):

                comboText[i] = '|+|'.join(text)

                '''
                if j == 0:
                    print(i)
                    print(text)
                    print(comboText[i])
                '''

                comboTextLength = len(set(comboText[i].split('|+|')))

                if comboTextLength > comboTextMaxLength:
                    comboTextMaxLength = comboTextLength

            '''
            if j == 0:
                print(comboTextMaxLength)
                # comboTextMaxLength of 1 is problematic because it means none of the comboed words added anything (happens with lexicons for example when there are no replacements)
                if comboTextMaxLength == 1:
                    print(comboName)
            '''

            #print(list(comboText))

            #print(comboText)

            # remove feature redundancy (this should also be applied to the regular generator, potentially)
            for i, text in enumerate(comboText):

                #print(len(set(text.split('_'))))

                if len(set(text.split('|+|'))) < comboTextMaxLength or comboTextMaxLength == 1:
                    comboText[i] = data['Word'][j][i]

            #print(comboName, comboText, '\n')

            comboReturn[comboName].append(comboText)


    #print(comboReturn)

    return(comboReturn)



### Read Input File Function ###
def ReadRawText (path, classLabels = True):

    path = nltk.data.find(path)
    raw = open(path, 'rb').read().decode("utf-8", "ignore").split('\n') #.splitlines() #.decode("utf-8", "replace")
    # fix extra empty row read
    if raw[len(raw) - 1] == '':
        raw = raw[:len(raw) - 1]

    if classLabels:
        doOnce = True
        classLabels = []
        for i, item in enumerate(raw):
            itemSplit = item.split('\t')
            classLabels.append(itemSplit[0])

            if len(itemSplit) <= 1:
                print('\n', '# Error: At least one class label was not found in the input file. Please check your input file and retry. Aborting... #', '\n', sep = '')
                print('Error on line:', i)
                print('Input:', raw[i-1])
                quit()
            elif len(itemSplit) == 2:
                raw[i] = itemSplit[1]
            elif len(itemSplit) > 2:
                raw[i] = '\t'.join( itemSplit.pop(0) )

    print('Total Text Items Read:', len(raw))

    return({'corpus': raw, 'classLabels': classLabels})

### Construct Legomena Representations ###
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
            vocab = vectorizer.get_feature_names_out()
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

### Vectorizer Helper Function ###
def BuildFeatureVector (data, vectorizer, vectorizerName, feature, debug = False):

    # Using standard scikit vectorizers. For custom analyzer, see http://stackoverflow.com/questions/26907309/create-ngrams-only-for-words-on-the-same-line-disregarding-line-breaks-with-sc

    train_data_features = vectorizer.fit_transform( data )
    #train_data_features = train_data_features.toarray()

    names = vectorizer.get_feature_names_out()

    #debug = True

    if feature == 'Misspelling' and debug == True:
        print('### ' + feature + ' ###')
        print(vectorizerName)
        print(data, '\n')
        #print(names, '\n')
        #print(vectorizer.vocabulary_)

        vocab = vectorizer.get_feature_names_out()
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

### Convert Representations into Feature Vectors ###
def VectorProcessor (data, maxNgram = 3, vader = False, maxFeatures = None, buildVectors = 'b', removeZeroVariance = True, combineFeatures = False, minDF = 5, removeDupColumns = False, classLabels = False, runLegomena = True, additionalCols = False, writeRepresentations = False, justRepresentations = False, outputpath = '', outputFileName = ''):

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
        #end_time_str = str(end_time.strftime(fmt))
        #print('### Stage execution finished at ' + end_time_str + ' (Time Elapsed: {})'.format(pd.#to_timedelta(end_time - start_time).round('1s')) + ' ###\n')
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

def ResultWriter (df, outputpath, outputFileName, index = False, header = False, compression = None):

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

def GenerateColumnKey(df, outputpath, outputFileName):

    # |~| separates vectorizer, category, and feature (in that order); always 2 in label (e.g., BINARY|~|WORD|~|hello)
    # |-| replaces spaces within features from higher order n-grams, e.g., "the|-|cat|-|jumped" (3-gram); this also applies to character n-grams that include spaces, e.g., g|-|a == 'g a'
    # |_| indicates a composite feature was generated, e.g., WordPOS of cat|_|NN
    # |&| indicates a category is a two-way combo, e.g., POS|&|HYPERNYM
    # |+| indicates a combo composite feature was formed, e.g., NN|+|CANINE based on the Word 'dog'
    # _ can appear as part of a substitution (e.g., the hypernym for style, EXPRESSIVE_STYLE)
    # category names with spaces (e.g., from lexicon file names) will have their white space stripped
    # original words are in all lower case; substituted word tags are in all caps (e.g., POSITIVE, NEUTRAL), as are the latter half of word composites (e.g., dog_NN, dog_CANINE, keith_PERSON)

    start = datetime.now(get_localzone())

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
