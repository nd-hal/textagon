#!/usr/bin/python3

path = 'C:/Users/daved/Downloads/BEEF'

import sys
import os
import re
import cgi

import psutil
import os

import cgitb; cgitb.enable()

print('Content-type: text/html\n')

#sys.stdout = sys.__stderr__

#print(os.getpid())
f = open("pid.txt", "a")
f.write('\n' + str(os.getpid()))
f.close()

f = open("pid.txt", "r")
keepproc = f.read().split('\n')
f.close()

#print(keepproc)

killFlag = False

for proc in psutil.process_iter():
    pinfo = proc.as_dict(attrs=['pid', 'name'])
    procname = str(pinfo['name'])
    procpid = str(pinfo['pid'])
    if "python" in procname and procpid not in keepproc:
        #print("Notice: Stopped Outstanding Python Process ", proc)
        proc.kill()
        killFlag = True

### COLLECT FORM INPUT ###

form = cgi.FieldStorage()
#print(form.keys())

lexiconFile = form['lexiconFile']
#print(lexiconFile.filename)

inputFile = form['inputFile']
exclusionsFile = form['exclusionsFile']
outputFileName = form.getvalue('outputFileName')
maxCores = form.getvalue('maxCores')
inputLimit = form.getvalue('inputLimit')
maxFeatures = form.getvalue('maxFeatures')
maxNgram = form.getvalue('maxNgram')
wnaReturnLevel = form.getvalue('wnaReturnLevel')
minDF = form.getvalue('minDF')

if minDF[0] == '.':
    minDF = '0'+minDF

def HandleBinary (which, form):

    if form.getvalue(which) == 'on':
        res = str(1)
    else:
        res = str(0)

    return(res)

vader = HandleBinary('vader', form)
removeZeroVariance = HandleBinary('removeZeroVariance', form)
removeDupColumns = HandleBinary('removeDupColumns', form)
combineFeatures = HandleBinary('combineFeatures', form)
index = HandleBinary('index', form)

correctSpelling = HandleBinary('correctSpelling', form)
additionalCols = HandleBinary('additionalCols', form)
writeRepresentations = HandleBinary('writeRepresentations', form)

# collect vectorizers
buildVectors = ''
for vectorType in ['c', 'b', 't', 'C', 'B']:
    #print(form.getvalue(vectorType))
    if form.getvalue(vectorType) == 'on':
        buildVectors += vectorType

if buildVectors == '':
    buildVectors = 'tbc'

# collect input file
if inputFile.filename:
    fn = os.path.basename(inputFile.filename)
    inputFileFullPath = path + '/upload/' + fn
    open(inputFileFullPath, 'wb').write(inputFile.file.read())
    message = 'The input file "' + fn + '" was uploaded successfully!'
    inputFileName = inputFile.filename
else:
    message = 'No input file was attached!'
    inputFileName = None

# fix output file name if needed
if outputFileName == '':
    outputFileName = 'output'

# collect lexicon file
if lexiconFile.filename:
    fn = os.path.basename(lexiconFile.filename)
    lexiconFileFullPath = path + '/upload/' + fn
    open(lexiconFileFullPath, 'wb').write(lexiconFile.file.read())
    lexiconFileName = lexiconFile.filename
else:
    lexiconFileName = None
    lexiconFileFullPath = 'None'

# collect exclusions file
if exclusionsFile.filename:
    fn = os.path.basename(exclusionsFile.filename)
    exclusionsFileFullPath = path + '/upload/' + fn
    open(exclusionsFileFullPath, 'wb').write(exclusionsFile.file.read())
    exclusionsFileName = exclusionsFile.filename
else:
    exclusionsFileName = None
    exclusionsFileFullPath = 'None'

### PRINT LOG HTML ###

#sys.stdout = sys.__stdout__

print('<html><body>')
print('<title>Output</title>')
print('<div style="text-align: left; font-family: Verdana; margin: 50px 0 0 50px;">')
if killFlag:
    print('<p style="color: red; font-weight: bold;">Notice: Killed outstanding multi-core processes!</p>')
if inputFileName is None:
    print('<p><b>No data file attached. Please go back and try again!</b></p>')
else:
    print('<p><b>Request received. Now processing...</b></p>')
    print('<b>Input file:</b> %s<br>' % inputFileName )
    print('<b>Output file:</b> %s.csv<br>' % outputFileName )
    print('<b>Log file:</b> <a href="../output/' + outputFileName + '_log.txt" target="_new">' + outputFileName + '_log.txt</a><br>')

    #print('<textarea style="width: 600px; height: 600px;" id="results"></textarea>')

    #sys.stdout = sys.__stderr__

    import subprocess

    #sys.stdout = sys.__stdout__

    import re

    #log = str(open('log.txt', 'rb').read())
    #log = log.replace('b"', '"')
    #print('<script>document.getElementById("results").value = ' + log + ';</script>')

    # reset log
    f = open("output/" + outputFileName + "_log.txt", "w")
    f.close()

    f = open("output/" + outputFileName + "_log.txt", "a+")

    print('<p><b>Log (3s Update Interval):</b></p>')
    print('<iframe src="../output/' + outputFileName + '_log.txt" id="log" style="width: 900px; height: 600px;"></iframe>')
    print('<script>document.getElementById("log").value = ""; setInterval(function() { document.getElementById("log").contentWindow.location.reload(true); }, 3000);</script>')

    sys.stdout.flush()

    # write arguments to a file to avoid Popen issues
    args = open("args.txt", "w")

    argList = [ inputFileFullPath,
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
                correctSpelling,
                additionalCols,
                writeRepresentations,
                exclusionsFileFullPath ]

    '''
    for arg in argList:
        args.write(arg + '\n')
    args.close()
    '''

    args.write(' '.join([ 'python cgi-bin/processText.py',
        inputFileFullPath,
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
        correctSpelling,
        additionalCols,
        writeRepresentations,
        exclusionsFileFullPath
        ]))
    args.close()

    subprocess.Popen(' '.join([ 'python cgi-bin/processText-serverModel.py',
        inputFileFullPath,
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
        correctSpelling,
        additionalCols,
        writeRepresentations,
        exclusionsFileFullPath
        ]),
        stdout = f, stderr = f, shell=True)

    f.close()

    #print('<p style="font-weight: bold;">Once execution has finished (see log above), use these links to download your CSV file and key:<br><br><a href="../output/' + outputFileName + '.csv">' + outputFileName + '.csv</a><br><a href="../output/' + outputFileName + '_key.txt">' + outputFileName + '_key.txt</a><br><a href="../output/' + outputFileName + '_key_FRN.txt">' + outputFileName + '_key_FRN.txt</a></p>')
    print('<p style="font-weight: bold;">Once execution has finished (see log above), use this links to download your output files:<br><br><a href="../output/" target="_new">Open Output Location</a></p>')

print("</div>")
print('</html></body>')
