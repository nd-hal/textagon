from nltk.corpus import WordNetCorpusReader as WNCR, wordnet as wn
import nltk
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from importlib.resources import files

#nltk.download('universal_tagset')


class WNAffect:
    def __init__(self, wordnet16_dir, wn_domains_dir):

        wn16_path = str(files('textagon.data').joinpath(f"{wordnet16_dir}/dict"))
        wn_domains_path = str(files('textagon.data').joinpath(f"{wn_domains_dir}/wn-affect-1.1/a-synsets.xml"))

        #self.wn16 = WNCR(wn16_path, omw_reader = None)
        self.wn16 = WNCR(wn16_path, wn16_path)

        #WN16 = WNCR('/Users/ddobolyi/Downloads/textagon-portable/external/extracted/wordnet-1.6/dict', 
        #            '/Users/ddobolyi/Downloads/textagon-portable/external/extracted/wordnet-1.6/dict')

        synList = ET.parse(wn_domains_path).getroot()
        posTags = [child.tag for child in synList]
        allPOS = [pos.split('-')[0] for pos in posTags]
        self.WNA11Synsets = {}

        for eachPOS in allPOS:
            self.WNA11Synsets[eachPOS] = pd.DataFrame([item.attrib for item in synList.findall(eachPOS + '-syn-list/' + eachPOS + '-syn')])
            self.WNA11Synsets[eachPOS]['intid'] = self.WNA11Synsets[eachPOS]['id'].str[2:].astype('int')

    def GetWNAffect(self, word):

        recognizedPOS = {'adj': wn.ADJ, 'adv': wn.ADV, 'noun': wn.NOUN, 'verb': wn.VERB}
        mappedPOS = nltk.map_tag('en-ptb', 'universal', nltk.pos_tag([word])[0][1]).lower()

        if mappedPOS in recognizedPOS:
            nltkPOS = recognizedPOS[mappedPOS]

        synsets = self.wn16.synsets(word, nltkPOS)
        offsets = [i.offset() for i in synsets]

        #print(synsets)
        #print(offsets)

        if mappedPOS != 'noun':
            checkMatch = self.WNA11Synsets[mappedPOS][self.WNA11Synsets[mappedPOS].intid.isin(offsets)]['noun-id'].values[0]
        else:
            checkMatch = self.WNA11Synsets[mappedPOS][self.WNA11Synsets[mappedPOS].intid.isin(offsets)]['id'].values[0]

        return(self.WNA11Synsets['noun'][self.WNA11Synsets['noun'].id.values == checkMatch].categ.values[0])