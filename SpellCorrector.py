# author: Ian Laird
# file name: SpellChecker.py
# class: NLP
# instructor: Dr. Lin
# due date: April 8, 2019
# date last modified: April 6, 2019

import nltk
from NGramModel_Improved import NGramModel_Improved
from Wrapper import Wrapper

# SpellCorrector
# can be used to find typos in lines
class SpellCorrector:

    # constructor
    # params:
    #   corp: the corpus that the ngram models are to be trained on
    #   n: the n to be used in the ngram models
    #   wordCount: the number of words to be considered for each misspelled word
    # return: none
    def __init__(self, corp = nltk.corpus.brown, n = 3, wordCount = 2000):
        SpellCorrector.NGRAM_MAX_WORD = 15000
        self.corp = corp
        self.n = n
        self.wordCount = wordCount
        self.nGramModel = NGramModel_Improved(corp, n, SpellCorrector.NGRAM_MAX_WORD)
        self.posModel = NGramModel_Improved(corp, n, SpellCorrector.NGRAM_MAX_WORD, True)
        # now get the probability of each pos for each word
        self.PosTags = dict.fromkeys(NGramModel_Improved.tags, dict.fromkeys(NGramModel_Improved.setOfAllWords, 0))
        for word, tag in NGramModel_Improved.filteredTaggedWords:
            self.PosTags[tag][word] += 1

    # check
    # checks a line for typos
    # params:
    #   line: the line to be checked for typos
    # return:
    #   x: the location of the word in the sentence
    #   y: the misspelled word
    #   z: a list of up to five spelling suggestions for the word
    def check(self, line):
        generator = Wrapper(line, self.nGramModel, self.wordCount, self.posModel, self.PosTags)
        return [(x, y, z) for x, y, z in generator.run()]

