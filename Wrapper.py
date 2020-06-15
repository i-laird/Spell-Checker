# author: Ian Laird
# file name: Wrapper.py
# class: NLP
# instructor: Dr. Lin
# due date: April 8, 2019
# date last modified: April 6, 2019

import nltk
import string

from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
from NGramModel_Improved import NGramModel_Improved


# wordGeneratorFactory
#
# This creates a generator that creates words with more and more edit distance from the original word.
#
# param:
#   word: the word that is to have similar words generated
#   runNum: the number of words to generate from the generator
#
# return:
#   newWord: the generated word
#   distance: the edit distance from the original word
#
# there are two modes
#  add letter
#  delete letter
#  Thus changing a letter can be thought of as removing a letter and then adding a letter in the same location
#       this method of calculation will give a replacement an edit distance of two
def wordGeneratorFactory(word, runNum):
    # will be using a BFS to keep creating different and different string

    # will keep a set of already constructed strings to improve efficiency
    already_encountered = set()
    queue = list()
    # the first word in the queue is the start word which has an edit distance of 0
    queue.append((word, 0))
    wordCount = 0

    # while there are more words in the queue keep going (this loop is essentially infinite)
    while len(queue) > 0:
        # get the top word of the queue
        poppedWord, currentDistance = queue.pop(0)
        # now create all similar words and stick them in
        # first do all that have a one letter difference
        for i in range(len (poppedWord)):
            for c in string.ascii_lowercase:
                newWord = poppedWord[:i] + c + poppedWord[i+1:]
                if wordCount >= runNum:
                    return
                if newWord not in already_encountered:
                    already_encountered.add(newWord)
                    queue.append((newWord, currentDistance + 1))
                    wordCount += 1
                    yield (newWord, currentDistance + 1)

        # second do all which have a letter removed
        for i in range(len(poppedWord)):
            if wordCount >= runNum:
                return
            newWord = poppedWord[:i] + poppedWord[i+1:]
            if newWord not in already_encountered:
                already_encountered.add(newWord)
                queue.append((newWord, currentDistance + 1))
                wordCount += 1
                yield (newWord, currentDistance + 1)

        # now add a letter at each position
        for i in range(len(poppedWord) + 1):
            for c in string.ascii_lowercase:
                if wordCount >= runNum:
                    return
                newWord = poppedWord[:i] + c + poppedWord[i:]
                if newWord not in already_encountered:
                    already_encountered.add(newWord)
                    queue.append((newWord, currentDistance + 1))
                    wordCount += 1
                    yield (newWord, currentDistance + 1)


# getWeight
#
# returns the weight of a created sentence
#
# params:
#   word: the word whose weight is to be found
#   sentence: the preceding word in the current sentence (in order!)
#   tagsToProbability: the probability of this location in the sentence being a certain tag
#   distance: the edit distance of word
#   nGramModel: the NGram Model that has been trained on a corpus
#   PosFreq: gives the number of times a word appears as a certain part of speech
#
# return:
#   the weight of the new word
def getWeight(word, sentence, tagsToProbability, distance, nGramModel, PosFreq, ll):
    maxLoc = -1
    maxPosVal = -1
    for t,p in tagsToProbability.items():
        lWord = ll.lemmatize(word=word, pos=nGramModel.tagTranslator(tag=t))
        val = PosFreq[t][lWord] / NGramModel_Improved.wordFreq[lWord] if lWord in NGramModel_Improved.wordFreq else 1
        if val > maxPosVal:
            maxPosVal = val
            maxLoc = t
    lWord = ll.lemmatize(word=word, pos=nGramModel.tagTranslator(tag=maxLoc))
    sentence.append(lWord)
    sentenceProb = nGramModel.smartProb(sentence)
    sentence.pop()
    return (sentenceProb * ( 1 / (distance**2) ) * maxPosVal, maxLoc, lWord)


# probEachTag
# finds the probability of each tag in a sentence
def probEachTag(taggedSentence, POSModel):
    return {t: addTagToSentence(t, taggedSentence, POSModel) for t in NGramModel_Improved.tags}


# addTagToSentence
# pushes the given tag to a sentence and then pops it once the associated probability has been found
def addTagToSentence(tag, sentence, POSModel):
    sentence.append(tag)
    returnVal = POSModel.prob(sentence)
    sentence.pop()
    return returnVal


# Wrapper
#
# wrapper for a generator that catches spelling errors on a file
class Wrapper:
    def __init__(self, line, nGramModel, runNum, POSModel, tags, lem = WordNetLemmatizer()):
        self.line = line
        self.nGramModel = nGramModel
        self.runNum = runNum
        self.PosModel = POSModel
        self.tags = tags
        self.words = set(words.words())
        self.l = lem

    # run
    #
    # Generator for finding spelling errors and gives suggestions.
    #   a word is considered misspelled if there are not synsets for the word in wordnet
    #
    # return
    #   word: the misspelled word
    #   suggestion: the suggested new spelling for the word
    #   lineNum: the line of the file where this spelling error occurred
    def run(self):
        wordNum = -1
        sentence = list()
        taggedSentence = list()
        words = nltk.word_tokenize(self.line)
        # get the part of speech tagging for the sentence
        posSentence = nltk.pos_tag(words, tagset='universal')
        # loop through every word in the sentence as well as its tag
        for originalWord, originalTag in posSentence:
            # lemmatize the word according to the tag given
            # need to translate the tag from the corpus to one of the expected tags for the method
            word = self.l.lemmatize(originalWord, self.nGramModel.tagTranslator(originalTag)).lower()
            wordToMaxTag = dict()
            wordNum += 1
            # see if the word exists or is just punctuation characters
            if word in self.words or all(x in string.punctuation for x in word):
                # this word is spelled correctly so it can go on the sentence
                sentence.append(word)
                # keep the POS tag for the word as given by the tagger
                taggedSentence.append(originalTag)
                continue
            terms = dict()
            wordGenerator = wordGeneratorFactory(word, self.runNum)

            # get the probability for each tag for this position in the sentence
            tagsToProb = probEachTag(taggedSentence, self.PosModel)

            # generate new words and see how likely they are
            for (newWord, distance) in wordGenerator:
                # this eliminates a weird edge case where some words that end in two 's' are allowed
                if len(newWord) > 2 and newWord[len(newWord) - 1] == 's' and newWord[len(newWord) - 2] == 's':
                    continue
                # sentence.append(taggedWord)
                # get the weight of this new word
                weight, maxTag, lemmatizedWord = getWeight(newWord, sentence, tagsToProb, distance, self.nGramModel, self.tags, self.l)

                # see if this word is in the corpus
                # we have to do this after finding the weight because the best POS is used to lemmatize
                # if the word is not in the corpus then we just throw it out ( maybe change this in the future)
                if lemmatizedWord not in NGramModel_Improved.setOfAllWords :
                    continue

                # this keeps track of whatever tag is associated with each word
                wordToMaxTag[newWord] = maxTag
                if weight not in terms:
                    terms[weight] = list()

                # we keep a list of terms associated with each weight
                # (so later we can find the maximum weight terms)
                terms[weight].append(newWord)
                # sentence.pop()
            if len(terms) == 0:
                yield(wordNum, word, [])
                continue
            # now get the term with the highest weight
            returnSuggestions = list()
            termsSorted = sorted(terms.keys(), reverse=True)
            iterator = iter(termsSorted)
            suggestion = terms[termsSorted[0]][0]
            # return the top five hits
            for i in range(5):
                try:
                    returnSuggestions.extend(terms[next(iterator)])
                except StopIteration:
                    break
            # max_loc =  max(terms.keys(), key=(lambda x: terms[x]))
            # suggestion = terms[max_loc]

            # take the top suggestion and use it in future n gram calculations
            sentence.append(suggestion)
            # use the tag that is associated with this word
            taggedSentence.append(wordToMaxTag[suggestion])
            yield (wordNum, originalWord, returnSuggestions[:5])