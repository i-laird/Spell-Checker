import nltk
from nltk.corpus import state_union
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk import ngrams

class NGramModel_Improved:

    def __init__(self, corpus, n, maxword, alternativeMode = False, lemmatizer = WordNetLemmatizer()):
        if n < 1 or maxword < 1:
            raise Exception("Silly noodle, negatives aren't fun")

        #Special Word
        self.SPECIALWORD = "??"

        #Number of Grams
        self.numberGrams = n

        #Number of words to consider
        maxNumberGrams = maxword

        try:
            NGramModel_Improved.taggedWords
        except AttributeError:
            NGramModel_Improved.taggedWords =  corpus.tagged_words(tagset='universal')
            NGramModel_Improved.filteredTaggedWords = [(lemmatizer.lemmatize(x0, self.tagTranslator(x1)).lower(), x1) for x0, x1 in NGramModel_Improved.taggedWords]

        if alternativeMode == True:
            # NGramModel.preTags = corpus.tagged_words(tagset='universal')
            # NGramModel.AllTags = [ (x[0].lower(), x[1]) for x in NGramModel.taggedWords]
            listOfTotalWords = [x[1] for x in NGramModel_Improved.filteredTaggedWords]
            NGramModel_Improved.tags = set(listOfTotalWords)
        else:
            #List of word in in corpus

            #listOfTotalWords = [x.lower() for x in corpus.words()]
            #NGramModel.filteredList = listOfTotalWords = [lemmatizer.lemmatize(x0, self.tagTranslator(x1)).lower() for x0,x1 in NGramModel.taggedWords]
            listOfTotalWords = [x0 for x0,x1 in NGramModel_Improved.filteredTaggedWords]

            NGramModel_Improved.setOfAllWords = set(listOfTotalWords)

        #Dictionary of Words
        self.dictionaryOfWords = Counter(listOfTotalWords)

        if alternativeMode == False:
            NGramModel_Improved.wordFreq = self.dictionaryOfWords

        self.counterSum = sum(self.dictionaryOfWords.values())

        #Dictionary of maxword most common
        self.listOfCommonWords = self.dictionaryOfWords.most_common(maxword)

        while self.SPECIALWORD in self.dictionaryOfWords:
            self.SPECIALWORD = self.SPECIALWORD + "?"

        #We have to use a dictionary here because a set cannot
        #house elements of length one, which, for example, a period can
        #be one of our most common words. Therefore we made a dictionary
        #with literally a dummy value
        dictOfCommonWords = dict()
        for word in self.listOfCommonWords:
            dictOfCommonWords[word.__getitem__(0)] = "dummy"


        #The new listing of words after we replace all the
        #undesired words
        self.newListingOfWords = [x if x in dictOfCommonWords else self.SPECIALWORD for x in listOfTotalWords]

        #get all the grams of smaller size
        self.grams = [ngrams(self.newListingOfWords, i) for i in range (1, self.numberGrams + 1)]

        #get the counter of the gramsOfSmaller
        self.numberOccurancesOfGrams = [Counter(x) for x in self.grams]


    #Returns the special word used
    def special_word(self):
        return self.SPECIALWORD

    #returns the frequency of a gram in a corpus
    def freq(self, l):
        #make sure the list l has same legnth of corpus trained on
        if len(l) != self.numberGrams:
            return -1

        term = tuple(l)
        #If our gram is in the corpus
        if term in self.numberOccurancesOfGrams[self.numberGrams - 1]:
            return self.numberOccurancesOfGrams[self.numberGrams - 1][term]
        else:
            return 0

    def prob(self, l):

        #see what the sentence that is actually going to be analyzed is
        sentence = l[-self.numberGrams:] if len(l) >= self.numberGrams else l[:]

        gram = tuple(sentence)

        #get the shorter list
        shorterList = sentence[:-1]

        #get the total amount of occurrences for that gram
        numTotalGram = self.numberOccurancesOfGrams[len(sentence) - 1][gram]

        smallGram = tuple(shorterList)

        #return the probability
        return numTotalGram / (self.numberOccurancesOfGrams[len(sentence) - 2][smallGram] + 1)

    def smartProb(self, l):
        gramNum = self.numberGrams if self.numberGrams <= len(l) else len(l)
        while True and gramNum > 1:
            #see what the sentence that is actually going to be analyzed is
            sentence = l[-gramNum:]

            gram = tuple(sentence)

            #get the shorter sentence
            shorterList = sentence[:-1]

            #get the total amount of occurrences for that gram
            numTotalGram = self.numberOccurancesOfGrams[len(sentence) - 1][gram]

            smallGram = tuple(shorterList)

            #return the probability
            prob = numTotalGram / (self.numberOccurancesOfGrams[len(sentence) - 2][smallGram] + 1)
            if prob > 0:
                return prob * gramNum
            gramNum -= 1
        # if here just return the probability of the word
        return self.dictionaryOfWords[l[len(l) - 1]] / self.counterSum

    def tagTranslator(self, tag):
        if tag[0] in ['J','j']:
            return 'a'
        elif tag[0] in ['R','r']:
            return 'r'
        elif tag[0] in ['V','v']:
            return 'v'
        return 'n'
"""
#Testing Driver
n = NGramModel(state_union, 2, 100)

#for gram in n.ngramsOfNewList:
#    print(gram)

l = []
l.append("is")
l.append("with")

print("2-gram:")
print(n.freq(l))
print(n.prob(l))
print("\n")


ng = NGramModel(state_union, 3, 100)

l = []
l.append("is")
l.append("with")
l.append("us")

print("3-gram:")
print(ng.freq(l))
print(ng.prob(l))
print("\n")

ngg = NGramModel(state_union, 4, 100)

l = []
l.append("to")
l.append("???")
l.append("people")
l.append("'")

print("4-gram:")
print(ngg.freq(l))
print(ngg.prob(l))
"""