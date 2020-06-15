# author: Ian Laird
# file name: Spell_Check_Driver.py
# class: NLP
# instructor: Dr. Lin
# due date: April 8, 2019
# date last modified: April 6, 2019

import sys
from SpellCorrector import SpellCorrector
from nltk.corpus import brown, state_union, twitter_samples, gutenberg, reuters


if len(sys.argv) > 2:
    print("Usage: SpellCorrector.py [fileName]")
    exit(1)

"""
num = input("Which corpus would you like to train on?\n1) Brown\n"
            "2) State of the Union\n3) Twitter Samples\n4) Gutenburg\n5) Reuters\n")
number = int(num)
if number == 1:
    corp = brown
elif number == 2:
    corp = state_union
elif number == 3:
    corp = twitter_samples
elif number == 4:
    corp = gutenberg
elif number == 5:
    corp = reuters
"""

num2 = input("What n would you like to use for n grams")
number2 = int(num2)

num3 = input("How many words should be considered for each incorrect word?")
number3 = int(num3)

if len(sys.argv) == 1:
    file = input("What is the fileName?")
else:
    file = sys.argv[1]

# speller = SpellCorrector(brown, number2, number3)
speller = SpellCorrector()
with open(file) as f:
    for line in f.readlines():
        returnList = speller.check(line)
        for (lineNum, misspelledWord, corrections) in returnList:
            print(str(lineNum) + "  " + misspelledWord + "  " +  " [ " + ",".join(corrections) + " ]")

exit(0)
