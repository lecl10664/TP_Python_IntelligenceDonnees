#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:41:44 2021

@author: leopoldclement
"""
# =============================================================================
# IMPORT
# =============================================================================
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk .download('stopwords')
nltk .download( 'gutenberg')
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk import sent_tokenize
from nltk import word_tokenize

# =============================================================================
# LOAD THE TEXT
# =============================================================================
filename = 'view.php.txt'
file=open(filename, 'rt')
text = file.read()
file.close()


print(sent_tokenize(text)[0])
print(sent_tokenize(text)[10])

sentences = sent_tokenize(text)
words=word_tokenize(sentences[0])
nltk.pos_tag(words)

text.split()[:100]
text_words = text.split()

print(stopwords.words('english'))

nltk .download('stopwords')
en_stops = set(stopwords.words('english'))

# =============================================================================
# Remove all stop words from the text and print the new text.
# =============================================================================
filtered_sentence = []  
for w in text_words:  
    if w not in en_stops:
        filtered_sentence.append(w)  
   
print(filtered_sentence) 

# =============================================================================
# 3)
# =============================================================================

nltk .download( 'conll2000')
from nltk.corpus import conll2000

for i in range(6):
    print(conll2000.sents()[i])

for i in range(6):
    print(conll2000.tagged_sents()[i])


# =============================================================================
# B. WordNet
# =============================================================================

nltk.download( 'wordnet')
from nltk.corpus import wordnet as wn

wn.synsets("Love")

syn = wn.synsets("Love")[0] 
print ("Synset name :  ", syn.name())  
print ("Synset meaning : ", syn.definition()) 
print ("Synset lemmas : ", syn.lemmas()) 
print ("Synset example : ", syn.examples())


# =============================================================================
# Print all synonyms and antonyms
# =============================================================================

synonyms = [] 
antonyms = [] 
  
for syn in wn.synsets("love"): 
    for l in syn.lemmas(): 
        synonyms.append(l.name()) 
        if l.antonyms(): 
            antonyms.append(l.antonyms()[0].name()) 
  
print(set(synonyms)) 
print(set(antonyms)) 

#3)

nltk.download('all')
help(nltk)

sorted(wn.langs())

#4)
wn.synset('love.n.01').lemma_names('jpn')


#5)

print(wn.synsets('cat'))
print(wn.synsets('dog'))
print(wn.synsets('car'))

dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')
car = wn.synset('car.n.01')

dog.path_similarity(cat)
dog.path_similarity(car)
car.path_similarity(cat)

dog.lch_similarity(cat)
dog.lch_similarity(car) 
car.lch_similarity(cat)

dog.wup_similarity(cat)
dog.wup_similarity(car)
car.wup_similarity(cat)


