

# NOTE: THIS IS NO LONGER THE WORKING VERSION> please refer to cluster.py in the main repository folder.

import numpy
#import pandas
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#TODO: Preprocessing. Generate 3 text files independently of this code. // or automated in another script
# File 1 contains the unique names of the files on separate lines
# File 2 contains the text of the files. Each email is separated by "END_OF_EMAIL"
# File 3 contains additional information (date, recipient), separated by "END"

"""
codes = open('codes.txt').read().split('\n')
texts = open('texts.txt').read().split('END_OF_EMAIL')
extra = open('extra.txt').read().split('END')

print(codes)
print(texts)
print(extra)
"""

emails = open('sample.txt').read().split('END_OF_EMAIL')
print(len(emails))

# Tokenize one email (called text)
# Note: Lemmatization is not possible, due to the distribution of multiple languages within text fragments.
def tokenize(text):
    # following @brandonrose's advice: tokenize by sentence and by word, to ensure that punctuation is caught
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    # filter out any tokens not containing letters (numeric tokens, raw punctuation)
    accepted_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            accepted_tokens.append(token)
    return accepted_tokens # returns a list of words for the given text


# Create a lexicon of tokens present in emails
# Lexicon will be used to evaluate at a glance what french/english/errors distribution is found in my emails
# Lower-case all text, then tokenize and extend words to a list.
def make_lexicon(list_of_texts):
    lexicon = []
    for text in list_of_texts:
        text = text.lower()
        current_words = tokenize(text)
    lexicon.extend(current_words)
    return lexicon

#Use sklearn module TfidfVectorizer to obtain document similarity matrix
vect = TfidfVectorizer(
                                 max_df=0.8, max_features=200000,
                                 min_df=0.06, stop_words=None,
                                 use_idf=True, tokenizer=tokenize, ngram_range=(1,3))


matrix = vect.fit_transform(emails) #apply this vectorizer to data, using fit_transform
print("\nMatrix to array: [list of [documents, which are a list of word tf-idf values]] ")
print(matrix.toarray())
print("\nMatrix shape: (number of documents, number of features) ")
print(matrix.shape)

#TODO: Work out whether I'm interested in cosine similarity or cos distance for hierarchical agglomerative clustering
similarity = cosine_similarity(matrix)
print("\nSimilarity: ")
print(similarity)

'''
dist = 1 - cosine_similarity(matrix)
print("\nDistance!:")
print(dist)
'''

#smalls = []
#for line in numpy.nditer(similarity, ): #TODO: work out how numpy arrays work and how to iterate through them without changing the values :o !
#    smalls.extend(min(line))
#print(min(smalls))

# END HERE 5th September.
