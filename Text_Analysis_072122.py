#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 21:31:52 2022

@author: deborahkewon
"""
##################################
########### import data ##########
##################################

import pandas as pd
import numpy as np
Reviews = pd.read_excel("Cleaned_Data_061522.xlsx")

# divide ratings into negative (1-2), neutral (3) and positive (4-5) #
def f(Reviews):
    if Reviews['Rating'] <=2:
        val = 'Negative'
    elif Reviews['Rating'] == 3:
        val = 'Neutral'
    else:
        val = 'Positive'
    return val

Reviews['Rating_Group'] = Reviews.apply(f, axis=1)

#######################################
# Number 9 - TF-IDF Graph ############
#####################################


# create each sentiment group
Negative_Group=Reviews.query("Rating_Group == 'Negative'")
Neutral_Group=Reviews.query("Rating_Group == 'Neutral'")
Positive_Group=Reviews.query("Rating_Group == 'Positive'")



# #stemming before TFIDF
Negative_Group['Review']=Negative_Group['Review'].astype(str)
Neutral_Group['Review']=Neutral_Group['Review'].astype(str)
Positive_Group['Review']=Positive_Group['Review'].astype(str)

Negative_Group['Review2'] = Negative_Group['Review'].str.replace('[^\w\s]','')
Neutral_Group['Review2'] = Neutral_Group['Review'].str.replace('[^\w\s]','')
Positive_Group['Review2'] = Positive_Group['Review'].str.replace('[^\w\s]','')

Negative_Group['Review'] = Negative_Group['Review'].str.replace('[^\w\s]',' ')
Neutral_Group['Review'] = Neutral_Group['Review'].str.replace('[^\w\s]',' ')
Positive_Group['Review'] = Positive_Group['Review'].str.replace('[^\w\s]',' ')


from nltk.stem import PorterStemmer, WordNetLemmatizer
porter_stemmer = PorterStemmer()
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)
Negative_Group['Review'] = Negative_Group['Review'].apply(stem_sentences)
Neutral_Group['Review'] = Neutral_Group['Review'].apply(stem_sentences)
Positive_Group['Review'] = Positive_Group['Review'].apply(stem_sentences)

# turn rating data frames into strings

Negative_Reviews = Negative_Group[Negative_Group['Review'].notnull()]
Negative_Review_List= Negative_Reviews['Review'].to_list()
Negative_Review_STR= ' '.join(Negative_Review_List)


Neutral_Reviews = Neutral_Group[Neutral_Group['Review'].notnull()]
Neutral_Review_List= Neutral_Reviews['Review'].to_list()
Neutral_Review_STR= ' '.join(Neutral_Review_List)


Positive_Reviews = Positive_Group[Positive_Group['Review'].notnull()]
Positive_Review_List= Positive_Reviews['Review'].to_list()
Positive_Review_STR= ' '.join(Positive_Review_List)


# lower strings for stopwrods

Negative_Review_STR_Lower= Negative_Review_STR.lower()
Negative_Review_STR_Lower= Negative_Review_STR_Lower.replace("’", "'") # if used a different apostrophe, twitter tokenizer doesn't work; making all apostrophies consistent

Neutral_Review_STR_Lower= Neutral_Review_STR.lower()
Neutral_Review_STR_Lower= Neutral_Review_STR_Lower.replace("’", "'")

Positive_Review_STR_Lower= Positive_Review_STR.lower()
Positive_Review_STR_Lower= Positive_Review_STR_Lower.replace("’", "'")


# bring stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# toeknizing & excluding stop words
# Tokenization 
import nltk
# from nltk.tokenize import word_tokenize - not using this pacakge since i want to extract "don't" as it is rather than "do" & "n't"
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

# negative
word_tokens_NEGATIVE=tknzr.tokenize(Negative_Review_STR_Lower)

filtered_sentence_NEGATIVE = [w for w in word_tokens_NEGATIVE if not w in stop_words]
filtered_sentence_NEGATIVE = []

for w in word_tokens_NEGATIVE:
    if w not in stop_words:
        filtered_sentence_NEGATIVE.append(w)

print(filtered_sentence_NEGATIVE)
Negative_Reviews_Cleaned= ' '.join(filtered_sentence_NEGATIVE) # string; lower case; tokenized; w/o stop words

import string
Negative_Reviews_Cleaned = Negative_Reviews_Cleaned.translate(str.maketrans('', '', string.punctuation))


# neutral
word_tokens_NEUTRAL=tknzr.tokenize(Neutral_Review_STR_Lower)

filtered_sentence_NEUTRAL = [w for w in word_tokens_NEUTRAL if not w in stop_words]
filtered_sentence_NEUTRAL = []

for w in word_tokens_NEUTRAL:
    if w not in stop_words:
        filtered_sentence_NEUTRAL.append(w)

print(filtered_sentence_NEUTRAL)
Neutral_Reviews_Cleaned= ' '.join(filtered_sentence_NEUTRAL) # string; lower case; tokenized; w/o stop words

Neutral_Reviews_Cleaned = Neutral_Reviews_Cleaned.translate(str.maketrans('', '', string.punctuation))


# positive
word_tokens_POSITIVE=tknzr.tokenize(Positive_Review_STR_Lower)

filtered_sentence_POSITIVE = [w for w in word_tokens_POSITIVE if not w in stop_words]
filtered_sentence_POSITIVE = []

for w in word_tokens_POSITIVE:
    if w not in stop_words:
        filtered_sentence_POSITIVE.append(w)

print(filtered_sentence_POSITIVE)
Positive_Reviews_Cleaned= ' '.join(filtered_sentence_POSITIVE) # string; lower case; tokenized; w/o stop words

Positive_Reviews_Cleaned = Positive_Reviews_Cleaned.translate(str.maketrans('', '', string.punctuation))


# TF - IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
All_Reviews= vectorizer.fit_transform([Negative_Reviews_Cleaned,Neutral_Reviews_Cleaned,Positive_Reviews_Cleaned])

tfidf_by_sentiment = pd.DataFrame(data = All_Reviews.toarray(),index = ['Neg','Neu','Pos'],columns=vectorizer.get_feature_names())

tfidf_by_sentiment=tfidf_by_sentiment.transpose()


### post tagging tf idf ###
### keep in mind that pos tagging can be incorrect sometimes due to stemming
from nltk import word_tokenize, pos_tag, pos_tag_sents
nltk.download('averaged_perceptron_tagger')

tfidf_by_sentiment['text'] = tfidf_by_sentiment.index
tfidf_by_sentiment['Pos_Tagging'] = nltk.pos_tag(tfidf_by_sentiment['text'])
tfidf_by_sentiment['Pos_Tagging'] = tfidf_by_sentiment['Pos_Tagging'].astype(str)
tfidf_by_sentiment['Pos_Tagging_C'] = tfidf_by_sentiment['Pos_Tagging'].str.rsplit(',').str[-1] 
tfidf_by_sentiment['Pos_Tagging_C'] = tfidf_by_sentiment['Pos_Tagging_C'].str.strip(')')



#tfidf_by_sentiment.to_excel('TF_IDF.xlsx', engine='xlsxwriter') 
    

''' end '''

###################################################################
###### Graph 10 - Most frequent words (unitgram) among wait related reviews by sentiment
###################################################################
import os
from nltk.collocations import *
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# Extract only ED related comments
Negative_Group["ED_Comment"] = Negative_Group["Review"].str.extract("( ed | er |emergency| ED | ER |Emergency|emerg|Emerg)")[0]
Neutral_Group["ED_Comment"] = Neutral_Group["Review"].str.extract("( ed | er |emergency| ED | ER |Emergency|emerg|Emerg)")[0]
Positive_Group["ED_Comment"] = Positive_Group["Review"].str.extract("( ed | er |emergency| ED | ER |Emergency|emerg|Emerg)")[0]
'''
# Exlcude null values and turn comments into lists
ED_Negative_Comment=Negative_Group[Negative_Group['ED_Comment'].notnull()]
ED_Neutral_Comment=Neutral_Group[Neutral_Group['ED_Comment'].notnull()]
ED_Positive_Comment=Positive_Group[Positive_Group['ED_Comment'].notnull()]

# ED review into lists
ED_Negative_Review_List= ED_Negative_Comment['Review'].to_list()
ED_Neutral_Review_List= ED_Neutral_Comment['Review'].to_list()
ED_Positive_Review_List= ED_Positive_Comment['Review'].to_list()


# Stemming to see most frequently appeared words regardless of pos
from stemming.porter2 import stem
ED_Negative_Review_List_Stemmed = [[stem(word) for word in sentence.split(" ")] for sentence in ED_Negative_Review_List]
ED_Neutral_Review_List_Stemmed = [[stem(word) for word in sentence.split(" ")] for sentence in ED_Neutral_Review_List]
ED_Positive_Review_List_Stemmed = [[stem(word) for word in sentence.split(" ")] for sentence in ED_Positive_Review_List]

# turn lists into strings
ED_Negative_Review_List_STR=  ','.join(str(v) for v in ED_Negative_Review_List_Stemmed)
ED_Neutral_Review_List_STR=  ','.join(str(v) for v in ED_Neutral_Review_List_Stemmed)
ED_Positive_Review_List_STR=  ','.join(str(v) for v in ED_Positive_Review_List_Stemmed)

# lower strings
ED_Negative_Review_List_STR_Lower=ED_Negative_Review_List_STR.lower()
ED_Neutral_Review_List_STR_Lower=ED_Neutral_Review_List_STR.lower()
ED_Positive_Review_List_STR_Lower=ED_Positive_Review_List_STR.lower()

# remove punctuations to remove performance
ED_Negative_Review_List_STR_Lower = ED_Negative_Review_List_STR_Lower.translate(str.maketrans('', '', string.punctuation))
ED_Neutral_Review_List_STR_Lower = ED_Neutral_Review_List_STR_Lower.translate(str.maketrans('', '', string.punctuation))
ED_Positive_Review_List_STR_Lower = ED_Positive_Review_List_STR_Lower.translate(str.maketrans('', '', string.punctuation))


# check word frequency by each sentiment type of ED comments
def compute_freq(sentence, n_value=1):

    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n_value)
    ngram_fdist = nltk.FreqDist(ngrams)
    return ngram_fdist

freq_dist_ED_Neg = compute_freq(ED_Negative_Review_List_STR_Lower)
freq_dist_ED_Neu = compute_freq(ED_Neutral_Review_List_STR_Lower)
freq_dist_ED_Pos = compute_freq(ED_Positive_Review_List_STR_Lower)


freq_dist_ED_Neg =pd.DataFrame(freq_dist_ED_Neg.items(), columns=['word', 'frequency']) 
freq_dist_ED_Neu =pd.DataFrame(freq_dist_ED_Neu.items(), columns=['word', 'frequency']) 
freq_dist_ED_Pos =pd.DataFrame(freq_dist_ED_Pos.items(), columns=['word', 'frequency']) 

# ED Comment % by sentiment 
# Negative - 30.2% (88/291) of reviews about ED
# Neutral - 18.2% (4/22) of reviews about ED
# Positive - 14.1% (26/185) of reviews about ED

# Most frequent word appeared in Neg ED comments - "Wait"
'''



# Extract only wait related comments
Negative_Group["Wait_Comment"] = Negative_Group["Review"].str.extract("(wait)")[0]
Neutral_Group["Wait_Comment"] = Neutral_Group["Review"].str.extract("(wait)")[0]
Positive_Group["Wait_Comment"] = Positive_Group["Review"].str.extract("(wait)")[0]

Negative_Group['totalwords'] = Negative_Group['Review'].str.split().str.len()
Neutral_Group['totalwords'] = Neutral_Group['Review'].str.split().str.len()
Positive_Group['totalwords'] = Positive_Group['Review'].str.split().str.len()

# Exlcude null values and turn comments into lists
Wait_Negative_Comment=Negative_Group[Negative_Group['Wait_Comment'].notnull()]
Wait_Neutral_Comment=Neutral_Group[Neutral_Group['Wait_Comment'].notnull()]
Wait_Positive_Comment=Positive_Group[Positive_Group['Wait_Comment'].notnull()]

# wait review into lists
Wait_Negative_Review_List= Wait_Negative_Comment['Review'].to_list()
Wait_Neutral_Review_List= Wait_Neutral_Comment['Review'].to_list()
Wait_Positive_Review_List= Wait_Positive_Comment['Review'].to_list()


# Stemming to see most frequently appeared words regardless of pos
from stemming.porter2 import stem
Wait_Negative_Review_List_Stemmed = [[stem(word) for word in sentence.split(" ")] for sentence in Wait_Negative_Review_List]
Wait_Neutral_Review_List_Stemmed = [[stem(word) for word in sentence.split(" ")] for sentence in Wait_Neutral_Review_List]
Wait_Positive_Review_List_Stemmed = [[stem(word) for word in sentence.split(" ")] for sentence in Wait_Positive_Review_List]

# turn lists into strings
Wait_Negative_Review_List_STR=  ','.join(str(v) for v in Wait_Negative_Review_List_Stemmed)
Wait_Neutral_Review_List_STR=  ','.join(str(v) for v in Wait_Neutral_Review_List_Stemmed)
Wait_Positive_Review_List_STR=  ','.join(str(v) for v in Wait_Positive_Review_List_Stemmed)

# lower strings
Wait_Negative_Review_List_STR_Lower=Wait_Negative_Review_List_STR.lower()
Wait_Neutral_Review_List_STR_Lower=Wait_Neutral_Review_List_STR.lower()
Wait_Positive_Review_List_STR_Lower=Wait_Positive_Review_List_STR.lower()

# remove punctuations to remove performance
Wait_Negative_Review_List_STR_Lower = Wait_Negative_Review_List_STR_Lower.translate(str.maketrans('', '', string.punctuation))
Wait_Neutral_Review_List_STR_Lower = Wait_Neutral_Review_List_STR_Lower.translate(str.maketrans('', '', string.punctuation))
Wait_Positive_Review_List_STR_Lower = Wait_Positive_Review_List_STR_Lower.translate(str.maketrans('', '', string.punctuation))


# check word frequency by each sentiment type of wait comments
def compute_freq(sentence, n_value=1):

    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n_value)
    ngram_fdist = nltk.FreqDist(ngrams)
    return ngram_fdist

freq_dist_Wait_Neg = compute_freq(Wait_Negative_Review_List_STR_Lower)
freq_dist_Wait_Neu = compute_freq(Wait_Neutral_Review_List_STR_Lower)
freq_dist_Wait_Pos = compute_freq(Wait_Positive_Review_List_STR_Lower)


freq_dist_Wait_Neg =pd.DataFrame(freq_dist_Wait_Neg.items(), columns=['word', 'frequency']) 
freq_dist_Wait_Neu =pd.DataFrame(freq_dist_Wait_Neu.items(), columns=['word', 'frequency']) 
freq_dist_Wait_Pos =pd.DataFrame(freq_dist_Wait_Pos.items(), columns=['word', 'frequency']) 


#Negative_Group.to_excel('Negative_Group.xlsx', engine='xlsxwriter') 
#Neutral_Group.to_excel('Neutral_Group.xlsx', engine='xlsxwriter') 
#Positive_Group.to_excel('Positive_Group.xlsx', engine='xlsxwriter') 





''' end '''



############################################
###### Graph 11 - Negation #################
############################################

#negation
import re



# lower strings for stopwrods
Negative_Group['Review2']=Negative_Group['Review2'].astype(str)
Negative_Group['Review2']=Negative_Group['Review2'].str.lower()

Neutral_Group['Review2']=Neutral_Group['Review2'].astype(str)
Neutral_Group['Review2']=Neutral_Group['Review2'].str.lower()

Positive_Group['Review2']=Positive_Group['Review2'].astype(str)
Positive_Group['Review2']=Positive_Group['Review2'].str.lower()


Negative_Review_List2= Negative_Group['Review2'].to_list()
Negative_Review_STR2= ' '.join(Negative_Review_List2)

Neutral_Review_List2= Neutral_Group['Review2'].to_list()
Neutral_Review_STR2= ' '.join(Neutral_Review_List2)

Positive_Review_List2= Positive_Group['Review2'].to_list()
Positive_Review_STR2= ' '.join(Positive_Review_List2)

#find negation terms and count (count multiple times if mentioned multiple times in one comment)
### negation - negative reviews
pattern = r"\s+|(?<=\s)'|'(?=\s)|(?<=\w)([,.!?])"
words_Neg = Negative_Review_List2

words_Neg=",".join(str(x) for x in words_Neg)
all_words = [s for s in re.split(pattern, words_Neg) if s]

# find negation words
words_Neg = [w for i,w in enumerate(all_words) if i and (all_words[i-1] in ["doesnt","dont","didnt","without","wont","not","never","no","wasnt","isnt","cant","shouldnt","wouldnt","couldnt","nobody","nothing","neither","nowhere"])]

# find words after negation words
negation_words_Neg = [w for i,w in enumerate(all_words) if i and (all_words[i] in ["doesnt","dont","didnt","without","wont","not","never","no","wasnt","isnt","cant","shouldnt","wouldnt","couldnt","nobody","nothing","neither","nowhere"])]

negation_words_Neg.insert(0, "dont")
negation_phrases_Neg= pd.DataFrame(np.column_stack([negation_words_Neg,words_Neg]), 
                               columns=['Negation','Words'])


# most frequently appeared words after negation words
negation_phrases_Neg['phrase'] = negation_phrases_Neg['Negation'].map(str) + ' ' + negation_phrases_Neg['Words'].map(str) 
negation_phrases_Neg['occurances'] = negation_phrases_Neg.groupby('phrase')['phrase'].transform('count')
negation_phrases_Neg =  negation_phrases_Neg.drop_duplicates()


### negation - neutral reviews
words_Neu = Neutral_Review_List2
words_Neu=",".join(str(x) for x in words_Neu)
all_words = [s for s in re.split(pattern, words_Neu) if s]

words_Neu = [w for i,w in enumerate(all_words) if i and (all_words[i-1] in ["doesnt","dont","didnt","without","wont","not","never","no","wasnt","isnt","cant","shouldnt","wouldnt","couldnt","nobody","nothing","neither","nowhere"])]


negation_words_Neu = [w for i,w in enumerate(all_words) if i and (all_words[i] in ["doesnt","dont","didnt","without","wont","not","never","no","wasnt","isnt","cant","shouldnt","wouldnt","couldnt","nobody","nothing","neither","nowhere"])]
negation_phrases_Neu= pd.DataFrame(np.column_stack([negation_words_Neu,words_Neu]), 
                               columns=['Negation','Words'])

# most frequently appeared words after negation words
negation_phrases_Neu['phrase'] = negation_phrases_Neu['Negation'].map(str) + ' ' + negation_phrases_Neu['Words'].map(str) 
negation_phrases_Neu['occurances'] = negation_phrases_Neu.groupby('phrase')['phrase'].transform('count')
negation_phrases_Neu =  negation_phrases_Neu.drop_duplicates()



### negation - positive reviews
words_Pos = Positive_Review_List2
words_Pos=",".join(str(x) for x in words_Pos)
all_words = [s for s in re.split(pattern, words_Pos) if s]

words_Pos = [w for i,w in enumerate(all_words) if i and (all_words[i-1] in ["doesnt","dont","didnt","without","wont","not","never","no","wasnt","isnt","cant","shouldnt","wouldnt","couldnt","nobody","nothing","neither","nowhere"])]


negation_words_Pos = [w for i,w in enumerate(all_words) if i and (all_words[i] in ["doesnt","dont","didnt","without","wont","not","never","no","wasnt","isnt","cant","shouldnt","wouldnt","couldnt","nobody","nothing","neither","nowhere"])]
negation_phrases_Pos= pd.DataFrame(np.column_stack([negation_words_Pos,words_Pos]), 
                               columns=['Negation','Words'])

# most frequently appeared words after negation words
negation_phrases_Pos['phrase'] = negation_phrases_Pos['Negation'].map(str) + ' ' + negation_phrases_Pos['Words'].map(str) 
negation_phrases_Pos['occurances'] = negation_phrases_Pos.groupby('phrase')['phrase'].transform('count')
negation_phrases_Pos =  negation_phrases_Pos.drop_duplicates()


# negation more in negative reviews; "no one" is the most frequent negation word/phrase among negative reviews


# Pos Tagging

# Negative
negation_phrases_Neg['Pos_Tagging'] = nltk.pos_tag(negation_phrases_Neg['Words'])
negation_phrases_Neg['Pos_Tagging'] = negation_phrases_Neg['Pos_Tagging'].astype(str)
negation_phrases_Neg['Pos_Tagging_C'] = negation_phrases_Neg['Pos_Tagging'].str.rsplit(',').str[-1] 


negation_phrases_Neg['Pos_Tagging_C'] = negation_phrases_Neg['Pos_Tagging_C'].astype(str)
negation_phrases_Neg['Pos_Tagging_C'] = negation_phrases_Neg['Pos_Tagging_C'].str.strip(')')

# neg_pos_tagging_count
neg_pos_tagging_count = negation_phrases_Neg['Pos_Tagging_C'].value_counts(dropna=False)


# Neutral
negation_phrases_Neu['Pos_Tagging'] = nltk.pos_tag(negation_phrases_Neu['Words'])
negation_phrases_Neu['Pos_Tagging'] = negation_phrases_Neu['Pos_Tagging'].astype(str)
negation_phrases_Neu['Pos_Tagging_C'] = negation_phrases_Neu['Pos_Tagging'].str.rsplit(',').str[-1] 


negation_phrases_Neu['Pos_Tagging_C'] = negation_phrases_Neu['Pos_Tagging_C'].astype(str)
negation_phrases_Neu['Pos_Tagging_C'] = negation_phrases_Neu['Pos_Tagging_C'].str.strip(')')

# neu_pos_tagging_count
neu_pos_tagging_count = negation_phrases_Neu['Pos_Tagging_C'].value_counts(dropna=False)


# Positive
negation_phrases_Pos['Pos_Tagging'] = nltk.pos_tag(negation_phrases_Pos['Words'])
negation_phrases_Pos['Pos_Tagging'] = negation_phrases_Pos['Pos_Tagging'].astype(str)
negation_phrases_Pos['Pos_Tagging_C'] = negation_phrases_Pos['Pos_Tagging'].str.rsplit(',').str[-1] 


negation_phrases_Pos['Pos_Tagging_C'] = negation_phrases_Pos['Pos_Tagging_C'].astype(str)
negation_phrases_Pos['Pos_Tagging_C'] = negation_phrases_Pos['Pos_Tagging_C'].str.strip(')')


# pos_pos_tagging_count
pos_pos_tagging_count = negation_phrases_Pos['Pos_Tagging_C'].value_counts(dropna=False)


# POS Tagging Definition - https://cs.nyu.edu/~grishman/jet/guide/PennPOS.html#:~:text=MD%20Modal,NN%20Noun%2C%20singular%20or%20mass


#negation_phrases_Neg.to_excel('negation_phrases_Neg.xlsx', engine='xlsxwriter') 
#negation_phrases_Neu.to_excel('negation_phrases_Neu.xlsx', engine='xlsxwriter') 
#negation_phrases_Pos.to_excel('negation_phrases_Pos.xlsx', engine='xlsxwriter') 


#find negation terms and count (only once per comment no matter how many times mentioned in one comment)
words_list = ["doesnt","dont","didnt","without","wont","not","never","no","wasnt","isnt","cant","shouldnt","wouldnt","couldnt","nobody","nothing","neither","nowhere"]

search_ = re.compile("\\b%s\\b" % "\\b|\\b".join(words_list))

Negative_Group['matches'] = Negative_Group.Review2.str.findall(search_)
Neutral_Group['matches'] = Neutral_Group.Review2.str.findall(search_)
Positive_Group['matches'] = Positive_Group.Review2.str.findall(search_)


Negative_Group['negation count'] = Negative_Group['matches'].apply(len)
Neutral_Group['negation count'] = Neutral_Group['matches'].apply(len)
Positive_Group['negation count'] = Positive_Group['matches'].apply(len)

#neg
#extract exact negation words
exactmatch = negation_phrases_Neg['phrase'] 
pattern = fr'\b({"|".join(exactmatch)})\b' 
Negative_Group['exact_word'] = Negative_Group['Review2'].str.findall(pattern).str.join(",")
Negative_Group['exact_word'] = Negative_Group['exact_word'].str.split(',').apply(lambda x : ','.join(set(x)))


neg_ext=Negative_Group['exact_word'].str.split(',')
neg_set=[]
for i in neg_ext.dropna():
    neg_set.extend(i)
negative_negation_count=pd.Series(neg_set).value_counts().sort_values(ascending=False).to_frame() ##### even though one comment has same, multiple negations (i.e. no one, no one), it counts only once per comment ######

#negative_negation_count.to_excel('negative_negation_count.xlsx', engine='xlsxwriter') 
#Negative_Group.to_excel('Negative_Reviews.xlsx', engine='xlsxwriter') 


#neu
#extract exact negation words
exactmatch = negation_phrases_Neu['phrase'] 
pattern = fr'\b({"|".join(exactmatch)})\b' 
Neutral_Group['exact_word'] = Neutral_Group['Review2'].str.findall(pattern).str.join(",")
Neutral_Group['exact_word'] = Neutral_Group['exact_word'].str.split(',').apply(lambda x : ','.join(set(x)))


neu_ext=Neutral_Group['exact_word'].str.split(',')
neu_set=[]
for i in neu_ext.dropna():
    neu_set.extend(i)
neutral_negation_count=pd.Series(neu_set).value_counts().sort_values(ascending=False).to_frame() ##### even though one comment has same, multiple negations (i.e. no one, no one), it counts only once per comment ######

#neutral_negation_count.to_excel('neutral_negation_count.xlsx', engine='xlsxwriter') 
#Neutral_Group.to_excel('Neutral_Reviews.xlsx', engine='xlsxwriter') 


#pos
#extract exact negation words
exactmatch = negation_phrases_Pos['phrase'] 
pattern = fr'\b({"|".join(exactmatch)})\b' 
Positive_Group['exact_word'] = Positive_Group['Review2'].str.findall(pattern).str.join(",")
Positive_Group['exact_word'] = Positive_Group['exact_word'].str.split(',').apply(lambda x : ','.join(set(x)))


pos_ext=Positive_Group['exact_word'].str.split(',')
pos_set=[]
for i in pos_ext.dropna():
    pos_set.extend(i)
positive_negation_count=pd.Series(pos_set).value_counts().sort_values(ascending=False).to_frame() ##### even though one comment has same, multiple negations (i.e. no one, no one), it counts only once per comment ######

#positive_negation_count.to_excel('positive_negation_count.xlsx', engine='xlsxwriter') 
#Positive_Group.to_excel('Positive_Reviews.xlsx', engine='xlsxwriter') 



''' end '''

###################################
#### Graph 8 - Topic Modeling #####
###################################

# can improve the performance by removing less important pos words; did not use this since neutral group pos words are usually less important

# Importing Gensim
import gensim
from gensim import corpora

#Tokenize the sentence into words
neg_tokens = [word for word in Negative_Reviews_Cleaned.split()]
neu_tokens = [word for word in Neutral_Reviews_Cleaned.split()]
pos_tokens = [word for word in Positive_Reviews_Cleaned.split()]

#Create dictionary
neg_dictionary = corpora.Dictionary([neg_tokens])
neg_doc_term_matrix = [neg_dictionary.doc2bow(doc.split()) for doc in neg_tokens]

neu_dictionary = corpora.Dictionary([neu_tokens])
neu_doc_term_matrix = [neu_dictionary.doc2bow(doc.split()) for doc in neu_tokens]

pos_dictionary = corpora.Dictionary([pos_tokens])
pos_doc_term_matrix = [pos_dictionary.doc2bow(doc.split()) for doc in pos_tokens]


# LDA model
Lda = gensim.models.ldamodel.LdaModel

# passes = 50;  higher-quality topics
neg_ldamodel = Lda(neg_doc_term_matrix, num_topics=3, id2word = neg_dictionary, passes=50) 
neu_ldamodel = Lda(neu_doc_term_matrix, num_topics=3, id2word = neu_dictionary, passes=50) 
pos_ldamodel = Lda(pos_doc_term_matrix, num_topics=3, id2word = pos_dictionary, passes=50) 

print(neg_ldamodel.print_topics(num_topics=3, num_words=3))
print(neu_ldamodel.print_topics(num_topics=3, num_words=3))
print(pos_ldamodel.print_topics(num_topics=3, num_words=3))

''' end '''

###################################
### Graph 5 - Family Reviews ######
###################################


#Negative_Reviews['My']  = np.where(Negative_Reviews.Review.str.contains(r'\b(my)\b'), 'my', 'na')
my = 'my\W+(?P<after>(?:\w+\W+){,2})' 
Negative_Reviews['Words_After_My']=Negative_Reviews.Review.str.extract(my, expand=True)
Neutral_Reviews['Words_After_My']=Neutral_Reviews.Review.str.extract(my, expand=True)
Positive_Reviews['Words_After_My']=Positive_Reviews.Review.str.extract(my, expand=True)



#Negative_Reviews['Our']  = np.where(Negative_Reviews.Review.str.contains(r'(^|\s+)our(\s+)?'), 'our', 'na')
our = 'our\W+(?P<after>(?:\w+\W+){,2})' 
Negative_Reviews['Words_After_Our']=Negative_Reviews.Review.str.extract(our, expand=True)
Neutral_Reviews['Words_After_Our']=Neutral_Reviews.Review.str.extract(our, expand=True)
Positive_Reviews['Words_After_Our']=Positive_Reviews.Review.str.extract(our, expand=True)

# create a family dictionary
family_list=['grandparent','wife','husband','kid','girlfriend','boyfriend','loved one','relative','child','uncle',
'aunt','son','daughter','father','mother','mom','dad','grandma','grandmother','grandpa','grandfather','baby','brother','sister','children','family','fiance','friend' ]
family_list_match = "|".join(family_list)

# if it's a review by family then 1 else 0
Negative_Reviews['Family_Review'] =Negative_Reviews[['Words_After_My', 'Words_After_Our']].apply(lambda x: x.str.contains(family_list_match,case=False)).any(axis=1).astype(int)
Neutral_Reviews['Family_Review'] =Neutral_Reviews[['Words_After_My', 'Words_After_Our']].apply(lambda x: x.str.contains(family_list_match,case=False)).any(axis=1).astype(int)
Positive_Reviews['Family_Review'] =Positive_Reviews[['Words_After_My', 'Words_After_Our']].apply(lambda x: x.str.contains(family_list_match,case=False)).any(axis=1).astype(int)


# family review count by sentiment
neg_family_review_count = Negative_Reviews['Family_Review'].value_counts(dropna=False) # family review - 76 (28%) ; no family review - 195 (72%)
neu_family_review_count = Neutral_Reviews['Family_Review'].value_counts(dropna=False) # family review - 2 (12%) ; no family review - 15 (88%)
pos_family_review_count = Positive_Reviews['Family_Review'].value_counts(dropna=False) # family review - 32 (24%) ; no family review - 100 (76%) 

#overall 26% - family review

''' end '''
''' hospital response sentiment - not reliable
nltk.download('opinion_lexicon')

from nltk.corpus import opinion_lexicon

pos_list=set(opinion_lexicon.positive())
neg_list=set(opinion_lexicon.negative())

pos_list = list(pos_list)
neg_list = list(neg_list)


df1 = Reviews['Hospital Response'].str.split(' ', expand=True)
neg_hos_res = df1.isin(neg_list).any(axis=1)


pos_hos_res= df1.isin(pos_list).any(axis=1)
Reviews['Hospital_Response_Sentiment'] = np.select([neg_hos_res, pos_hos_res], ['neg','pos'], default='non')
'''

