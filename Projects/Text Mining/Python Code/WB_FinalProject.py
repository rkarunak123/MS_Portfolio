# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:04:55 2019

@author: wahdi
"""

import csv
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

#create csv file in utf-8 format to more easily manipulate poems split out in various text files
with open("C:\\Users\\wahdi\\OneDrive\\IST 736\\FinalProject\\shakespeare.csv",'w',encoding='utf-8') as outfile:
   writer = csv.writer(outfile)
   for path, dirs, files in os.walk("C:\\Users\\wahdi\\OneDrive\\IST 736\\FinalProject\\authors\\"):
        for filename in files:
           with open (path + "\\" +filename, 'r') as infile:
               lines = [line.strip() for line in infile if line.strip()]
               writer.writerow([path, filename, lines])

#import data from csv file and split into three lists
colnames = ['author','title','poem']
data = pd.read_csv("C:\\Users\\wahdi\\OneDrive\\IST 736\\FinalProject\\shakespeare.csv",names= colnames)

#validate data pulled from csv correctly
#print(data)

#clean up author column by stripping filepath information
author_clean = []
for row in data.author:
    row = row.replace("C:\\Users\\wahdi\\OneDrive\\IST 736\\FinalProject\\authors\\","")
    author_clean.append(row)
#print(author_clean)

#clean up poem title column by stripping file extension ".txt"
title_clean = []
for row in data.title:
    row = row.replace(".txt","")
    title_clean.append(row)
#print(title_clean)
    
#clean the poem text to prep it for tokenization
poem_clean = []
for row in data.poem:
    row=row.lower()
    #row=row.replace("\\","") #remove back slashes
    row=row.replace(",","") #remove commas
    row=row.replace(".","") #remove peroids
    row=row.replace(";","") #remove semi-colons
    row=row.replace(":","") #remove colons
    row=row.replace("-","") #remove hashes
    row=row.replace("[","") #remove open brackets
    row=row.replace("]","") #remove close brackets
    row=row.replace("?","") #remove question marks
    row=row.replace("!","") #remove exclamation points
    row=row.replace("'","") #remove apostrophes
    row=row.replace("\"","") #remove double quote marks    
    row=row.replace("“","") #remove open quotation mark
    row=row.replace("”","") #remove close quotation mark
    poem_clean.append(row)
#print(poem_clean)

y = author_clean
X = poem_clean

# split data into test and train sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  unigram boolean vectorizer, set minimum document frequency to 4
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=4, stop_words='english')
#  unigram term frequency vectorizer, set minimum document frequency to 4
unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=4, stop_words='english')
#  unigram and bigram term frequency vectorizer, set minimum document frequency to 4
gram12_count_vectorizer = CountVectorizer(encoding='latin-1', ngram_range=(1,2), min_df=4, stop_words='english')

# fit vocabulary in training documents and transform the training documents into vectors
X_unigram_bool_train_vec = unigram_bool_vectorizer.fit_transform(X_train)
X_unigram_count_train_vec = unigram_count_vectorizer.fit_transform(X_train)
X_gram12_count_train_vec = gram12_count_vectorizer.fit_transform(X_train)

# use the vocabulary constructed from the training data to vectorize the test data. 
X_unigram_bool_test_vec = unigram_bool_vectorizer.transform(X_test)
X_unigram_count_test_vec = unigram_count_vectorizer.transform(X_test)
X_gram12_count_test_vec = gram12_count_vectorizer.transform(X_test)

# initialize the MNB models
nb_unigram_bool_clf= MultinomialNB()
nb_unigram_count_clf= MultinomialNB()
nb_gram12_count_clf= MultinomialNB()
# use the training data to train the MNB model
nb_unigram_bool_clf.fit(X_unigram_bool_train_vec,y_train)
nb_unigram_count_clf.fit(X_unigram_count_train_vec,y_train)
nb_gram12_count_clf.fit(X_gram12_count_train_vec,y_train)

# initialize the SVM models
svm_unigram_bool_clf= LinearSVC(C=1)
svm_unigram_count_clf= LinearSVC(C=1)
svm_gram12_count_clf= LinearSVC(C=1)
# use the training data to train the MNB model
svm_unigram_bool_clf.fit(X_unigram_bool_train_vec,y_train)
svm_unigram_count_clf.fit(X_unigram_count_train_vec,y_train)
svm_gram12_count_clf.fit(X_gram12_count_train_vec,y_train)

# print MNB confusion matrixes (row: ground truth; col: prediction)
y_nb_unigram_bool_pred = nb_unigram_bool_clf.fit(X_unigram_bool_train_vec, y_train).predict(X_unigram_bool_test_vec)
cm_unigram_bool_nb=confusion_matrix(y_test, y_nb_unigram_bool_pred, labels=['Christopher Marlowe','Edward de Vere','William Shakespeare'])
print("\n","MNB unigram bool model confusion matrix: ", "\n", cm_unigram_bool_nb)
y_nb_unigram_count_pred = nb_unigram_count_clf.fit(X_unigram_count_train_vec, y_train).predict(X_unigram_count_test_vec)
cm_unigram_count_nb=confusion_matrix(y_test, y_nb_unigram_count_pred, labels=['Christopher Marlowe','Edward de Vere','William Shakespeare'])
print("\n","MNB unigram count model confusion matrix: ", "\n", cm_unigram_count_nb)
y_nb_gram12_count_pred = nb_gram12_count_clf.fit(X_gram12_count_train_vec, y_train).predict(X_gram12_count_test_vec)
cm_gram12_count_nb=confusion_matrix(y_test, y_nb_gram12_count_pred, labels=['Christopher Marlowe','Edward de Vere','William Shakespeare'])
print("\n","MNB unigram + bigram count model confusion matrix: ", "\n", cm_gram12_count_nb)

# print SVM confusion matrixes (row: ground truth; col: prediction)
y_svm_unigram_bool_pred = svm_unigram_bool_clf.fit(X_unigram_bool_train_vec, y_train).predict(X_unigram_bool_test_vec)
cm_unigram_bool_svm=confusion_matrix(y_test, y_svm_unigram_bool_pred, labels=['Christopher Marlowe','Edward de Vere','William Shakespeare'])
print("\n","SVM unigram bool model confusion matrix: ", "\n", cm_unigram_bool_svm)
y_svm_unigram_count_pred = svm_unigram_count_clf.fit(X_unigram_count_train_vec, y_train).predict(X_unigram_count_test_vec)
cm_unigram_count_svm=confusion_matrix(y_test, y_svm_unigram_count_pred, labels=['Christopher Marlowe','Edward de Vere','William Shakespeare'])
print("\n","SVM unigram count model confusion matrix: ", "\n", cm_unigram_count_svm)
y_svm_gram12_count_pred = svm_gram12_count_clf.fit(X_gram12_count_train_vec, y_train).predict(X_gram12_count_test_vec)
cm_gram12_count_svm=confusion_matrix(y_test, y_svm_gram12_count_pred, labels=['Christopher Marlowe','Edward de Vere','William Shakespeare'])
print("\n","SVM unigram + bigram count model confusion matrix: ", "\n", cm_gram12_count_svm)

# print MNB classification reports
target_names_sent = ['Christopher Marlowe','Edward de Vere','William Shakespeare']
print("\n","Author MNB unigram bool model classification report: ", "\n",classification_report(y_test, y_nb_unigram_bool_pred, target_names=target_names_sent))
print("Author MNB unigram count model classification report: ", "\n",classification_report(y_test, y_nb_unigram_count_pred, target_names=target_names_sent))
print("Author MNB unigram + bigram count model classification report: ", "\n",classification_report(y_test, y_nb_gram12_count_pred, target_names=target_names_sent))

# print SVM classification reports
target_names_sent = ['Christopher Marlowe','Edward de Vere','William Shakespeare']
print("\n","Author SVM unigram bool model classification report: ", "\n",classification_report(y_test, y_svm_unigram_bool_pred, target_names=target_names_sent))
print("Author SVM unigram count model classification report: ", "\n",classification_report(y_test, y_svm_unigram_count_pred, target_names=target_names_sent))
print("Author SVM unigram + bigram count model classification report: ", "\n",classification_report(y_test, y_svm_gram12_count_pred, target_names=target_names_sent))

## find the MNB calculated posterior probability 
posterior_probs_unigram_bool_nb = nb_unigram_bool_clf.predict_proba(X_unigram_bool_test_vec)
posterior_probs_unigram_count_nb = nb_unigram_count_clf.predict_proba(X_unigram_count_test_vec)
posterior_probs_gram12_count_nb = nb_gram12_count_clf.predict_proba(X_gram12_count_test_vec)
## find the MNB posterior probabilities
print("Unigram bool posterior probabilities for authorship: \n",posterior_probs_unigram_bool_nb)
print("Unigram count posterior probabilities for authorship: \n",posterior_probs_unigram_count_nb)
print("Unigram + bigram count posterior probabilities for authorship: \n",posterior_probs_gram12_count_nb)

