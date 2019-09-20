'''
  This program shell reads tweet data for the twitter sentiment classification problem.
  The input to the program is the path to the Semeval directory "corpus" and a limit number.
  The program reads the first limit number of tweets
  It creates a "tweetdocs" variable with a list of tweets consisting of a pair
    with the list of tokenized words from the tweet and the label pos, neg or neu.
  It prints a few example tweets, as text and as tokens.
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifySemevalTweets.py  <corpus directory path> <limit number>
  
  Rebecca Karunakaran
  NLP Final Project
'''
# open python and nltk packages needed for processing
# while the semeval tweet task b data has tags for "positive", "negative", 
#  "objective", "neutral", "objective-OR-neutral", we will combine the last 3 into "neutral"
import os
import sys
import nltk
import re
import string
from nltk.tokenize import TweetTokenizer
from sklearn import svm
import random
from nltk.collocations import *

# define a feature definition function here

tweetdata = []
tweetdocs = []

# function to read tweet training file, train and test a classifier 
def processtweets(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  # initialize NLTK built-in tweet tokenizer
  twtokenizer = TweetTokenizer()
  
  os.chdir(dirPath)
  
  f = open('./downloaded-tweeti-b-dist.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  #    assuming that the tweets are sufficiently randomized
  
  for line in f:
    if (len(tweetdata) < limit):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the tweet and user ids, and keep the sentiment and tweet text
      tweetdata.append(line.split('\t')[2:4])
  
  for tweet in tweetdata[:10]:
    print (tweet)
    
  print("\n\nTotal number of tweets is ", len(tweetdata),"\n\n")
  
  # create list of tweet documents as (list of words, label)
  # where the labels are condensed to just 3:  'pos', 'neg', 'neu'
 
  # add all the tweets except the ones whose text is Not Available
  for tweet in tweetdata:
    if (tweet[1] != 'Not Available'):
      # run the tweet tokenizer on the text string - returns unicode tokens, so convert to utf8
      tokens = twtokenizer.tokenize(tweet[1])

      if tweet[0] == '"positive"':
        label = 'pos'
      else:
        if tweet[0] == '"negative"':
          label = 'neg'
        else:
          if (tweet[0] == '"neutral"') or (tweet[0] == '"objective"') or (tweet[0] == '"objective-OR-neutral"'):
            label = 'neu'
          else:
            label = ''
      tweetdocs.append((tokens, label))
  
  # print a few
  print("Sample 3 tweets -")
  for tweet in tweetdocs[:3]:
    print (tweet)
    
  #RK added
  print("\nTotal number of tweets after condensing the labels is ", len(tweetdocs))
  
  # get all words from the tweets and put into a frequency distribution -no preprocessing so far
  all_words_list = [word for (sent,cat) in tweetdocs for word in sent]
   
  print("\nTotal number of words is ", len(all_words_list))
  
  random.shuffle(all_words_list )
  
  all_words = nltk.FreqDist(all_words_list)

  # get the 1000 most frequently appearing keywords in the tweets
  featurewordcnt = 1000
  print(" The number of words used in the feature ", featurewordcnt)
  word_items = all_words.most_common(featurewordcnt)
  word_features = [word for (word,count) in word_items]
  print("\n10 most frequently appearing keywords is ")
  print(word_features[:10])

  # define features (keywords) of a document for a BOW/unigram baseline
  # each feature is 'contains(keyword)' and is true or false depending
  # on whether that keyword is in the document
  def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features

  # get features sets for a document, including keyword features and category feature
  featuresets = [(document_features(d, word_features), c) for (d, c) in tweetdocs]
  
  print(featuresets[0])
  
  # training using naive Baysian classifier, training set is approximately 90% of data
  train_set, test_set = featuresets[100:], featuresets[:100]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  
  # evaluate the accuracy of the classifier
  base_accuracy = nltk.classify.accuracy(classifier, test_set)
  print("Baseline Accuracy with no pre-processing is ", base_accuracy,"\n")

#  #PreProcessing
#  #-----------------------------------------------------------------------
#    
  # read stop words from file if used
  stopwords = [line.strip() for line in open('C:\Syracuse\FinalProjectData\SemEval2014TweetData\stopwords_twitter.txt')]
  stop_words = set(stopwords)
 
 
  #Remove punctuations and stop word
  all_words_list = [word for (sent,cat) in tweetdocs for word in sent if (word not in stop_words and word not in string.punctuation )]
  print('\n\n',all_words_list[:500])
  
  def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt 
   
  all_words_list1=[]
  all_words_list2=[]
  all_words_list3=[]
  
  #Removing Twitter Handles (@user)
  for word in all_words_list :
      all_words_list1.append( re.sub('@[^\s]+','',word))
 # print('\n\n',all_words_list1[:500])
  
  #Remove urls 
  for word in all_words_list1 :
      all_words_list2.append( re.sub(r"http\S+",'',word))
 # print('\n\n',all_words_list2[:500])
  
  #Remove emojis
  for word in all_words_list2 :
      all_words_list3.append( re.sub('\:\)|\;\)|\.*','',word))
  #print('\n\n',all_words_list3[:500])
  
  #final list of words
  all_words_list =  [word for word in all_words_list3 if word != '']
  print('\n\n',all_words_list[:500])
#  
  print("total number of words after preprocessing is ", len(all_words_list))
  
  all_words = nltk.FreqDist(all_words_list)

  # get the 1000 most frequently appearing keywords in the tweets
  word_items = all_words.most_common(1000)
  word_features_pp = [word for (word,count) in word_items]
  
  print(" \nTop 10 most frequent words after pre-processing -")
  print(word_features_pp[:10])
  
   # get features sets for a document, including keyword features and category feature
  featuresets = [(document_features(d, word_features_pp), c) for (d, c) in tweetdocs]
  
  # training using naive Baysian classifier, training set is approximately 90% of data
  train_set, test_set = featuresets[100:], featuresets[:100]
  classifier_p = nltk.NaiveBayesClassifier.train(train_set)
  
    
  print("\n10 Most Informative Features  ",classifier_p.show_most_informative_features(10))
  
  
  # evaluate the accuracy of the classifier
  base_accuracy = nltk.classify.accuracy(classifier_p, test_set)
  print("Baseline Accuracy with after pre-processing is ", base_accuracy,"\n")
# 
#---------------end preprocessing

#------------Common functions
  #Function to do cross_validation
  def cross_validation(num_folds, featuresets):
        subset_size = int(len(featuresets)/num_folds)
        print('each fold size', subset_size)
        accuracy_list = []
        # iterate over the folds
        for i in range(num_folds):
          test_this_round = featuresets[i*subset_size:][:subset_size]
          train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
          # train using train_this_round
          classifier = nltk.NaiveBayesClassifier.train(train_this_round)
          # evaluate against test_this_round and save accuracy
          accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
          print (i, accuracy_this_round)
          accuracy_list.append(accuracy_this_round)
        # find mean accuracy over all rounds
        print ('Accuracy with 10 fold Cross Validation ', sum(accuracy_list) / num_folds)
        
  # Function to compute precision, recall and F1 for each label
  def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label 
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        #print("\n\n TP",TP, "FP", FP, "FN",FN, "FP",FP)
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))
        
    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

 #------------------------POS Tag feature--------------------------------------------- 
  #POS Tag feature

  def POS_features(document,word_feature):
     document_words = set(document)
     tagged_words = nltk.pos_tag(document)
     #print(tagged_words)
     features = {}
     for word in word_features:
   	     features['contains({})'.format(word)] = (word in document_words)
            
     numNoun = 0  
     numVerbBase = 0
     numVerbPast = 0
     numVerbPresentP = 0
     numVerbPastP = 0
     numVerbN3PS = 0
     numVerb3P = 0
     numAdj = 0
     numAdverb = 0
     numPreposition = 0
     numDeterminer = 0
     numTo = 0
     numInterjection =0
     numCardinalNum = 0
     numCoordinatingConj = 0
     numPronoun = 0
    	
     for (word, tag) in tagged_words:
         if tag.startswith('N'): numNoun += 1
         if tag == 'VB': numVerbBase += 1
         if tag == 'VBD': numVerbPast += 1
         if tag == 'VBG': numVerbPresentP += 1
         if tag == 'VBN': numVerbPastP += 1
         if tag == 'VBP': numVerbN3PS += 1
         if tag == 'VBZ': numVerb3P += 1
         if tag.startswith('J'): numAdj += 1
         if tag.startswith('R'): numAdverb += 1
         if tag.startswith('I'): numPreposition += 1  
         if tag.startswith('D'): numDeterminer += 1
         if tag=='TO' : numTo += 1
         if tag=='UH' : numInterjection += 1
         if tag=='CD' : numCardinalNum += 1
         if tag=='CC' : numCoordinatingConj += 1
         if tag.startswith('PR'): numPronoun += 1
         
     features['nouns'] = numNoun
     features['verbsbase'] = numVerbBase
     features['verbspast'] = numVerbPast
     features['verbspresentp'] = numVerbPresentP
     features['verbspastp'] = numVerbPastP
     features['verbsN3PS'] = numVerbN3PS
     features['verbs3P'] = numVerb3P
     features['adjectives'] = numAdj
     features['adverbs'] = numAdverb
     features['prepoistion'] = numPreposition
     features['determiner'] = numDeterminer
     features['to'] = numTo
     features['interjection'] = numInterjection
     features['cardinalnum'] = numCardinalNum
     features['coordinatingconj'] = numCoordinatingConj
     features['pronoun'] = numPronoun
     return features

  # Try out the POS features.
  POS_featuresets = [(POS_features(d, word_features_pp), c) for (d, c) in tweetdocs]
  
  # number of features for document 0
  print("Length of POS ", len(POS_featuresets[0][0].keys()))
  
  #Show the first sentence in your (randomly shuffled) documents and look at its POS tag features.
  print("\n\n",tweetdocs[1])
  # the pos tag features for this sentence
  print('num nouns ', POS_featuresets[1][0]['nouns'])
  print('num verbsbase', POS_featuresets[1][0]['verbsbase'])
  print('num verbspast', POS_featuresets[1][0]['verbspast'])
  print('num adjectives', POS_featuresets[1][0]['adjectives'])
  print('num adverbs', POS_featuresets[1][0]['adverbs'])
  print('num pronoun', POS_featuresets[1][0]['pronoun'])
  print('determiner', POS_featuresets[1][0]['determiner'])
  print(POS_featuresets[1])
      

  #split data into a training set and a test set, using a 90%/10% split
  size = int(len(POS_featuresets) * 0.1)
  train_set, test_set = POS_featuresets[size:], POS_featuresets[:size]

  # train classifier on the training set
  classifier_pos = nltk.NaiveBayesClassifier.train(train_set)
  
  print("\n10 Most Informative Features -POS ",classifier_pos.show_most_informative_features(10))
  

  # evaluate the accuracy with Hold out method
  Accuracy_pos = nltk.classify.accuracy(classifier_pos, test_set)
  print(" Accuracy with POS tagging HoldOut ", Accuracy_pos,"\n")
  
        
  #Run the cross-validation on our word feature sets with 10 folds.
  cross_validation(10, POS_featuresets)
  

#build the reference and test lists from the classifier on the test set
  predictedlist = []
  goldlist = []
  for (features, label) in test_set:
     goldlist.append(label)
     predictedlist.append(classifier_pos.classify(features))
     
  print("\n\n Gold List size -", len(goldlist))
  print("\n\n Predicted List size -", len(predictedlist))
  
  #Now we use the NLTK function to define the confusion matrix, and we print it out:

  cm = nltk.ConfusionMatrix(goldlist, predictedlist)
  print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
        
  eval_measures(goldlist, predictedlist)
  
#------------------POS Tag feature---------------------

#-------Subjectivity--------------------------------------
  
  SLpath = "C:/Syracuse/subjclueslen1-HLTEMNLP05.tff"
  
  def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
        #print(word,sldict[word])
    return sldict

  SL = readSubjectivity(SLpath)

  # define features that include word counts of subjectivity words
  def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    weakNeu = 0
    strongNeu = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            if strength == 'weaksubj' and polarity == 'neutral':
                weakNeu += 1
            if strength == 'strongsubj' and polarity == 'neutral':
                strongNeu += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)     
            features['neutralcount'] = weakNeu + (2 * strongNeu)   
    return features

  SL_featuresets = [(SL_features(d, word_features_pp, SL), c) for (d, c) in tweetdocs]
  
  print(SL_featuresets[0][1])
  
  #split data into a training set and a test set, using a 90%/10% split
  size = int(len(SL_featuresets) * 0.1)
  train_set, test_set = SL_featuresets[size:], SL_featuresets[:size]

   # retrain the classifier using these features
  classifier_subj = nltk.NaiveBayesClassifier.train(train_set)
  
  print("\n10 Most Informative Features -Subjectivity ",classifier_subj.show_most_informative_features(10))
  
  Accuracy_Subj=nltk.classify.accuracy(classifier_subj, test_set)
  print(" \nAccuracy using Subjectivity- HoldOut ", Accuracy_Subj,"\n")
  
  #  #Run the cross-validation on our word feature sets with 10 folds.
  cross_validation(10, SL_featuresets)
  
  #build the reference and test lists from the classifier on the test set
  predictedlist = []
  goldlist = []
  for (features, label) in test_set:
     goldlist.append(label)
     predictedlist.append(classifier_subj.classify(features))
     
  print("\n\n Gold List size -", len(goldlist))
  print("\n\n Predicted List size -", len(predictedlist))
  
  #Now we use the NLTK function to define the confusion matrix, and we print it out:

  cm = nltk.ConfusionMatrix(goldlist, predictedlist)
  print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
        
  eval_measures(goldlist, predictedlist)
  
#-----------------end Subjectivity--------------------------

#----------Bing Liu Opinion Lexicon
  neglist = []
  poslist = []
   
  NegFile = open('C:/Syracuse/opinion-lexicon-English/negative-words.txt', 'r')
  PosFile = open('C:/Syracuse/opinion-lexicon-English/positive-words.txt', 'r')
    # initialize an empty dictionary
    
  for word in NegFile:
      neglist.append(word.strip())
  for word in PosFile:
      poslist.append(word.strip())
  print(len(neglist))

  
  # define features that include word counts of positive and negative opinion
  def Op_features(document, word_features):
    
    document_words = set(document)
    features = {}
    for word in word_features:
   	     features['contains({})'.format(word)] = (word in document_words)
    # count variables for postive ,negative and neutral opinion words
    numPos = 0
    numNeg = 0
    numNeu = 0
    
    for word in document_words:
        if word.lower() in neglist:
            numNeg = numNeg +1
        else:
                if word.lower() in poslist:
                    numPos = numPos +1
                else:
                        numNeu = numNeu +1
        
    features['Positive'] = numPos
    features['Negative'] = numNeg
    features['Neutral'] = numNeu
    return features

  # Try out the  features.
  Op_featuresets = [(Op_features(d, word_features_pp), c) for (d, c) in tweetdocs]

  #Show the first sentence in your tweet
  print("\n\n",tweetdocs[1])
  # the pos tag features for this sentence
  print('Number Postive Opinion ', Op_featuresets[1][0]['Positive'])
  print('Number Negative Opinion ', Op_featuresets[1][0]['Negative'])
  print('Number Neutral Opinion ', Op_featuresets[1][0]['Neutral'])
  
  #split data into a training set and a test set, using a 90%/10% split
  size = int(len(Op_featuresets) * 0.1)
  train_set, test_set = Op_featuresets[size:], Op_featuresets[:size]

  # train classifier on the training set
  classifier_op = nltk.NaiveBayesClassifier.train(train_set)
  
  
  print("\n10 Most Informative Features -Opinion ",classifier_op.show_most_informative_features(20))
  
  Accuracy_Op=nltk.classify.accuracy(classifier_op, test_set)
  print(" \nAccuracy using Opinion- HoldOut ", Accuracy_Op,"\n")
  
  #Run the cross-validation on our word feature sets with 10 folds.
  cross_validation(10, Op_featuresets)
  
  #build the reference and test lists from the classifier on the test set
  predictedlist = []
  goldlist = []
  for (features, label) in test_set:
     goldlist.append(label)
     predictedlist.append(classifier_op.classify(features))
     
  print("\n\n Gold List size -", len(goldlist))
  print("\n\n Predicted List size -", len(predictedlist))
  
  #Now we use the NLTK function to define the confusion matrix, and we print it out:

  cm = nltk.ConfusionMatrix(goldlist, predictedlist)
  print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
        
  eval_measures(goldlist, predictedlist)
  
  #end ----------Bing Liu Opinion Lexicon-----------------------------
  
  #-------------Combining bigrams with Opinion Lexicon------------------------
  

  bigram_measures = nltk.collocations.BigramAssocMeasures()
  
  finder = nltk.collocations.BigramCollocationFinder.from_words(all_words_list)
  
  #chi-squared measure to get bigrams that are informative features. 
  bigram_features = finder.nbest(bigram_measures.pmi, 500)
  
  # define features that include word counts of positive and negative opinion

  def Com_features(document, word_features, bigram_features):
    
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    for word in word_features:
   	     features['contains({})'.format(word)] = (word in document_words)
    # count variables for postive ,negative and neutral opinion words
    numPos = 0
    numNeg = 0
    numNeu = 0
    
    for word in document_words:
        if word.lower() in neglist:
            numNeg = numNeg +1
        else:
                if word.lower() in poslist:
                    numPos = numPos +1
                else:
                        numNeu = numNeu +1
        
    features['Positive'] = numPos
    features['Negative'] = numNeg
    features['Neutral'] = numNeu
    
    for bigram in bigram_features:
        features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    

    return features

  # Try out the  features.
  Com_featuresets = [(Com_features(d, word_features_pp,bigram_features), c) for (d, c) in tweetdocs]
   #split data into a training set and a test set, using a 90%/10% split
  size = int(len(Com_featuresets) * 0.1)
  train_set, test_set = Com_featuresets[size:], Com_featuresets[:size]

  # train classifier on the training set
  classifier_com = nltk.NaiveBayesClassifier.train(train_set)
  
  
  print("\n10 Most Informative Features -bigrams with Opinion sentiment ",classifier_com.show_most_informative_features(20))
  
  Accuracy_com=nltk.classify.accuracy(classifier_com, test_set)
  print(" \nAccuracy using bigrams with Opinion sentiment ", Accuracy_com,"\n")
  
  #Run the cross-validation on our word feature sets with 10 folds.
  cross_validation(10,Com_featuresets)
  
  #build the reference and test lists from the classifier on the test set
  predictedlist = []
  goldlist = []
  for (features, label) in test_set:
     goldlist.append(label)
     predictedlist.append(classifier_com.classify(features))
     
  print("\n\n Gold List size -", len(goldlist))
  print("\n\n Predicted List size -", len(predictedlist))
  
  #Now we use the NLTK function to define the confusion matrix, and we print it out:

  cm = nltk.ConfusionMatrix(goldlist, predictedlist)
  print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
        
  eval_measures(goldlist, predictedlist)
  

"""
commandline interface takes a directory name with semeval task b training subdirectory 
       for downloaded-tweeti-b-dist.tsv
   and a limit to the number of tweets to use
It then processes the files and trains a tweet sentiment classifier.

"""
#if __name__ == '__main__':
#    if (len(sys.argv) != 3):
#        print ('usage: classifytweets.py <corpus-dir> <limit>')
#        sys.exit(0)
#    processtweets(sys.argv[1], sys.argv[2])

processtweets('C:\Syracuse\FinalProjectData\SemEval2014TweetData\corpus', 8000)