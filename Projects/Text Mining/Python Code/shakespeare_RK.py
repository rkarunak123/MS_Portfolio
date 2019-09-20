# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:11:07 2019

@author: jhaskins
"""

import nltk
import numpy 
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from wordcloud import WordCloud
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import seaborn as sb
from sklearn.model_selection import KFold, cross_val_score 


rootDataPath = "C:/Users/Sunila/Documents/Python Scripts/Text Mining/Data/shakespeare/"

files = os.listdir(rootDataPath)

#Select the plays - 15 
classDF =   pd.DataFrame(data = np.array([  
["Alls Well That Ends Well", "C", 1602], 
["As You Like It", "C", 1599],
#["The Comedy of Errors", "C", 1589],
#["Loves Labours Lost", "C", 1594],
#["Measure for Measure", "C", 1604],
#["The Merchant of Venice", "C", 1596],
#["The Merry Wives of Windsor", "C", 1600],
["A Midsummer Nights Dream", "C", 1595],
["Much Ado about Nothing", "C", 1598],
#["The Taming of the Shrew", "C", 1593],
["The Tempest", "C", 1611],
#["Twelfth Night", "C", 1599],
#["The Two Gentlemen of Verona", "C", 1594],
["The Winters Tale", "C", 1610],
["King Henry IV First", "H", 1597],  
#["King Henry IV Second", "H", 1597], 
#["King Henry V", "H", 1598],
#["King Henry VI", "H", 1590], 
#["King Henry VIII", "H", 1612], 
["King John", "H", 1596], 
#["Pericles Price of Tyre", "H", 1608], 
#["King Richard II", "H", 1595],
#["King Richard III", "H", 1592],
["Antony and Cleopatra", "T", 1606], 
#["Coriolanus", "T", 1607],  
["Cymbeline", "T", 1609],  
["Hamlet", "T", 1600],  
["Julius Caesar", "T", 1599],  
["King Lear", "T", 1605], 
["Macbeth", "T", 1605],  
#["Othello", "T", 1604],  
["Romeo and Juliet", "T", 1594]#,  
#["Timon of Athens", "T", 1607], 
#["Titus Andronicus", "T", 1593],  
#["Troilus and Cressida", "T", 1601],
#["A Lovers Complaint" ,"P", 1609],
#["The Rape of Lucrece","P", 1594],
#["Venus and Adonis","P", 1593] 
]), 
 columns= ['Title','Genre','Year']) 
    
classDF  = classDF.sort_values(by=['Title'])

classDF['File'] = files

scenes = ['I','II','III','IV','V','VI']

playCharactersDF = []
playDF = []
playActDF = []
playActSceneDF = []
playActSceneCharacterDF = []
gender="F"
enterFlag = 0
exitFlag = 0

stopwords = nltk.corpus.stopwords.words('english')
newStopWords=['art','doth','dost','ere','hast','hath','hence','hither','nigh','oft','thither','thee','thou','thine','thy','tis',
'twas','wast','whence','wherefore','whereto','withal','ye','yon','yonder']
stopwords.extend(newStopWords)

#Create the corpus

for index, dfRow in classDF.iterrows():
    filePath = rootDataPath+dfRow['File']
    
    startCaptureFlag = 0 

    
    if dfRow['Genre'] not in ['P']:
        FILE=open(filePath,"r")   
        act = ''
        character = ''
        for row in FILE:
            row = row.replace("'","")
            row = row.replace("DRAMATIS PERSONAE","")
            row=row.lstrip()
            row=row.rstrip()
            if len(row) > 0:
                #print(len(row))
                #print(len(row)-1)
                if row[0] == '[':
                    enterFlag = 1
                if row[-1] == ']':
                    exitFlag = 1
            row = re.sub('\W+', ' ', row)
            row=row.lstrip()
            row=row.rstrip()
            row=row.strip()

            #print(row)
            if row.lower() != dfRow['Title'].lower() and len(row) > 0  and enterFlag == 0 :
                if len(row) > 3:
                    Mylist=row.split(" ")
                    NewList=[]                   
                    

                    if Mylist[0].lower() in ['page','steward','fairy', 'fairies'
                                            ,'forester','gentleman','messenger'
                                            ,'egyptian','guard','attendant'
                                            ,'widow', 'soothsayer','clown','lady'
                                            ,'captain','jupiter', 'mother','frenchman'
                                            ,'ghost','servant','poet']:
                        Mylist[0] = Mylist[0].upper()
                        
                    if (len(Mylist) > 1 and (
                            (Mylist[0].upper() == 'MARK' and Mylist[1].upper() == 'ANTONY') 
                         or (Mylist[0].upper() == 'OCTAVIUS' and Mylist[1].upper() == 'CAESAR') 
                         or (Mylist[0].upper() == 'DOMITIUS' and Mylist[1].upper() == 'ENOBARBUS')
                         or (Mylist[0].upper() == 'DUKE' and Mylist[1].upper() == 'SENIOR')
                         or (Mylist[0].upper() == 'DUKE' and Mylist[1].upper() == 'FREDERICK')
                         or (Mylist[0].upper() == 'LE' and Mylist[1].upper() == 'BEAU')
                         or (Mylist[0].upper() == 'POSTHUMUS' and Mylist[1].upper() == 'LEONATUS')
                         or (Mylist[0].upper() == 'CAIUS' and Mylist[1].upper() == 'LUCIUS')
                         or (Mylist[0].upper() == 'SICILIUS' and Mylist[1].upper() == 'LEONATUS')  
                         or (Mylist[0].upper() == 'KING' and Mylist[1].upper() == 'CLAUDIUS')  
                         or (Mylist[0].upper() == 'LORD' and Mylist[1].upper() == 'POLONIUS')     
                                                  or (Mylist[0].upper() == 'PRINCE' and Mylist[1].upper() == 'FORTINBRAS')     
                                                  or (Mylist[0].upper() == 'QUEEN' and Mylist[1].upper() == 'GERTRUDE')     
                         or (Mylist[0].upper() == 'DECIUS' and Mylist[1].upper() == 'BRUTUS')   
                         or (Mylist[0].upper() == 'METELLUS' and Mylist[1].upper() == 'CIMBER')   
                         or (Mylist[0].upper() == 'KING' and Mylist[1].upper() == 'JOHN') 
                         or (Mylist[0].upper() == 'PRINCE' and Mylist[1].upper() == 'HENRY')
                         or (Mylist[0].upper() == 'KING' and Mylist[1].upper() == 'PHILIP')           
                         or (Mylist[0].upper() == 'CARDINAL' and Mylist[1].upper() == 'PANDULPH')  
                         or (Mylist[0].upper() == 'QUEEN' and Mylist[1].upper() == 'ELINOR')         
                         or (Mylist[0].upper() == 'LADY' and Mylist[1].upper() == 'FAULCONBRIDGE')  
                         or (Mylist[0].upper() in [ 'FIRST','SECOND','THIRD','FOURTH', 'PLAYER', 'FRENCH', 'ENGLISH'] 
                             and Mylist[1].upper() in ['LORD','GENTLEMAN','SOLDIER','MESSENGER'
                                                      ,'SERVANT','GUARD','ATTENDANT','OFFICER'
                                                      ,'GAOLER','LADY','CAPTAIN','SENATOR'
                                                      ,'TRIBUNE', 'BROTHER','PLAYER','KING','QUEEN'
                                                      ,'CLOWN', 'SAILOR','CITIZEN','COMMONER'
                                                      ,'HERALD', 'EXECUTIONER','PRIEST'])
                        or (Mylist[0].upper() == 'A' and Mylist[1].upper() == 'LORD'))):
                        Mylist[0] = (Mylist[0].upper() + ' ' + Mylist[1].upper())
                      
                        del Mylist[1]
                        
      
                        
                    if  (len(Mylist) > 1 and Mylist[0].upper() == 'PHILIP' and Mylist[1].upper() == 'KING' )  : 
                        Mylist[0] = 'KING PHILIP'                        
                        
                        
                    if  (len(Mylist) > 1 and Mylist[0].upper() == 'JAMES' and Mylist[1].upper() == 'GURNEY' )  : 
                        del Mylist[0]

                    if  (len(Mylist) > 1 and Mylist[0].upper() == 'KING' and Mylist[1].upper() == 'HENRY' )  : 
                        Mylist[0] = 'KING HENRY IV'    
                        
                    if (Mylist[-1].upper() == 'POET' and Mylist[-2].upper() == 'THE'):
                        if (Mylist[-3].upper() == 'CINNA'):
                            Mylist[0] = Mylist[-3].upper()  + ' ' + Mylist[-2].upper() + ' ' + Mylist[-1].upper()
                            del Mylist[-1]
                            del Mylist[-1]
                            del Mylist[-1]

                    if (Mylist[-1].upper() == 'PRIEST' and Mylist[-2].upper() == 'FIRST'):
                        Mylist[0] = Mylist[-2].upper() + ' ' + Mylist[-1].upper()                        
                     
                    if len(Mylist) > 1  and (Mylist[0].upper() == 'JAQUES' and Mylist[1].upper() == 'JAQUES'):
                        Mylist[0] = ('JAQUES DE BOYS')
                        del Mylist[1]   
                        del Mylist[1]   
                        del Mylist[1] 
                        
                    if len(Mylist) > 1  and (Mylist[0].upper() == 'JAQUES' and Mylist[1].upper() == 'DE' and Mylist[1].upper() == 'BOYS'): 
                        Mylist[0] = ('JAQUES DE BOYS')
                        del Mylist[1]   
                        del Mylist[1]   
                        del Mylist[1] 
                        
                    if (Mylist[0].upper() == 'SIR' and Mylist[1].upper() == 'OLIVER'):
                         if  Mylist[2].upper() == 'MARTEXT':
                             Mylist[0] = (Mylist[0] + ' ' + Mylist[1] + ' ' + Mylist[2])
                             del Mylist[1] 
                             del Mylist[1]  
                    
                    if (Mylist[0].lower() == 'english' and Mylist[1].lower() == 'ambassadors'):
                        Mylist[0] = 'FIRST AMBASSADOR'
                     
                         
                    if (Mylist[0].upper() in [ 'JULIUS','MARCUS','YOUNG'] and Mylist[1].upper() in ['CAESAR','BRUTUS','CATO']):   
                         Mylist[0] = (Mylist[1].upper() ) 
                         del Mylist[1]
                         
                    if (Mylist[0].upper() in [ 'OCTAVIUS','POPILIUS'] and Mylist[1].upper() in ['CAESAR','LENA']):   
                        Mylist[0] = (Mylist[0].upper() )                          
                        del Mylist[1]
                    
                    if (Mylist[0].upper() == 'MARCUS' and Mylist[1].upper() == 'ANTONIUS'):   
                        Mylist[0] = 'ANTONY'                        
                        del Mylist[1]
                    
                    if startCaptureFlag == 0:
                        if row.isupper() and len(Mylist) > 1:                             
                             temp = []
                             for i in Mylist :
                                 temp.append(i.lower())                             
                             Mylist = temp
                             if Mylist[0].upper() in ['SEXTUS','TITUS','MENENIUS']:
                                 Mylist[0] = Mylist[-1].upper()
                             else:
                                 Mylist[0] = Mylist[0].upper()
                                 
                    if Mylist[0].upper() == 'ROUSILLON':
                        Mylist[0] = Mylist[0].lower()
        
                    if Mylist[-1].lower() in ([ 'widow', 'soothsayer','clown','page','captain','gentlemen','lord','poet']):
                        Mylist[-1]  =  Mylist[-1].upper() 

                    if Mylist[-1].upper() == 'COUNTESS':
                        Mylist[-1]  =  Mylist[-1].lower()                         
                    #   
                    if Mylist[0].lower() == 'scene' and Mylist[1].upper() in scenes:
                        scene = Mylist[0] + " " + Mylist[1]
                   
                    if Mylist[0].lower() != 'scene':
                        if Mylist[0].lower() == 'act':
                            act = row.upper()
                            if startCaptureFlag == 0:
                                startCaptureFlag =  1
                        if Mylist[0].lower() != 'act':    
                            for word in Mylist:
                                newList = []
                                #print(word.isupper() and startCaptureFlag == 0)
                                if word.isupper() and startCaptureFlag == 0 and len(word) > 1 :
#                                and not in playCharactersDF[playCharactersDF[1] == dfRow['Title']][2] :
                                    dupflag = 0
                                    if len(playCharactersDF) > 0:
                                        for i in playCharactersDF:
                                           if (i[0] == dfRow['Title'] and i[1] == word 
                                               or dfRow['Title'] == 'Hamlet' and word in [ 'KING', 'POLONIUS', 'CLAUDIUS','FORTINBRAS', 'GERTRUDE' ] 
                                               or dfRow['Title'] == 'King John' and word in [ 'DE', 'BURGH', 'FAULCONBRIDGE','LYMOGES','PHILIP' ] 
                                               ):
                                               dupflag = 1
                                    if dupflag == 0:    
                                        playCharactersDF.append([dfRow['Title'],word]) 
                     
                                if enterFlag == 0 and startCaptureFlag ==  1:
                                    if word.isupper() and len(word) > 3:
                                        character = word
                                    else:     
                                    #    print(word)
                                        NewList.append(word)
                            Text=" ".join(NewList)
                            Text=Text.replace("\\n","")
                            Text=Text.strip("\\n")
                            Text= Text.lstrip()
                            Text= Text.rstrip()
                            Text= Text.strip()
                            
                            if startCaptureFlag == 1 and len(Text) > 0:
                                playDF.append([dfRow['Title'],Text])  
                                playActDF.append([dfRow['Title'],act,Text]) 
                                playActSceneDF.append([dfRow['Title'],act,scene,Text]) 
                                playActSceneCharacterDF.append([dfRow['Title'],act,scene,character,Text]) 
                               
                             
                                
            if enterFlag == 1 and exitFlag == 1:
                enterFlag = 0
                exitFlag = 0
           
        print(FILE.readlines())    
        FILE.close()
playDF=pd.DataFrame(playActSceneCharacterDF,columns=["Title","Act","Scene","Character","Text"])
myDF1= playDF[playDF['Title'].isin(['Antony and Cleopatra','A Midsummer Nights Dream','King Henry IV First','Romeo and Juliet','Hamlet','Much Ado about Nothing','Julius Caesar','Alls Well That Ends Well','The Tempest','The Winters Tale','King Lear'])]
myDF= myDF1[myDF1['Character'].isin(["MARK ANTONY","OCTAVIUS CAESAR","KING HENRY IV","ROMEO","HAMLET","KING CLAUDIUS","OSWALD","HOTSPUR" ,"CLEOPATRA",
            "OCTAVIA","HERMIA","HELENA","TITANIA","JULIET","LADY CAPULET","NURSE","QUEEN GERTRUDE","OPHELIA","URSULA","BEATRICE","MARGARET","CORDELIA",
            "REGAN","HIPPOLYTA","CALPURNIA","PORTIA","COUNTESS","MARIANA","DIANA","ARIEL","MIRANDA","HERMIONE","CORDELIA","GONERIL","VIOLENTA","COUNTESS"])]

chr= myDF.iloc[:,3].unique().tolist()
print(chr)

#Add gender labels
CharListM=["MARK ANTONY","OCTAVIUS CAESAR","KING HENRY IV","ROMEO","HAMLET","KING CLAUDIUS","OSWALD","HOTSPUR"]
CharListF=["CLEOPATRA","OCTAVIA","HERMIA","HELENA","TITANIA","JULIET","LADY CAPULET","NURSE","QUEEN GERTRUDE","OPHELIA","URSULA","BEATRICE","MARGARET",
           "CORDELIA","REGAN","HIPPOLYTA","CALPURNIA","PORTIA","COUNTESS","MARIANA","DIANA","ARIEL","MIRANDA","HERMIONE","CORDELIA","GONERIL","VIOLENTA","COUNTESS"]

def findgender(character):
        if character in CharListM:
                return "M"
        elif character in CharListF:
                   return "F"
        else:
                    return "N"

GenderList=[]
for index, row in myDF.iterrows():
    if str(findgender(myDF.loc[index,"Character"])) != "N":
           GenderList.append(str(findgender(myDF.loc[index,"Character"])))

myDF["Gender"]=GenderList

#Review the corpus
##WordCloud
Dialogues = ' '.join(myDF['Text'])
#print(Dialogues)

wordcloud = WordCloud(width = 400, height = 400, 
                background_color ='white', 
                min_font_size = 2).generate(Dialogues) 
  
                        
plt.figure(figsize = (5, 5), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()

#Preprocessing
#---------------------------------------------------------------------------------------------
#Convert to all words from uppercase to lowercase to avoid double counting
print('\n\n Convert to all words from uppercase to lowercase to avoid double counting\n')
myDF['Text']  = myDF['Text'] .apply(lambda x: x.lower()) 

#Remove punctuations
print('\n Remove Punctuations as it does not add any extra information, will help reduce the size\n')
myDF['Text'] = myDF['Text'].str.replace('[^\w\s]',' ')

#Remove single char which result after punctuations are removed
print("\nRemove single char which result after punctuations are removed\n")
myDF['Text'] = myDF['Text'].apply(lambda x: re.sub(r"\s+[a-zA-Z]\s+", " ", x))

#Remove stop words
print("\nRemove stop words\n" )
myDF['Text']= myDF['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))

#Apply Porters stemmer
#ps = PorterStemmer()   ## method from nltk
#myDF['Text'] =myDF['Text'].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))

#Apply Lemmatization

lemmatizer = WordNetLemmatizer()
myDF['Text'] = myDF['Text'].apply(lambda x: " ".join(lemmatizer.lemmatize(x)  for x in x.split()))

## Looking at word frequency

GenderListM=[]
GenderListF=[]

for index, row in myDF.iterrows():
    if (myDF.loc[index,"Gender"] == "M") :
        GenderListM.append(myDF.loc[index,"Text"])
    else:
        GenderListF.append(myDF.loc[index,"Text"])
        
        
Dialogues_M = ''.join(str(GenderListM))
Dialogues_F = ''.join(str(GenderListF))

Dialogues_M=Dialogues_M.replace(',','')
Dialogues_M=Dialogues_M.replace("'","")
Dialogues_F=Dialogues_F.replace(',','')
Dialogues_F=Dialogues_F.replace("'","")

print("\nFrequency Distribution Plot\n")

#Tokenization
print('\n Tokenization\n')
        
Dialogues_M_tokenized= nltk.word_tokenize(Dialogues_M)
Dialogues_F_tokenized= nltk.word_tokenize(Dialogues_F)

print("Word Frequency Plot for Male Characters\n")
fdistM = FreqDist(Dialogues_M_tokenized)
fdistM.plot(20,cumulative=False)

print("Word Frequency Plot for Female Characters\n")
fdistF = FreqDist(Dialogues_F_tokenized)
fdistF.plot(20,cumulative=False)

import matplotlib.pyplot as plt

Distwords_M = pd.DataFrame(fdistM.most_common(20))

label= Distwords_M.iloc[:,0]
val = Distwords_M.iloc[:,1]

index = np.arange(len(label))
plt.bar(index, val)
plt.xlabel('Words ', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.xticks(index, label, fontsize=10, rotation=50)
plt.title('Top 20 words for Male Actors')
plt.show()


Distwords_F = pd.DataFrame(fdistF.most_common(20))

label= Distwords_F.iloc[:,0]
val = Distwords_F.iloc[:,1]

index = np.arange(len(label))
plt.bar(index, val)
plt.xlabel('Words ', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.xticks(index, label, fontsize=10, rotation=50)
plt.title('Top 20 words for Female Actors')
plt.show()

#----------------------------------------------------------------------------------
#Common functions
#-----------------------------------------------------------------------------------
Models_Result=[]

def plotConfMatrix(title,cm):
    sb.set(font_scale=1.4)
    cm_df = pd.DataFrame(cm,columns=["M","F"],index=["M","F"])
  
    sb.heatmap(cm_df, annot=True,annot_kws={"size": 12},fmt='d')
    plt.xlabel('Actual ')
    plt.ylabel('Predicted ') 
    plt.title('Confusion Matrix for ' + title)
    plt.show()

def Model_Evaluation(model, modelname,X_train_vec,X_test_vec):
    y_pred = model.fit(X_train_vec, y_train).predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted')
    recall = recall_score(y_test, y_pred,average='weighted')
    F1_score = f1_score(y_test, y_pred,average='weighted')
    print("\n ", modelname, " Accuracy is ", round(accuracy,2))

    labels=["M","F"]
    cm=confusion_matrix(y_test, y_pred,  labels = labels)
    print("\n ", modelname, " - Confusion Matrix")
    print(cm)

    plotConfMatrix("Confusion Matrix for Gender Analysis",cm)

    Models_Result.append({"Model Name   ": modelname, "Accuracy":round(accuracy,2), })
    

    
#--------------------------------------------------------------------------------------
#Prepare the Train and Test datasets 
#---------------------------------------------------------------------------------------
#Split the training data so as to use 70% for train and 30% for test
X=myDF['Text']
y=myDF['Gender']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



#------------------------------------------------------------------------------------------
#Vectorization Models
#-------------------------------------------------------------------------------------------


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
    

#Model1 - #  n-gram count vectorizer
ngram_count_vectorizer = CountVectorizer(input='content',
                                           analyzer = 'word',
                                           lowercase = True,
                                           stop_words=stopwords,
                                           tokenizer= LemmaTokenizer(),
                                           ngram_range=(1, 4),
                                           binary=False
                              
                            
                                    )
#Model2 - n-gram boolean vectorizer
ngram_bool_vectorizer = CountVectorizer(input='content',
                                           analyzer = 'word',
                                           lowercase = True,
                                           stop_words=stopwords,
                                           ngram_range=(1, 4),
                                           binary=True,
                                           #tokenizer= LemmaTokenizer()
                                           
                                    )
#model3 - tfid vectorizer
unigram_tfidf_vectorizer=TfidfVectorizer( input='content',
                        analyzer = 'word',
                        lowercase = True,
                        stop_words=stopwords,
                        use_idf=True,
                        ngram_range=(1, 4),
                        binary=False,
                        min_df=.0001
            
                        )

#---------------------------------------------------------------------------------------------------
# MNB and SVM algorithm results
#-----------------------------------------------------------------------------------------------
# initialize the MNB model
nb_clf= MultinomialNB()

# initialize the LinearSVC model
svm_clf = LinearSVC(C=1)

    
#Model 1
# fit vocabulary in training documents and transform the training documents into vectors
X_train_vec = ngram_count_vectorizer.fit_transform(X_train)

# fit vocabulary in test documents and transform the test documents into vectors
X_test_vec = ngram_count_vectorizer.transform(X_test)

# use the training data to train the MNB model
nb_clf.fit(X_train_vec,y_train)
 
# use the training data to train the SVM model
svm_clf.fit(X_train_vec,y_train)   

Model_Evaluation(nb_clf,"MNB - ngram count vectorizer",X_train_vec,X_test_vec)
Model_Evaluation(svm_clf,"SVM - ngram count vectorizer",X_train_vec,X_test_vec)


MNB_Result = pd.DataFrame(Models_Result)

print("\nAccuracy Result Summary using ngram count vectorizer ")
print(tabulate(MNB_Result, headers='keys', tablefmt='psql'))

#Model 2
Models_Result=[]
# fit vocabulary in training documents and transform the training documents into vectors
X_train_vec = ngram_bool_vectorizer.fit_transform(X_train)

# fit vocabulary in test documents and transform the test documents into vectors
X_test_vec = ngram_bool_vectorizer.transform(X_test)

# use the training data to train the MNB model
nb_clf.fit(X_train_vec,y_train)
 
# use the training data to train the SVM model
svm_clf.fit(X_train_vec,y_train)   

Model_Evaluation(nb_clf,"MNB - ngram boolean vectorizer",X_train_vec,X_test_vec)
Model_Evaluation(svm_clf,"SVM - ngram boolean vectorizer",X_train_vec,X_test_vec)


MNB_Result = pd.DataFrame(Models_Result)

print("\nAccuracy Result Summary using ngram boolean vectorizer ")
print(tabulate(MNB_Result, headers='keys', tablefmt='psql'))

#Model 3
Models_Result=[]
# fit vocabulary in training documents and transform the training documents into vectors
X_train_vec = unigram_tfidf_vectorizer.fit_transform(X_train)

# fit vocabulary in test documents and transform the test documents into vectors
X_test_vec = unigram_tfidf_vectorizer.transform(X_test)

# use the training data to train the MNB model
nb_clf.fit(X_train_vec,y_train)
 
# use the training data to train the SVM model
svm_clf.fit(X_train_vec,y_train)   

Model_Evaluation(nb_clf,"MNB - unigram_tfidf_vectorizer",X_train_vec,X_test_vec)
Model_Evaluation(svm_clf,"SVM - unigram_tfidf_vectorizer",X_train_vec,X_test_vec)


MNB_Result = pd.DataFrame(Models_Result)

print("\nAccuracy Result Summary using unigram tfidf vectorizer ")
print(tabulate(MNB_Result, headers='keys', tablefmt='psql'))




#10 most indicative words for each model
#-------------------------------------------------------------------------------------

stopwords_nonames = nltk.corpus.stopwords.words('english')
newStopWords=['art','doth','dost','ere','hast','hath','hence','hither','nigh','oft','thither','thee','thou','thine','thy','tis',
'twas','wast','whence','wherefore','whereto','withal','ye','yon','yonder',"mark antony","octavius caesar","king henry iv","romeo","hamlet","king claudius","oswald","hotspur" ,"cleopatra",
            "octavia","hermia","helena","titania","juliet","lady capulet","nurse","queen gertrude","ophelia","ursula","beatrice","margaret","cordelia",
            "regan","hippolyta","calpurnia","portia","countess","mariana","diana","ariel","miranda","hermione","cordelia","goneril","violenta","countess","benedick","lysander","demetrius","charmian","charmia","gertrude","mortimer","horatio","kate","brutus","richard","oswald","philippa","pompey","france","tybalts","eros","si","alack"]
stopwords_nonames.extend(newStopWords)

#uni-gram count vectorizer
unigram_Tfidf_vectorizer = TfidfVectorizer( input='content',
                        analyzer = 'word',
                        lowercase = True,
                        stop_words=stopwords_nonames,
                        use_idf=True,
                        ngram_range=(1, 1),
                        binary=False,
                        min_df=.001
            
                        )

# fit vocabulary in training documents and transform the training documents into vectors
X_train_vec = unigram_Tfidf_vectorizer .fit_transform(X_train)

# fit vocabulary in test documents and transform the test documents into vectors
X_test_vec = unigram_Tfidf_vectorizer .transform(X_test)


# use the training data to train the MNB model
nb_clf.fit(X_train_vec,y_train)
 
# use the training data to train the SVM model
svm_clf.fit(X_train_vec,y_train)  

#Based on coeff values per class
def print_top10_indicativewords(vectorizer, clf, class_labels):
    feature_names = vectorizer.get_feature_names()
    class_label=clf.classes_
    top10= np.argsort(clf.coef_[0])[-10:]
    #print(top10)
    print("%s: %s" % (class_label[1], " ".join(feature_names[j] for j in top10)))
    #print("\n")
    top10= np.argsort(clf.coef_[0])[0:10]
    #print(top10)
    print("%s: %s" % (class_label[0], " ".join(feature_names[j] for j in top10)))
    print("\n")
        

   
print("\n Top 10 indicative words using MNB\n")    
print_top10_indicativewords(unigram_Tfidf_vectorizer, nb_clf, myDF['Gender'])
print(" Top 10 indicative words using SVM")    
print_top10_indicativewords(unigram_Tfidf_vectorizer ,svm_clf, myDF['Gender'])    

#---------------------------------------------------------------------------------------
#Using Cross Validation
#-------------------------------------------------------------------
#
#kf=KFold(n_splits=10, shuffle=False)
#
## fit vocabulary in training documents and transform the training documents into vectors
#X = ngram_count_vectorizer.fit_transform(X_train)
#
#colnames = ngram_count_vectorizer.get_feature_names()
#rownames = X.toarray()
##print(colnames)
##print(rownames)
#
#myPlayDF=pd.DataFrame(rownames,columns=colnames)
#
#
#myPlayDF_NoLables = myPlayDF.drop(["Gender"], axis=1)
##Lie dectection
#results_lie = cross_val_score(nb_clf, myPlayDF_NoLables, myDF['Gender'], cv=kf, scoring='accuracy')
#print("\n\nThe accuracy using mnb cross validation  method for Lie detection is" ,round(results_lie.mean(),2))
