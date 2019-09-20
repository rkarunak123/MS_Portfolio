# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:11:07 2019

@author: jhaskins
"""

import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.stop.old_english.stops import STOPS_LIST
from nltk.corpus import stopwords
import os
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import ward, dendrogram

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems



rootDataPath = "C:\\Users\\jhaskins\\Desktop\\Text Mining\\DATA\\shakespeare\\"

files = os.listdir(rootDataPath)


STOPWORDS = "a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your"

STOPWORDS = STOPWORDS.split(",")   


lemmer=WordNetLemmatizer()



classDF =   pd.DataFrame(data = np.array([  
["A Midsummer Nights Dream", "C", 1595],
["Alls Well That Ends Well", "C", 1602], 
["Antony and Cleopatra", "T", 1606],
["As You Like It", "C", 1599],
["Cymbeline", "T", 1609], 
["Hamlet", "T", 1600],  
["Julius Caesar", "T", 1599],  
["King Henry IV First", "H", 1597],  
["King John", "H", 1596], 
["King Lear", "T", 1605], 
["Macbeth", "T", 1605],
["Much Ado about Nothing", "C", 1598],
["Romeo and Juliet", "T", 1594],
["The Tempest", "C", 1611],
["The Winters Tale", "C", 1610]#,
#["The Comedy of Errors", "C", 1589],
#["Loves Labours Lost", "C", 1594],
#["Measure for Measure", "C", 1604],
#["The Merchant of Venice", "C", 1596],
#["The Merry Wives of Windsor", "C", 1600],


#["The Taming of the Shrew", "C", 1593],
#["Twelfth Night", "C", 1599],
#["The Two Gentlemen of Verona", "C", 1594],
#["King Henry IV Second", "H", 1597], 
#["King Henry V", "H", 1598],
#["King Henry VI", "H", 1590], 
#["King Henry VIII", "H", 1612], 

#["Pericles Price of Tyre", "H", 1608], 
#["King Richard II", "H", 1595],
#["King Richard III", "H", 1592], 
#["Coriolanus", "T", 1607],   
#["Othello", "T", 1604],  
 
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

enterFlag = 0
exitFlag = 0




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
            if (row.lower() != dfRow['Title'].lower() or dfRow['Title'] in ['King John', 'Macbeth']  ) and len(row) > 0  and enterFlag == 0 :
                if len(row) > 3:
                    Mylist=row.split(" ")
                    NewList=[]                   
                    

                    if Mylist[0].lower() in ['page','steward','fairy', 'fairies'
                                            ,'forester','gentleman','messenger'
                                            ,'egyptian','guard','attendant'
                                            ,'widow', 'soothsayer','clown','lady'
                                            ,'captain','jupiter', 'mother','frenchman'
                                            ,'ghost','servant','poet','hostess'
                                            ,'sheriff','vintner','chamberlain','servant'
                                            ,'ostler','herald','knight','fool','doctor'
                                            ,'son','soldier','porter','gentlewoman'
                                            ,'sexton','watchman','master','boatswain'
                                            ,'mariners','shepherd','gaoler','mariner'
                                            ,'time']:
                        Mylist[0] = Mylist[0].upper()
                        
                    if (len(Mylist) > 1 and (
                            (Mylist[0].upper() == 'MARK' and Mylist[1].upper() == 'ANTONY') 
                         or (Mylist[0].upper() == 'OCTAVIUS' and Mylist[1].upper() == 'CAESAR' and dfRow['Title'] != 'Julius Caesar') 
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
                         or (Mylist[0].upper() == 'OWEN' and Mylist[1].upper() == 'GLENDOWER')  
                         or (Mylist[0].upper() == 'SIR' and Mylist[1].upper() == 'MICHAEL')  
                         or (Mylist[0].upper() == 'QUEEN' and Mylist[1].upper() == 'ELINOR')       
                         or (Mylist[0].upper() == 'LADY' and Mylist[1].upper() in ['FAULCONBRIDGE','PERCY','MORTIMER','MACBETH','MACDUFF'])  
                         or (Mylist[0].upper() == 'KING' and Mylist[1].upper() == 'LEAR')  
                         or (Mylist[0].upper() == 'OLD' and Mylist[1].upper() == 'MAN')  
                         or (Mylist[0].upper() == 'YOUNG' and Mylist[1].upper() == 'SIWARD') 
                         or (Mylist[0].upper() == 'DON' and Mylist[1].upper() in ['PEDRO','JOHN']) 
                         or (Mylist[0].upper() == 'FRIAR' and Mylist[1].upper() in ['FRANCIS','LAURENCE','JOHN']) 
                         or (Mylist[0].upper() in [ 'FIRST','SECOND','THIRD','FOURTH', 'PLAYER', 'FRENCH', 'ENGLISH'] 
                             and Mylist[1].upper() in ['LORD','GENTLEMAN','SOLDIER','MESSENGER'
                                                      ,'SERVANT','GUARD','ATTENDANT','OFFICER'
                                                      ,'GAOLER','LADY','CAPTAIN','SENATOR'
                                                      ,'TRIBUNE', 'BROTHER','PLAYER','KING','QUEEN'
                                                      ,'CLOWN', 'SAILOR','CITIZEN','COMMONER'
                                                      ,'HERALD', 'EXECUTIONER','PRIEST', 'CARRIER'
                                                      ,'TRAVELLER','WITCH','APPARITION','MURDERER'
                                                      ,'WATCHMAN','CAPULET','MUSICIAN','PAGE'
                                                      ])
                        or (Mylist[0].upper() == 'A' and Mylist[1].upper() == 'LORD'))):
                        Mylist[0] = (Mylist[0].upper() + ' ' + Mylist[1].upper())
                        del Mylist[1]
                        
      
                        
                    if  (len(Mylist) > 1 and Mylist[0].upper() == 'PHILIP' and Mylist[1].upper() == 'KING' )  : 
                        Mylist[0] = 'KING PHILIP'        
                        
                    if  (len(Mylist) > 1 and Mylist[0].upper() == 'ARCHIBALD'  and dfRow['Title'] == 'King Henry IV First')  : 
                        Mylist[0] = 'DOUGLAS'                            
                        
                    if  (len(Mylist) > 1 and Mylist[0].upper() == 'CLAUDIUS' and Mylist[1].upper() == 'KING' )  : 
                        Mylist[0] = 'KING CLAUDIUS'   
                    
                    if  (len(Mylist) > 1 and Mylist[0].upper() == 'POLONIUS' and Mylist[1].upper() == 'LORD' )  : 
                        Mylist[0] = 'LORD POLONIUS'   
                        
                    if  (len(Mylist) > 1 and Mylist[0].upper() == 'FORTINBRAS' and Mylist[1].upper() == 'PRINCE' )  : 
                        Mylist[0] = 'PRINCE FORTINBRAS'   
                            
                        
                    if  (len(Mylist) > 1 and Mylist[0].upper() == 'JAMES' and Mylist[1].upper() == 'GURNEY' )  : 
                        del Mylist[0]
                        
                    if  (len(Mylist) > 0 and Mylist[0].upper() == 'HENRY')  : 
                        Mylist[0] = 'PRINCE HENRY'

                    if  (len(Mylist) > 1 and Mylist[0].upper() == 'KING' and Mylist[1].upper() == 'HENRY' )  : 
                        Mylist[0] = 'KING HENRY IV'    
                        
                    if  (len(Mylist) > 2 and Mylist[0].upper() == 'KING' and Mylist[1].upper() == 'HENRY')  : 
                        if  Mylist[1].upper() == 'IV':
                            Mylist[0] =    Mylist[0]  + ' ' + Mylist[1]  + ' ' + Mylist[2] 
                            del Mylist[1]
                            del Mylist[2] 
                        else:
                            Mylist[0] =   'KING HENRY IV'    
                            del Mylist[1]
                            
                    if (len(Mylist) > 2 and Mylist[-1].upper() == 'POET' and Mylist[-2].upper() == 'THE'):
                        if (Mylist[-3].upper() == 'CINNA'):
                            Mylist[0] = Mylist[-3].upper()  + ' ' + Mylist[-2].upper() + ' ' + Mylist[-1].upper()
                            del Mylist[-1]
                            del Mylist[-1]
                            del Mylist[-1]
                            
                    if (len(Mylist) > 1 and Mylist[-1].upper() == 'CAPULET' and Mylist[-2].upper() == 'SECOND'):
                        Mylist[0] = Mylist[-2].upper() + ' ' + Mylist[-1].upper() 
                        del Mylist[-1]
                        del Mylist[-1]
                    
                    if (Mylist[-1].upper() == 'MAN' and Mylist[-2].upper() == 'OLD'):
                        Mylist[0] = Mylist[-2].upper() + ' ' + Mylist[-1].upper() 
                        del Mylist[-1]
                        del Mylist[-1]
                            
                    if (len(Mylist) > 2 and Mylist[0].upper() == 'CINNA' and Mylist[1].upper() == 'THE'):
                        if (Mylist[2].upper() == 'POET'):
                            Mylist[0] = Mylist[0].upper()  + ' ' + Mylist[1].upper() + ' ' + Mylist[2].upper()
                            del Mylist[1]
                            del Mylist[2]
                      

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
                        
                    if (len(Mylist) > 1 and Mylist[0].upper() == 'SIR' and Mylist[1].upper() in ['OLIVER','WALTER']):
                         if  Mylist[2].upper() in ['MARTEXT','BLUNT']:
                             Mylist[0] = (Mylist[0] + ' ' + Mylist[1] + ' ' + Mylist[2])
                             del Mylist[1] 
                             del Mylist[1]  
                    
                    if (Mylist[0].lower() == 'english' and Mylist[1].lower() == 'ambassadors'):
                        Mylist[0] = 'FIRST AMBASSADOR'
                     
                         
                    if (Mylist[0].upper() in [ 'JULIUS','MARCUS','YOUNG'] and Mylist[1].upper() in ['CAESAR','BRUTUS','CATO']):   
                         Mylist[0] = (Mylist[1].upper() ) 
                         del Mylist[1]
                         
                    if (Mylist[0].upper() in [ 'OCTAVIUS','POPILIUS','EDMUND'] and Mylist[1].upper() in ['CAESAR','LENA','MORTIMER']):   
                        Mylist[0] = (Mylist[0].upper() )                          
                        del Mylist[1]
                    
                    if (Mylist[0].upper() == 'MARCUS' and Mylist[1].upper() == 'ANTONIUS'):   
                        Mylist[0] = 'ANTONY'                        
                        del Mylist[1]
                        
                    if (Mylist[0].upper() == 'LEAR'):   
                        Mylist[0] = 'KING LEAR'                        
                        
                        
                    if ( Mylist[0].upper() == 'THOMAS' and Mylist[1].upper() == 'PERCY'):   
                        Mylist[0] = 'EARL OF WORCESTER'                        
                        del Mylist[1]     
                    
                    if (len(Mylist) > 2 and Mylist[0].upper() == 'KING'  and Mylist[1].upper() == 'OF' and Mylist[2].upper() == 'FRANCE' and dfRow['Title'] != 'Alls Well That Ends Well' ):
                        Mylist[0] = (Mylist[0] + ' ' + Mylist[1] + ' ' + Mylist[2])
                        del Mylist[1] 
                        del Mylist[1] 
                    
                    if (len(Mylist) > 2 and Mylist[0].upper() in ['EARL','ARCHBISHOP']  and Mylist[1].upper() == 'OF' and Mylist[2].upper() in ['FRANCE','WORCESTER','YORK'] ):
                        Mylist[0] = (Mylist[0] + ' ' + Mylist[1] + ' ' + Mylist[2])
                        del Mylist[1] 
                        del Mylist[1]  
                        
                        
                    if (len(Mylist) > 1 and Mylist[0].upper() in ['EARL','DUKE'] and Mylist[1].upper() == 'OF'):
                         if  Mylist[2].upper() in ['BURGUNDY','CORNWALL','ALBANY','KENT','GLOUCESTER']:
                             Mylist[0] =  Mylist[2]
                             del Mylist[1] 
                             del Mylist[1]                          
                        
                    if (len(Mylist) > 2 and  Mylist[0].upper() == 'HENRY' and Mylist[1].upper() == 'PERCY'): 
                        if(Mylist[1].upper() == 'EARL'):
                            Mylist[0] = 'NORTHUMBERLAND'                        
                            del Mylist[1]  
                        if(Mylist[1].upper() == 'SURNAME'):
                            Mylist[0] = 'HOTSPUR'                        
                            del Mylist[1]  
                            
                    if (len(Mylist) > 1 and  Mylist[0].upper() == 'RICHARD' and Mylist[1].upper() == 'SCROOP'):                         
                        Mylist[0] = 'ARCHBISHOP OF YORK'
                        del Mylist[1]
                       
                    if (len(Mylist) > 1 and Mylist[0].upper() == 'SIR' and Mylist[1].upper() in ['RICHARD','JOHN']):
                         if  Mylist[2].upper() in ['FALSTAFF','VERNON']:
                             Mylist[0] = Mylist[2].upper()
                             del Mylist[1] 
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
        
                    if Mylist[-1].lower() in ([ 'widow', 'soothsayer','clown','page','captain'
                                                ,'gentlemen','lord','poet','hostess','herald'
                                                , 'gentlewoman', 'sexton','shepherd','mariner'
                                                ,'gaoler']):
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
                                               or dfRow['Title'] == 'Alls Well That Ends Well' and word in [ 'LORD']
                                               or dfRow['Title'] == 'Hamlet' and word in [ 'KING', 'POLONIUS', 'CLAUDIUS','FORTINBRAS', 'GERTRUDE' ]
                                               or dfRow['Title'] == 'King Henry IV First' and word in [ 'HENRY' ,'PRINCE','JOHN','MISTRESS','QUICKLY'
                                                                                                       ,'EARL','OF','WORCESTER','KING','IV', 'ARCHBISHOP'
                                                                                                       , 'YORK', 'PERCY' , 'EDMUND' ]
                                               or dfRow['Title'] == 'King John' and word in [ 'DE', 'BURGH', 'FAULCONBRIDGE','LYMOGES','PHILIP' ,'KING','LORD'] 
                                               or dfRow['Title'] == 'Romeo and Juliet' and word in [ 'ESCALUS' ] 
                                               or dfRow['Title'] == 'As You Like It' and word in [ 'CLOWN','PAGE' ] 
                                               or dfRow['Title'] == 'King Lear' and word in [ 'KING','LEAR' ]
                                              
                                               
                                               ):
                                               dupflag = 1
                                    if dupflag == 0:    
                                        playCharactersDF.append([dfRow['Title'],word]) 
                     
                                if enterFlag == 0 and startCaptureFlag ==  1:
                                    if word.isupper() and len(word) > 3:
                                        character = word
                                    else:     
                                    #    print(word)
                                        NewList.append(lemmer.lemmatize(word))
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
#        
        
print('DONE')



finalPlay = []
title= ""
words = ""
append = 0


for play in playDF:
    if title != play[0] and append == 0:  
        title = play[0]
        words = ""
        append = 1
        
    if title != play[0] and append == 1:
        finalPlay.append([title,words])
        title = play[0]
        words = ""
    else:
        words = words + ' ' + play[1]
    

finalPlay.append([title,words])

FinalPlayDF = pd.DataFrame(np.array(finalPlay))
FinalPlayDF.columns = ["Play","Dialog"]

#stopwords = set(stopwords.words('english')) # + STOPS_LIST 

vectorizer = CountVectorizer(input='content'
                           ,stop_words  = STOPWORDS
                           , max_df=0.8
                           , max_features=200000
                           , min_df=0.2                   
                           , tokenizer=tokenize_and_stem, ngram_range=(1,3)
                           )

dtm = vectorizer.fit_transform(FinalPlayDF.Dialog)

vocab = vectorizer.get_feature_names()  # change to a list
dtm = dtm.toarray()  # convert to a regular array
PDF=pd.DataFrame(data=dtm)
PDF.columns = vocab



print(PDF)
print('DOne')

dist = euclidean_distances(dtm)
print("Euclidean Dist:\n", np.round(dist,0))  #The dist between Emma and Pride is 3856

#Measure of distance that takes into account the
#length of the document: called cosine similarity
cosdist = 1 - cosine_similarity(dtm)
print("Cos Sim: ", np.round(cosdist,3))  #cos dist should be .02

## Visualizing Distances
##An option for visualizing distances is to assign a point in a plane
## to each text such that the distance between points is proportional 
## to the pairwise euclidean or cosine distances.
## This type of visualization is called multidimensional scaling (MDS) 
## in scikit-learn (and R  -  mdscale).
### The following Figure is odd and I am working on it. Feel free to play with it too. 
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
## "precomputed" means we will give the dist (as cosine sim)
pos = mds.fit_transform(cosdist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
names=FinalPlayDF.Play
fig1 = plt.figure()
for x, y, name in zip(xs, ys, names):
    plt.scatter(x, y)
    plt.text(x, y, name)
plt.show()
#-------------



GenreDF = pd.concat([classDF.Genre, FinalPlayDF], axis=1)



dtm = vectorizer.fit_transform(GenreDF.Dialog)

vocab = vectorizer.get_feature_names()  # change to a list
dtm = dtm.toarray()  # convert to a regular array
PDF=pd.DataFrame(data=dtm)
PDF.columns = vocab



print(PDF)
print('DOne')





dist = euclidean_distances(dtm)
print("Euclidean Dist:\n", np.round(dist,0))  #The dist between Emma and Pride is 3856

#Measure of distance that takes into account the
#length of the document: called cosine similarity
cosdist = 1 - cosine_similarity(dtm)
print("Cos Sim: ", np.round(cosdist,3))  #cos dist should be .02

## Visualizing Distances
##An option for visualizing distances is to assign a point in a plane
## to each text such that the distance between points is proportional 
## to the pairwise euclidean or cosine distances.
## This type of visualization is called multidimensional scaling (MDS) 
## in scikit-learn (and R  -  mdscale).
### The following Figure is odd and I am working on it. Feel free to play with it too. 
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
## "precomputed" means we will give the dist (as cosine sim)
pos = mds.fit_transform(cosdist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]



#set up colors per clusters using a dict
cluster_colors = {'C': '#1b9e77', 'T': '#d95f02', 'H': '#7570b3'}

#set up cluster names using a dict
cluster_names = {'C': 'Comedy', 
                 'T': 'Tragedy', 
                 'H': 'Historical'}


#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=GenreDF.Genre, title=GenreDF.Play)) 

groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='on',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='on',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='on')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

    
    
plt.show() #show the plot

#names=GenreDF.Genre
#fig1 = plt.figure()
#for x, y, name in zip(xs, ys, names):
#    plt.scatter(x, y)
#    plt.text(x, y, name)
#plt.show()



#set up colors per clusters using a dict
cluster_colors = {'C': '#1b9e77', 'T': '#d95f02', 'H': '#7570b3'}

#set up cluster names using a dict
cluster_names = {'C': 'Comedy', 
                 'T': 'Tragedy', 
                 'H': 'Historical'}



#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
                  
#set up cluster names using a dict
cluster_names = { 0: 'Cluster 1', 
                 1: 'Cluster 2',
                 2: 'Cluster 3',
                 3: 'Cluster 4',
                 4: 'Cluster 5'}

#/*** k means ***/
from sklearn.cluster import KMeans
for i in range(2,6):
    
    num_clusters = i
    
    km = KMeans(n_clusters=num_clusters)
    
    km.fit(dtm)
    
    clusters = km.labels_.tolist()
    
    cosdist = 1 - cosine_similarity(dtm)
    
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    
    pos = mds.fit_transform(cosdist)  # shape (n_components, n_samples)
    
    xs, ys = pos[:, 0], pos[:, 1]
    
    
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=GenreDF.Play)) 
    
    #group by cluster
    groups = df.groupby('label')
    
    
    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    
    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
                label=cluster_names[name], color=cluster_colors[name], 
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='on',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='on')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='on',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='on')
        
    ax.legend(numpoints=1)  #show legend with only 1 point
    
    #add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  
    
        
        
    plt.show() #show the plot
