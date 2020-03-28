# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:58:44 2020

@author: MMOHTASHIM
"""

import numpy as np
import pickle
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import argparse
from data_creation import *
import pandas as pd


###complex_symptoms in simple terms for better word embedding and user-friendly terms
complex_symtoms={'polydypsia':'thirst',
'orthopnea':'shortness of breath',
'weepiness':'sadness',
'hypokinesia':'decreased bodily movement loss of muscle movement',
'rhonchus':'pitched rattling lung sounds',
'haemoptysis':'coughing up blood',
'apyrexial':'absence of a fever',
'dysuria':'painful or difficult urination',
'ecchymosis':'discoloration of the skin bruising',
'orthostasis':'decrease in blood pressure',
'transaminitis':' elevated liver enzymes inflammation in the liver',
'asterixis':'tremor of the hands',
'prostatism':'obstruction of the bladder enlarged prostate gland',
'formication':'sensation like insects crawling over the skin',
'hypesthesia':'loss of sensitivity',
'cardiomegaly':'abnormal enlargement of the heart',
'cicatrisation':'process of a wound healing to produce scar tissue',
'hypometabolism':'abnormally low metabolic rate',
'oliguria':'small amounts of urine',
'photopsia':'perceived flashes of light in the field of vision',
'macule':'area of skin discoloration',
'atypia':'structural abnormality in a human cell',
'stridor':'vibrating noise when breathing',
'aphagia':'inability or refusal or difficulty to swallow',
'fremitus':'sensation felt as chest vibrates',
'bradykinesia':'slowness of movement',
'hematochezia':'passage of fresh blood through the anus',
'egophony':'resonance of voice sounds heard from lungs',
'paraparesis':'paralysis of the lower limbs',
'dysesthesia':'damage to peripheral nerves',
'polymyalgia':'pain and stiffness of the muscles',
'retropulsion':'disorder of locomotion not able to walk',
'hypersomnolence':'sleepiness',
'urinoma':'accumulation of urine in the body',
'hypoalbuminemia':'proteins liver problem',
'pustule':' blister or pimple on the skin',
'titubation':'nervous disorder',
'dysdiadochokinesia':'impaired ability to perform rapid, alternating movements',
'monocytosis':'chronic inflammation',
'tenesmus':'disorder of the rectum',
'fecaluria':'mixture of feces and urine and pee',
'pneumatouria':'passage of gas or air in urine and pee',
'hydropneumothorax':'air and fluid in lungs',
'uncoordination':'lacking coordination',
'fatigability':'fatigue',
'primigravida':'woman pregnant',
'proteinemia':'abnormal quantities of protein in the urine and pee',
'phonophobia':'extreme sensitivity to light',
'charleyhorse':'stiffness in an arm or leg',
'hypertonicity':'abnormally high tension',
'prodrome':'early symptom disease or illness',
'hypoproteinemia':'abnormally low level of protein'}
stop_words = set(stopwords.words('english'))
def create_symptom_embedding(create_embedding,glovec_dir):
    '''''
    Input-create_embedding---Whether You want to create new embedding for the symtoms
          glovec_dir-Glovec.txt file location
    
    This function creates a word embedding for all of the symptoms in my dataset.Since, some
    symptoms have complex medical terms attached to them. I was able to identify them and break 
    down them into simple words with clear defination(Captured from good medical sources).It saves 
    the word embedding vector for each symptoms into a dictionary and pickle/saves that 
    file called symptom_embebdding.
    
    Returns None
    
    
    '''
    
    ########################################################
    symptom_dictionary,diease_dictionary=dictionary_creation()
    

     ##loading glovec embedding
    if create_embedding:
        # load the whole embedding into memory
        embeddings_index = dict()
        f = open(glovec_dir,encoding="utf8")
        for line in f:
        	values = line.split()
        	word = values[0]
        	coefs = np.array(values[1:], dtype='float32')
        	embeddings_index[word] = coefs
        f.close()
        with open('glove_embebdding.pickle','wb') as file:
                pickle.dump(embeddings_index,file)
    else:       
        with open('glove_embebdding.pickle','rb') as file:
            embeddings_index=pickle.load(file)
    ##################################################
    
    
    embedding_dictionary={}
    for symptom in symptom_dictionary.keys():
        if symptom!='':
            embedding_vectorss=[]
            for word in symptom.split():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_vectorss.append(embedding_vector)
            if embedding_vectorss==[]:
                
                complex_symptom_simple=word_tokenize(complex_symtoms[symptom])
                complex_symptom_simple = [w.lower() for w in complex_symptom_simple]
                table = str.maketrans('', '', string.punctuation)
                complex_symptom_simple = [w.translate(table) for w in complex_symptom_simple]    
                complex_symptom_simple = [word for word in complex_symptom_simple if word.isalpha()]
               
                
                for simple_symptom in complex_symptom_simple:
                    embedding_vector = embeddings_index.get(simple_symptom)
                    if embedding_vector is not None:
                        embedding_vectorss.append(embedding_vector)
                    
            embedding_dictionary['{}'.format(symptom)]=np.mean(np.array(embedding_vectorss),axis=0,keepdims=True)
                    
                
        
    with open('symptom_embebdding.pickle','wb') as file:
                pickle.dump(embedding_dictionary,file)
    
    return None


def User_Input_symtom(user_input='My Chest Hurts',min_symptoms=5):
    
    ''''
    Input-User Input from chatbot interaction
          min_symptoms-How many symtoms to be used for Machine Learning Model 
    This is a helper function for chatbot.py. This function takes user input which is assumed to be
    a sentence telling his symthoms and that through created word embedding identify the revelant 
    embedding from the sentence so that these sympthoms could be fed to the Machine Learning model
    trained in Machine_Learning_Model.py and revelant diease could be detected.
    
    Returns symptoms_detected-List of features to be feed to Machine Learning Model(Will be Processed)
    
    
    
    '''
    
    with open('glove_embebdding.pickle','rb') as file:
        embeddings_index=pickle.load(file)
    
    with open('symptom_embebdding.pickle','rb') as file:
        symptom_embedding_dictionary=pickle.load(file)
    
    ###Cleaning User Sentence Pipeline
    user_words=word_tokenize(user_input)
    user_tokens = [w.lower() for w in user_words]
    table = str.maketrans('', '', string.punctuation)
    user_stripped = [w.translate(table) for w in user_tokens]    
    user_words = [word for word in user_stripped if word.isalpha()]
    ####End of Pipeline
   
    
    ##Finding Revelant Embedding for User
    embedding_vectorss=[]
    for word in user_words:
        if word not in stop_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                    
                    embedding_vectorss.append(embedding_vector)

    ##Calculating l2 norm
    final_embedding_user=np.mean(np.array(embedding_vectorss),axis=0,keepdims=True)
    L2_norms_symothoms={}  
    
    for symptom in list(symptom_embedding_dictionary.keys()):
        L2_norm=np.linalg.norm(final_embedding_user[0]-symptom_embedding_dictionary[symptom][0])
        L2_norms_symothoms[symptom]=L2_norm
    
    L2_norms_symothoms={k: v for k, v in sorted(L2_norms_symothoms.items(), key=lambda item: item[1])}
 
    
    return list(L2_norms_symothoms.keys())[:min_symptoms]


def machine_learning_prediction(user_input):
    L2_norms_symothoms=User_Input_symtom(user_input,min_symptoms=3)

    symptom_dictionary,diease_dictionary=dictionary_creation()
    _,_,df=data_machine_learning(load=True)

    symptom_code=[symptom_dictionary[symptom] for symptom in L2_norms_symothoms]
    print(L2_norms_symothoms)
    revelant_descriptors=[]
    for code in symptom_code:
        revelant_descriptors.append(df[(df[code]==1)].index.tolist())
    
    intial_possible_dieases=[]
    
    for descriptors in revelant_descriptors:
        for diease_code in descriptors:
             intial_possible_dieases.append(diease_dictionary[diease_code])
    return intial_possible_dieases

 
    
####Now Moving Ahead USE LOGIC in the Paper FOUND and continue..........
    #######Intial Dieases FOund from this function
    ####For each diease ask for whether the symptom is present or not
                ###Plus one score if user says yes
                ###else zero
    ###Diease with the highest scire from the intial search are kept
    #############################################
def diease_specific_detector(diease):
    symptom_dictionary,diease_dictionary=dictionary_creation()##reverse it
    pass
            
            
    
   
##Build Function which will feed data to Machine Learning Model

if __name__=="__main__":
#       parser=argparse.ArgumentParser(description="User_Testing for Main Chatbot")
#       
#       parser.add_argument('-e','--Embedding',type=bool,help="Wether want to create new embedding or not;if not then there must be existing embedding")
#       parser.add_argument('-i','--User_Input',help="Symtoms Sentence---User_Input")
#       parser.add_argument('-p','--glovec',help='Directory for Glovec.TXT')
#
#       args=parser.parse_args()
#       
#       create_symptom_embedding(create_embedding=args.Embedding,glovec_dir=args.glovec)
#       User_Input_symtom(user_input=args.User_Input,min_symptoms=5)
       create_symptom_embedding(create_embedding=False,glovec_dir=r'C:\Users\MMOHTASHIM\Anaconda3\libs\Cilent-Project-Medical Chatbot\glove.6B.300d.txt')
       print(machine_learning_prediction(user_input='I have pain in my foot'))
    
    