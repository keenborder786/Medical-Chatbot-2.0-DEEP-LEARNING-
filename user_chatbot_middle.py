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


##Will pickle it later,breaking of complex symtom terms for better lingustic understanding
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
'hypoproteinemia':'abnormally low level of protein',
'asthenia':'abnormal physical weakness or lack of energy',
'syncope':'temporary loss of consciousness caused by a fall in blood pressure',
'vertigo':'a sensation of whirling and loss of balance, associated particularly with looking down from a great height, or caused by disease affecting the inner ear or the vestibular nerve; giddiness',
'palpitation':'noticeably rapid, strong, or irregular heartbeat',
'angina pectoris':' condition marked by severe pain in the chest, often also spreading to the shoulders, arms, and neck, owing to an inadequate blood supply to the heart',
'polyuria':'production of abnormally large volumes of dilute urine',
'rale':' abnormal rattling sound heard when examining unhealthy lungs with a stethoscope',
'dyspnea on exertion':'Shortness of breath on exertion/urine/pee',
'pleuritic pain':'intense sharp, stabbing, or burning pain',
'tachypnea':'abnormally rapid breathin',
'malaise':'general feeling of discomfort, illness, or unease whose exact cause is difficult to identify',
'dyspnea':'difficult or laboured breathing',
'dysarthria':'difficult or unclear articulation of speech',
'hemiplegia':'paralysis of one side of the body',
't wave inverted':'occurs when blood flow to the heart muscle (myocardium) is obstructed by a partial or complete blockage of a coronary artery by a buildup of plaques.',
'presence of q wave':'heart attack',
'erythema':'superficial reddening of the skin, usually in patches, as a result of injury or irritation causing dilatation of the blood capillaries',
'hepatosplenomegaly':'liver and spleen swell beyond their normal size',
'hematuria':'the presence of blood in urine',
'renal angle tenderness':'back pain near kidney',
'pruritus':'severe itching of the skin, as a symptom of various ailments',
'hyponatremia':'low sodium concentration in the blood',
'Hemodynamic Stability':'force at which the heart pumps blood is stable',
'guaiac positive':'hidden (occult) blood in a stool sample',
'monoclonal':' antibodies that are made by identical immune cells',
'haemorrhage':'scape of blood from a ruptured blood vesse',
'pallor':'an unhealthy pale appearance',
'heme positive':'blood in the feces',
'arthralgia':'pain in a joint',
'sputum purulent':'pus,composed of white blood cells, cellular debris, dead tissue, serous fluid, and viscous liquid',
'hypoxemia':'abnormally low concentration of oxygen in the blood',
'hypercapnia':'condition of abnormally elevated carbon dioxide (CO2) levels in the blood',
'patient non compliance':'not take a prescribed medication or follow a prescribed course of treatment',
'hyperkalemia':'potassium level in your blood thats higher than normal',
'urgency of\xa0micturition':'sudden, compelling urge to urinate',
'ascites':'the accumulation of fluid in the peritoneal cavity, causing abdominal swelling',
'enuresis':'involuntary urination, especially by children at night/ bedwet/ urination during sleep',
'lesion':'region in an organ or tissue which has suffered damage through injury or disease, such as a wound, ulcer, abscess, or tumour.',
'cushingoid facies':'face develops a rounded appearance due to fat deposits on the sides of the face',
'emphysematous change':'progressive disease of the lungs that primarily causes shortness of breath',
'muscle hypotonia':'reduced muscle strength',
'hyperacusis':'increased sensitivity to certain frequencies and volume ranges of sound',
'cyanosis':'bluish discoloration of the skin due to poor circulation or inadequate oxygenation of the blood.',
'clonus':'muscular spasm involving repeated, often rhythmic, contractions',
'anorexia':'ack or loss of appetite for food',
'anosmia':'the loss of the sense of smell, either total or partial',
'metastatic lesion':'spread of cancer cells',
'hemianopsia homonymous':'visual field loss on the left or right side eye',
'hematocrit decreased':'percentage of red blood cells is below the lower limits of normal',
'aura':'warning sensation experienced before an attack of epilepsy or migraine',
'myoclonus ':'spasmodic jerky contraction of groups of muscles',
'left\xa0atrial\xa0hypertrophy':'enlargement and thickening (hypertrophy) of the walls of your hearts main pumping chamber',
'catatonia':'abnormality of movement and behaviour arising from a disturbed mental state (typically schizophrenia',
'paresthesia':'an abnormal sensation, typically tingling or pricking (‘pins and needles’)',
'gravida 0':'pregnancy and birth',
'lung nodule':'small growth on the lung and can be benign or malignant',
'distended abdomen':'usually used to refer to distension or swelling of the abdomen',
'macerated skin':'softening and breaking down of skin resulting from prolonged exposure to moisture',
'sinus rhythm':'Abnormal Heart Beat/Fast Heart Beating',
'hypersomnia':'feel excessive sleepiness during the day',
'hyperhidrosis disorder':'excessive sweating',
'mydriasis':'dilation of the pupil of the eye',
'extrapyramidal sign':'remor, slurred speech, akathesia, dystonia, anxiety, distress, paranoia',
'splenomegaly':'abnormal enlargement of the spleen',
'photophobia':'extreme sensitivity to light',
'cachexia':'weakness and wasting of the body due to severe chronic illness',
'hypocalcemia result':'muscle cramps in your legs or your arms',
'hypothermia, natural':'drop in body temperature to dangerously low levels',
'stupor':'state of near-unconsciousness or insensibility',
'hirsutism' :'abnormal growth of hair on face and body',
'urge incontinence':'unstoppable urge to urinate',
'qt interval prolonged':'heart rhythm disorder that can cause serious irregular heart rhythms',
'ataxia':'loss of full control of bodily movements',
"Heberden's node":'bony enlargement of the terminal joint of a finger commonly associated with osteoarthritis',
"hepatomegaly":'abnormal enlargement of the liver',
'sciatica':'pain affecting the back, hip, and outer side of the leg',
'colic abdominal':'abdominal pain',
'hypokalemia':'deficiency of potassium in the bloodstream',
'nasal discharge present':'runny nose',
'achalasia':'difficulty in passing food into the stomach',
'posterior\xa0rhinorrhea':'sore throat and/or coughing',
'todd paralysis':'seizure is followed by a brief period of temporary paralysis',
'myalgia':'pain in a muscle or group of muscles',
'dyspareunia':'difficult or painful sexual intercourse',
"poor dentition":'missing teeth',
"inappropriate affect":"reduction in an individual's expressive range and the intensity of emotional responses",
'welt':'red, raised mark or scar; a weal',
'tinnitus':'ringing or buzzing in the ears',
'para 2':'two pregnancies and two deliveries after 24 weeks',
'hyperemesis':'severe or prolonged vomiting',
"regurgitates after swallowing":"swallowed food back up through one's throat and out the mouth--difficulty in eating food",
"pulsus\xa0paradoxus":"high blood pressure and fast heart beat during inspiration",
"gravida 10":"number of times someone is or has been pregnant",
"bruit":"a sound, especially an abnormal one, heard through a stethoscope; a murmur",
"scleral\xa0icterus":'Yellowish Skin and Face',
"blanch":"white or pale skin color",
"elation":"great happiness and exhilaration",
"sedentary":"inactive",
"flare":"worsening in severity of a disease or condition",
"pericardial friction rub":"discomfort that radiates in the chest",
"hoard":"hoarding disorder",
"para 1":"given birth once",
"Murphy's sign":"Pain Near gallbladder",
"flatulence":"accumulation of gas/food gas",
}
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
        glovec_words=[]
        for line in f:
            values = line.split()
            word = values[0]
            glovec_words.append(word)
            coefs = np.array(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        with open('glove_embebdding.pickle','wb') as file:
                pickle.dump(embeddings_index,file)
        with open('glove_words.pickle','wb') as file:
                pickle.dump(glovec_words,file)
    else:       
        with open('glove_embebdding.pickle','rb') as file:
            embeddings_index=pickle.load(file)
    ##################################################
    
    
    embedding_dictionary={}
    
    for symptom in symptom_dictionary.keys():
        embedding_vectorss=[]
        if symptom in complex_symtoms.keys():

                complex_symptom_simple=word_tokenize(complex_symtoms[symptom])
                complex_symptom_simple = [w.lower() for w in complex_symptom_simple]
                table = str.maketrans('', '', string.punctuation)
                complex_symptom_simple = [w.translate(table) for w in complex_symptom_simple]    
                complex_symptom_simple = [word for word in complex_symptom_simple if word.isalpha()]
               
                
                for simple_symptom in complex_symptom_simple:
                    embedding_vector = embeddings_index.get(simple_symptom)
                    if embedding_vector is not None:
                        embedding_vectorss.append(embedding_vector)
                        
        else:
            for word in symptom.split():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_vectorss.append(embedding_vector)
            
                    
        embedding_dictionary['{}'.format(symptom)]=np.mean(np.array(embedding_vectorss),axis=0,keepdims=True)
                    
                
        
    with open('symptom_embebdding.pickle','wb') as file:
                pickle.dump(embedding_dictionary,file)
    
    return None


def User_Input_symtom_embedding(user_input='My Chest Hurts',min_symptoms=5):
    
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
    '''
    input-user intial symthoms captured from interaction with chatbot.
    
    Get Intial Symtoms from user and then through User_Input_symtom_embedding,
    capture the revelant symthom.Once Captured, this function calculates the intial
    possible dieases that the user can have.
    
    return intial_dieases-an array of intial dieases that user can have.
    
    '''
    
    
    
    L2_norms_symothoms=User_Input_symtom_embedding(user_input,min_symptoms=10)

    
    symptom_dictionary,diease_dictionary=dictionary_creation()
    _,_,df=data_machine_learning(load=True)

    symptom_code=[symptom_dictionary[symptom] for symptom in L2_norms_symothoms]
   
    revelant_descriptors=[]
    for code in symptom_code:
        revelant_descriptors.append(df[(df[code]==1)].index.tolist())
    
    intial_possible_dieases=[]
    
    for descriptors in revelant_descriptors:
        for diease_code in descriptors:
             intial_possible_dieases.append(diease_dictionary[diease_code])
    intial_possible_dieases=list(set(intial_possible_dieases))
    return intial_possible_dieases

 

def machine_learning_prediction_2(user_input):
    '''
    pass
    
    '''        
    L2_norms_symothoms=User_Input_symtom_embedding(user_input,min_symptoms=10)
    simple_symtoms=[]
    for simple_symothom in L2_norms_symothoms:
        if simple_symothom in complex_symtoms.keys():
            simple_symtoms.append(complex_symtoms[simple_symothom])
        else:
            simple_symtoms.append(simple_symothom)
        
   
    return simple_symtoms
        


def diease_specific_detector(diease):
    '''
    input-Intial Diease Detected from machine_learning_prediction
    
    
    This function will identify the revelant symtomps for that diease and return those
    symtomps to user.User will tell us whether they are experiencing those symtomps or not.
    
    return symtomps
    
    
    '''
    
    
    
    symptom_dictionary,diease_dictionary=dictionary_creation()##reverse it
    reversed_diease_dictionary={v:k for k,v in diease_dictionary.items()}

    reversed_symptom_dictionary={v:k for k,v in symptom_dictionary.items()}
    
    df_logic=pd.read_csv('Logic_Symtom_Data.csv')
    df_logic=df_logic[['Diease_Code','Symptom_Code']]
    
    diease_code=reversed_diease_dictionary[diease]
    revelant_symtoms=df_logic[(df_logic['Diease_Code']==diease_code)]['Symptom_Code'].values.tolist()
    
    symtoms=[reversed_symptom_dictionary[s_c] for s_c in revelant_symtoms]
    
    return symtoms
    
    
            
            
    
   
##Build Function which will feed data to Machine Learning Model

if __name__=="__main__":
       create_symptom_embedding(create_embedding=False,glovec_dir=r'C:\Users\MMOHTASHIM\Anaconda3\libs\Cilent-Project-Medical Chatbot\glove.6B.300d.txt')
#       print(machine_learning_prediction(user_input='I have pain in my foot'))
       machine_learning_prediction_2('My ears are hurting real bad')
    
    