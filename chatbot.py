# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:53:15 2020

@author: MMOHTASHIM
"""

##Lets Start Building the Chatbot

from Machine_Learning_Model import *
from data_creation import *
from user_chatbot_middle import *
import numpy as np
import pandas as pd


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



symptom_dictionary,diease_dictionary=dictionary_creation()##reverse it
print(list(symptom_dictionary.keys()))


def main_brain():
    print('Hello Sir/Mam I am Medi Robot and I will help you to diagonse your illness,Releax as I will help you')
    print('But First Let me ask you some questions')
    intial_info=True
    ###############INTIAL INFO BLOCK###################################################
    while intial_info:
        name=True
        while name:
            print('Sir/Mam What is your name?')
            user_name=input()
            if user_name is not None and user_name!='' and user_name!=' ':
                name=False
                print('Greetings {}'.format(user_name))
            else:
                print('Enter valid name')
                
        gender=True
        while gender:
            print('{} What is your gender?'.format(user_name))
            user_gender=input()
            if user_gender.lower() in ['male','female']:
                gender=False
                print('Noted')
            else:
                print('Enter valid gender')
        
        age=True
        while age:
            print('{} What is your age?'.format(user_name))
            user_age=int(input())
            if user_age in list(range(6,150)):
                age=False
                print('Noted')
            else:
                print('Enter valid age')
                
        print('So you are {} with gender:{} and age:{}'.format(user_name,user_gender,user_age))
        print('If satisfied with the information,please type in yes')
        
        confirmed=input()
        if confirmed.lower()=='yes':
            intial_info=False
        else:
            print('Please enter the information again')
    #####################################################################################################
    
    #################MAIN MEDICAL DIANOSIS################################################################
    intial_dieases=True
    while intial_dieases:
        print('                           ')
        print('Okay Let me help you to diagnose your illness')
        print('Firstly tell me, what kind of symptoms are you experiencing?Please be straightfoward')
        
        print('Please write in a clear way as it help me to understand better')
        print('This way I will be more accurate')
        user_intial_symtomps=input()
        
        print('Hmmmm! please wait')
        output=machine_learning_prediction(user_intial_symtomps)
        if len(output)==1:
            print('You are experiencing from {}'.format(output))
            break
        
        print('Okay I have noted your intial symtomps')
        print('                ')
        print('You might be suffering from any of the following:')
        print('                                     ')
        print('{}'.format(output))
        print('                                         ')
        print('But Dont Worry Let me further analyse and find the specific diease but for that I need more input from you')
        
        
        
        
        random_questions=['OKay Are you experiencing','Do you feel?',
                          'Hmmmm,are you feeling?','Do you have?','Please tell me do you have?',
                          'Let me know if you have?','Have you experience?']
        
        df_logic=pd.read_csv('Logic_Symtom_Data.csv')
        
        df_logic=df_logic[['Diease_Code','Symptom_Code']]
        df_logic['Score']=0
        

        detected_sym=[]
        print(output)
        for diease in output:##Loop over all intial possible dieases
            symtoms=diease_specific_detector(diease)#####Specific symtoms for diease detected from intial diagonsis
            for i_s in symtoms:##Loop over symtoms for intial diease
                if i_s in detected_sym:##Not asking for same symtoms again and again
                    pass
                else:
                    
                    if i_s in complex_symtoms.keys():#########breaking complex symtomps
                        orginal_is=i_s
                        i_s=complex_symtoms[i_s]
                        
                    print('Please type in only yes or no')
                    print('####################')
                    index_q=np.random.randint(0,len(random_questions)-1) 
                    print(random_questions[index_q],'{}'.format(i_s))
                    detected_sym.append(i_s)
            
                    user_answer=input()
                    
                    if user_answer.lower()=='yes':
                        try:##If user says yes then give score(+1) to all dieases that have the revelant symtoms
                            r_index=df_logic[(df_logic['Symptom_Code']==symptom_dictionary[i_s])].index.tolist()
                            df_logic.loc[r_index,'Score']=df_logic.loc[r_index,'Score']+1
                        except:
                            r_index=df_logic[(df_logic['Symptom_Code']==symptom_dictionary[orginal_is])].index.tolist()
                            df_logic.loc[r_index,'Score']=df_logic.loc[r_index,'Score']+1
                    
        
        intial_dieases=False
        print(diease_dictionary[df_logic.loc[df_logic['Score'].idxmax(),'Diease_Code']])
        df_logic.to_csv('Logic_Symtom_Data--{}--{}---{}.csv'.format(user_name,user_gender,
                             user_age))##Storing the data for revelant user.
        
    
    ########################################################################
    
    
    
            
            
    
main_brain() 