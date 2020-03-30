# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:31:03 2020

@author: MMOHTASHIM
"""

##Lets Start Building the Chatbot

from Machine_Learning_Model import *
from data_creation import *
from user_chatbot_middle import *
import numpy as np
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

reverse_complex_symtoms={v:k for k,v in complex_symtoms.items()}

symptom_dictionary,diease_dictionary=dictionary_creation()##reverse it


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
        random_questions=['OKay Are you experiencing','Do you feel?',
                          'Hmmmm,are you feeling?','Do you have?','Please tell me do you have?',
                          'Let me know if you have?','Have you experience?']
        
        
        
                
        df_logic=pd.read_csv('Logic_Symtom_Data.csv')
        
        df_logic=df_logic[['Diease_Code','Symptom_Code']]
        df_logic['Score']=0
        
        print('                           ')
        print('Okay Let me help you to diagnose your illness')
        print('Firstly tell me, what kind of symptoms are you experiencing?Please be straightfoward')
        
        print('Please write in a clear way as it help me to understand better')
        print('This way I will be more accurate')
        user_intial_symtomps=input()
        
        print('Hmmmm! please wait')
        symtomps=machine_learning_prediction_2(user_intial_symtomps)

        
        print('Okay I have noted your intial symtomps')
        print('                            ')
        
        print('Okay let me further ask you some more questions so I could be more specific about your problem')
        print('                                   ')
        
        print('I will ask you a bunch of questions ;please answer yes or no')
        
        
        for sym in symtomps:
            index_q=np.random.randint(0,len(random_questions)-1) 
            print(random_questions[index_q],'{}'.format(sym))
            
            user_answer=input()
                    
            if user_answer.lower()=='yes':
                if sym in complex_symtoms.values():
                    real_symtom=reverse_complex_symtoms[sym]
                    r_index=df_logic[(df_logic['Symptom_Code']==symptom_dictionary[real_symtom])].index.tolist()
                    df_logic.loc[r_index,'Score']=df_logic.loc[r_index,'Score']+1
                else:
                    r_index=df_logic[(df_logic['Symptom_Code']==symptom_dictionary[sym])].index.tolist()
                    df_logic.loc[r_index,'Score']=df_logic.loc[r_index,'Score']+1
            
            
        intial_dieases=False
        print(diease_dictionary[df_logic.loc[df_logic['Score'].idxmax(),'Diease_Code']])
        df_logic.to_csv('Logic_Symtom_Data--{}--{}---{}.csv'.format(user_name,user_gender,
                             user_age))##Storing the data for revelant user.
        
    
    ########################################################################
            
            
if __name__=="__main__":   
	main_brain() 