# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:53:15 2020

@author: MMOHTASHIM
"""

##Lets Start Building the Chatbot

from Machine_Learning_Model import *
from data_creation import *
from user_chatbot_middle import *
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
        
        print('Please write in the following way: I am feeling <symptom-1> and <symptom-2> and .....')
        print('This way I will be more accurate')
        user_intial_symtomps=input()
        
        print('Hmmmm! please wait')
        output=machine_learning_prediction(user_intial_symtomps)
        if len(output)==1:
            print('You are experiencing from {}'.format(output))
            break
        
        print('Okay I have noted your intial symtomps')
        print('You might be suffering from any of the following:')
        print('{}'.format(output))
        
        print('But Dont Worry Let me further analyse and find the specific diease but for that I need more input from you')
        break
        for diease in output:
            diease_specific_detector(diease)
        
        
        
    
    
    
    ########################################################################
    
    
    
            
            
    
main_brain() 