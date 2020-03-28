# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:27:26 2020

@author: MMOHTASHIM
"""

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import pickle
import os
import random

def separator_code(instance):
    ''''
    A helper function which sperates diease/symptom code from actual name
    '''
    code=instance[:13]
    return code

def separator_name(instance):
    ''''
    A helper function which sperates diease/symptom code from actual name
    '''
    name=instance[14:]
    
    if '^' in instance:
        index_limit=instance.index('^')
        name=instance[14:index_limit]
    
    return name

def dictionary_creation():
    '''''
    input None
    
    This functions creates a dictionary 
    for both diease and symtoms with values as their name and key for their medical code
    
    return code-symptom dictionary
    '''
#    'http://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html'  #linkt for the revelant dataset
    df=pd.read_csv('Data_Set.csv')

    diease_code=df['Disease'].map(separator_code,na_action='ignore').dropna().tolist()
    diease_name=df['Disease'].map(separator_name,na_action='ignore').dropna().tolist()
    
    symptom_code=df['Symptom'].map(separator_code,na_action='ignore').dropna().tolist()
    symptom_name=df['Symptom'].map(separator_name,na_action='ignore').dropna().tolist()

    symptom_dictionary={k:v for k,v in list(zip(symptom_name,symptom_code))}
    diease_dictionary={k:v for k,v in list(zip(diease_code,diease_name))}
    
    ##This is just a bug I found in the dataset
    del symptom_dictionary['']
    symptom_dictionary['pulmonary diseases']='UMLS:C0032739'
    return symptom_dictionary,diease_dictionary


def fill_in_nan(series):
    ''''
    input Pandas Series
    
    Function to Fill the Missing Nan values with diease names
    
    return cleaned df
    
    ''' 
    cleaned_series=[]
    for value in series:
        if type(value)==str:
            value_current=value
        cleaned_series.append(value_current)
    return np.array(cleaned_series)
    


def data_machine_learning(load):
    ''''
    input: load-Boolean Variable to load a Existing CSV FILE
    
    This Cleans the dataframe so that data is preprocessed 
    and ready for machine learning input
    
    returns two arrays X--Input features for machine learning model 
    and y-label for machine learning model
    
    '''
    if load:
        df_new=pd.read_csv('New-Data-Set.csv',index_col='Disease')
        X=df_new[df_new.columns.tolist()].values.tolist()
        y=df_new.index.tolist()  
        
    else:
        df=pd.read_csv('Data_Set.csv')
    
        df['Diease_Code']=df['Disease'].map(separator_code,na_action='ignore')
        df['Symptom_Code']=df['Symptom'].map(separator_code,na_action='ignore')
        
        df['Diease_Code']=fill_in_nan(df['Diease_Code'])
        df.to_csv('Logic_Symtom_Data.csv')
        
        
        df=df.drop(['Disease','Symptom','Count of Disease Occurrence'],1)
    
        df_new=pd.get_dummies(df['Symptom_Code'])

        df_new['Disease']=df['Diease_Code']
        df_new=df_new.groupby('Disease').sum()
        df_new.to_csv('New-Data-Set.csv')

        X=df_new[df_new.columns.tolist()].values.tolist()
        y=df_new.index.tolist()


    return X,y,df_new




def upsample_X_machine_learning(upsample_number,max_features_off):
    ''''
    inputs upsample_number-How much each instance would be upsampled
           max_features_off-Each Symptom Random Turning on and off
           
    Due to Lack of Data my machine learning model was performing sub-optimially
    therefore I upsampled my data. The logic is simple for each diease,I randomly
    picked up a Symptom from a number of revelant  Symptoms for the diease and turned 
    it off(making sure that not all symptoms are turned off)-by turning turning off
    I mean converting 1 to 0-Please see the New_Data_Set for Clarity. 
    I did it for a number of times and generated new instances of data
    
    return Upsamled_X,Upsampled_y
    '''
    X,y,df_new=data_machine_learning(True)

    for diease in df_new.index.tolist():

        for _ in range(upsample_number):
            diease_array=np.array(df_new.loc[diease,:])     
            truth_list_index=np.where(diease_array==1)[0]##Randomly picking up the symptoms      
            
            random_pick_features=np.random.randint(1,max_features_off)       
            random_index=random.choices(truth_list_index,k=random_pick_features)       
            
            diease_array[random_index]=0
            X.append(diease_array)
            y.append(diease)
    
    Zipped=list(zip(X, y))##Randomly suffling the X and y to avoid biasness
    random.shuffle(Zipped)
    X, y = zip(*Zipped)
    
    X=np.array(X)
    y=np.array(y)
    
    return X,y
    
            




     
if __name__=="__main__":
#    
#    parser=argparse.ArgumentParser(description="Create the Revelant DataSet for Machine Learning Model")
#
#    parser.add_argument('-s','--UPSAMPLE',type=int,help="Upsample_number")
#    parser.add_argument('-f','--MAX_FEATURES',type=int,help="max_features_off")
#
#    args=parser.parse_args()
#     
#    X,y=upsample_X_machine_learning(upsample_number=args.UPSAMPLE,max_features_off=args.MAX_FEATURES)
    data_machine_learning(load=False)

     
    
    
    

        

        

     
     
     
    