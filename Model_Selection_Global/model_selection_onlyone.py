#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '../')
import os
from os import listdir
from os.path import isfile, join

import modelSelectionbyLevel_scores2 as msL
import tools as taxTools
import logging
from imp import reload
from sklearn.preprocessing import LabelEncoder


from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, classification_report, make_scorer, roc_auc_score, confusion_matrix


# <h2>Model selection</h2>

# <h3>Train the models with classifiers</h3>
# <ul>
#     <li>Decision Tree</li>
#     <li>Random Forest</li>
#     <li>Extra Trees</li>
# </ul>
# 
# <h3>Compare to use this metrics</h3>
# <ul>
#     <li>F1 micro</li>
#     <li>F1 macro</li>
#     <li>Acurracy</li>
#     <li>Matthews correlation coefficient</li>
# </ul>

# In[41]:


reload(logging)
logging.basicConfig(filename='log.log', level=logging.INFO, format='%(message)s')

scores = pd.DataFrame()

fold = 10 #set with the fold quantity
cores = 40 #set with the number of cores you have on the server that you are running
label = "global" #choose a label to put on the files names
classes = pd.DataFrame()
target_column = "CLASS" #set with the target column name on the datasetd[]


print("Reading file")
logging.info("\nReading file")
taxonomyTable = pd.read_csv("../CATH_balanced.csv", sep=",", dtype={target_column: 'category'}, nrows=500)
count = taxTools.Count(taxonomyTable, target_column )

try:
    dir = "/predict"       
    os.makedirs(dir)

except OSError:
    pass

l=[]
X, y_aux, yBin_aux = msL.PreProcessing(taxonomyTable, target_column, count, l)
indexs = X['Unnamed: 0']
X = X.drop(columns=['Unnamed: 0'])
label_encoder = LabelEncoder()
y_code = label_encoder.fit_transform(y_aux)

nameY = "y_"+label+".csv"
y_aux.to_csv(nameY, sep="\t")

l = ["predict_"+str(i+2)+"_"+label_encoder.classes_ for i in range(fold-1)]
flat_classes = [item for sublist in l for item in sublist]

scores, df_predict_DecisionTreeClassifier_aux, df_predict_RandomForestClassifier_aux, df_predict_ExtraTreesClassifier_aux =  msL.DiferentsFolds_Trees_Pred(X, y_code, fold, cores, label, count, label_encoder)

scores.columns = ["model", "fold", "time_train", "time_test", "score_train", "score_test"]
nameScores = "scores_"+label+".csv"
scores.to_csv(nameScores, sep="\t")

df_predict_DecisionTreeClassifier_aux.index = indexs.index
name = "predict/DecisionTree_"+label+".csv"
df_predict_DecisionTreeClassifier_aux.to_csv(name, sep="\t")

df_predict_RandomForestClassifier_aux.index = indexs.index
name = "predict/RandomForest_"+label+".csv"
df_predict_RandomForestClassifier_aux.to_csv(name, sep="\t")

df_predict_ExtraTreesClassifier_aux.index = indexs.index
name = "predict/ExtraTrees_"+label+".csv"
df_predict_ExtraTreesClassifier_aux.to_csv(name, sep="\t")