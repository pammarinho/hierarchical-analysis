from joblib import Parallel, delayed

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, classification_report, make_scorer, roc_auc_score, confusion_matrix

from sklearn.model_selection import StratifiedKFold

import numpy as np

import time
import logging
import os

from scipy import stats
from scipy.sparse.csgraph import connected_components

from sklearn import datasets

import pandas as pd
from memory_profiler import profile

#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import matplotlib.lines as mlines

############ Pre-processing database ############

def PreProcessing(taxonomyT, level, count, levelDrop):
    print("Runing pre-processing")
    logging.info("\nRuning pre-processing\n")
    taxonomyFinalAux = pd.eval('taxonomyT.copy()')
    
    print("Level selected: ", level)
    taxonomyFinalAux = taxonomyFinalAux.drop(levelDrop, axis=1)
    
    y = taxonomyFinalAux[level]
    yBin = label_binarize(y, classes=count.index)
    taxonomyFinalAux = taxonomyFinalAux.drop(columns=[level])
    X = taxonomyFinalAux
    print("Done pre-processing")
    logging.info("\nDone pre-processing\n")
    return pd.eval('X'), pd.eval('y'), pd.eval('yBin')

############ Create Model with Cross Validation ############

@profile
def DiferentsFolds(X, Y, folds, cores, label, countkingdom, label_encoder):
    
    try:
        dir = 'predict'       
        os.makedirs(dir)
        dir = 'prob'       
        os.makedirs(dir)
        
    except OSError:
        pass


    scores = pd.DataFrame()

    df_predict_SGDClassifier = pd.DataFrame()
    df_predict_GaussianNB = pd.DataFrame()
    df_predict_MultinomialNB = pd.DataFrame()
    df_predict_DecisionTreeClassifier = pd.DataFrame()
    df_predict_RandomForestClassifier = pd.DataFrame()
    df_predict_ExtraTreesClassifier = pd.DataFrame()
    df_predict_AdaBoostClassifier = pd.DataFrame()

    df_predict_prob_SGDClassifier = pd.DataFrame()
    df_predict_prob_GaussianNB = pd.DataFrame()
    df_predict_prob_MultinomialNB = pd.DataFrame()
    df_predict_prob_DecisionTreeClassifier = pd.DataFrame()
    df_predict_prob_RandomForestClassifier = pd.DataFrame()
    df_predict_prob_ExtraTreesClassifier = pd.DataFrame()
    df_predict_prob_AdaBoostClassifier = pd.DataFrame()

    for i in range(2, folds+1):
        print("\n\n__________________________________Fold ",i,"__________________________________")
        logging.info("\n\n__________________________________Fold %s__________________________________", i)
        scores, df_predict_SGDClassifier, df_predict_GaussianNB, df_predict_MultinomialNB, df_predict_DecisionTreeClassifier, df_predict_RandomForestClassifier, df_predict_ExtraTreesClassifier, df_predict_AdaBoostClassifier, df_predict_prob_SGDClassifier, df_predict_prob_GaussianNB, df_predict_prob_MultinomialNB, df_predict_prob_DecisionTreeClassifier, df_predict_prob_RandomForestClassifier, df_predict_prob_ExtraTreesClassifier, df_predict_prob_AdaBoostClassifier = ModelSelection(X, Y, i, cores, scores, label, df_predict_SGDClassifier, df_predict_GaussianNB, df_predict_MultinomialNB, df_predict_DecisionTreeClassifier, df_predict_RandomForestClassifier, df_predict_ExtraTreesClassifier, df_predict_AdaBoostClassifier, countkingdom, df_predict_prob_SGDClassifier, df_predict_prob_GaussianNB, df_predict_prob_MultinomialNB, df_predict_prob_DecisionTreeClassifier, df_predict_prob_RandomForestClassifier, df_predict_prob_ExtraTreesClassifier, df_predict_prob_AdaBoostClassifier, label_encoder)

    scores.columns = ["model", "fold", "time_train", "time_test", "score_train", "score_test"]
    nameScores = "scores_"+label+".csv"
    scores.to_csv(nameScores, sep="\t")

    return pd.eval('scores'), pd.eval('df_predict_SGDClassifier'), pd.eval('df_predict_GaussianNB'), pd.eval('df_predict_MultinomialNB'), pd.eval('df_predict_DecisionTreeClassifier'), pd.eval('df_predict_RandomForestClassifier'), pd.eval('df_predict_ExtraTreesClassifier'), pd.eval('df_predict_AdaBoostClassifier'), pd.eval('df_predict_prob_SGDClassifier'), pd.eval('df_predict_prob_GaussianNB'), pd.eval('df_predict_prob_MultinomialNB'), pd.eval('df_predict_prob_DecisionTreeClassifier'), pd.eval('df_predict_prob_RandomForestClassifier'), pd.eval('df_predict_prob_ExtraTreesClassifier'), pd.eval('df_predict_prob_AdaBoostClassifier')


@profile
def ModelSelection(X, Y, folds, cores, df_scoress, label, df_predict_SGDClassifier, df_predict_GaussianNB, df_predict_MultinomialNB, df_predict_DecisionTreeClassifier, df_predict_RandomForestClassifier, df_predict_ExtraTreesClassifier, df_predict_AdaBoostClassifier, countkingdom, df_predict_prob_SGDClassifier, df_predict_prob_GaussianNB, df_predict_prob_MultinomialNB, df_predict_prob_DecisionTreeClassifier, df_predict_prob_RandomForestClassifier, df_predict_prob_ExtraTreesClassifier, df_predict_prob_AdaBoostClassifier, label_encoder):
    

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    skf.get_n_splits(X, Y)



    ############ SGDClassifier ############ 
    model_SGDClassifier = SGDClassifier(n_jobs=cores, loss="log", max_iter=1000, random_state=0)
    print("\nModel:", model_SGDClassifier)
    logging.info("\nModel: %s", model_SGDClassifier)

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal)(train_index, test_index, model_SGDClassifier, folds, X, Y, "SGDClassifier", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]
    
    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)


    df_predict_SGDClassifier[name] = pd.eval('list(aux[name])')
    name = "predict/SGDClassifier_"+label+".csv"
    df_predict_SGDClassifier.to_csv(name, sep="\t")

    #Treatement of probabilitys
    y_predict_probs = [d['y_predict_prob'] for d in out]
    y_predict_probs_aux = [val for sublist in y_predict_probs for val in sublist]

    df_predict_probs = pd.DataFrame(pd.eval('y_predict_probs_aux'))
    df_predict_probs.index = pd.eval('index')
    df_predict_probs.sort_index(inplace=True)
    df_predict_prob_SGDClassifier = pd.concat(pd.eval('[df_predict_prob_SGDClassifier,df_predict_probs]'), axis =1)
    df_predict_prob_SGDClassifier.index = pd.eval('df_predict_probs.index')

    name = "prob/SGDClassifier_"+label+".csv"
    df_predict_prob_SGDClassifier.to_csv(name, sep="\t")

    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")



    ############ GaussianNB ############ 
    model_GaussianNB = GaussianNB()
    print("\nModel:", model_GaussianNB)
    logging.info("\nModel: %s", model_GaussianNB)

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal)(train_index, test_index, model_GaussianNB, folds, X, Y, "GaussianNB", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]
    
    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)


    df_predict_GaussianNB[name] = pd.eval('list(aux[name])')
    name = "predict/GaussianNB_"+label+".csv"
    df_predict_GaussianNB.to_csv(name, sep="\t")

    #Treatement of probabilitys
    y_predict_probs = [d['y_predict_prob'] for d in out]
    y_predict_probs_aux = [val for sublist in y_predict_probs for val in sublist]

    df_predict_probs = pd.DataFrame(pd.eval('y_predict_probs_aux'))
    df_predict_probs.index = pd.eval('index')
    df_predict_probs.sort_index(inplace=True)
    df_predict_prob_GaussianNB = pd.concat(pd.eval('[df_predict_prob_GaussianNB,df_predict_probs]'), axis =1)
    df_predict_prob_GaussianNB.index = pd.eval('df_predict_probs.index')

    name = "prob/GaussianNB_"+label+".csv"
    df_predict_prob_GaussianNB.to_csv(name, sep="\t")

    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")



    ############ MultinomialNB ############ 
    model_MultinomialNB = MultinomialNB()
    print("\nModel:", model_MultinomialNB)
    logging.info("\nModel: %s", model_MultinomialNB)

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal)(train_index, test_index, model_MultinomialNB, folds, X, Y, "MultinomialNB", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]

    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]
    
    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)


    df_predict_MultinomialNB[name] = pd.eval('list(aux[name])')
    name = "predict/MultinomialNB_"+label+".csv"
    df_predict_MultinomialNB.to_csv(name, sep="\t")

    #Treatement of probabilitys
    y_predict_probs = [d['y_predict_prob'] for d in out]
    y_predict_probs_aux = [val for sublist in y_predict_probs for val in sublist]

    df_predict_probs = pd.DataFrame(pd.eval('y_predict_probs_aux'))
    df_predict_probs.index = pd.eval('index')
    df_predict_probs.sort_index(inplace=True)
    df_predict_prob_MultinomialNB = pd.concat(pd.eval('[df_predict_prob_MultinomialNB,df_predict_probs]'), axis =1)
    df_predict_prob_MultinomialNB.index = pd.eval('df_predict_probs.index')

    name = "prob/MultinomialNB_"+label+".csv"
    df_predict_prob_MultinomialNB.to_csv(name, sep="\t")

    #Treatement of scores
    scoress = [d['s'] for d in out]
    #scoress_aux = [val for sublist in scoress for val in sublist]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")



    ############ DecisionTreeClassifier ############ 
    model_DecisionTreeClassifier = DecisionTreeClassifier(random_state=0, max_depth=20, min_samples_split=5)
    print("\nModel:", model_DecisionTreeClassifier)
    logging.info("\nModel: %s", model_DecisionTreeClassifier)

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal)(train_index, test_index, model_DecisionTreeClassifier, folds, X, Y, "DecisionTree", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]
    
    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)

    df_predict_DecisionTreeClassifier[name] = pd.eval('list(aux[name])')
    df_predict_DecisionTreeClassifier.index = pd.eval('aux.index')
    name = "predict/DecisionTree_"+label+".csv"
    df_predict_DecisionTreeClassifier.to_csv(name, sep="\t")

    #Treatement of probabilitys
    y_predict_probs = [d['y_predict_prob'] for d in out]
    y_predict_probs_aux = [val for sublist in y_predict_probs for val in sublist]

    df_predict_probs = pd.DataFrame(pd.eval('y_predict_probs_aux'))
    df_predict_probs.index = pd.eval('index')
    df_predict_probs.sort_index(inplace=True)
    df_predict_prob_DecisionTreeClassifier = pd.concat(pd.eval('[df_predict_prob_DecisionTreeClassifier,df_predict_probs]'), axis =1)
    df_predict_prob_DecisionTreeClassifier.index = pd.eval('df_predict_probs.index')

    name = "prob/DecisionTree_"+label+".csv"
    df_predict_prob_DecisionTreeClassifier.to_csv(name, sep="\t")

    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")



    ############ RandomForestClassifier ############ 
    model_RandomForestClassifier = RandomForestClassifier(n_jobs=cores, n_estimators=100, max_depth=20, min_samples_split=5, random_state=0)
    print("\nModel:", model_RandomForestClassifier)
    logging.info("\nModel: %s", model_RandomForestClassifier)

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal)(train_index, test_index, model_RandomForestClassifier, folds, X, Y, "RandomForest", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]

    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)

    df_predict_RandomForestClassifier[name] = pd.eval('list(aux[name])')
    name = "predict/RandomForest_"+label+".csv"
    df_predict_RandomForestClassifier.to_csv(name, sep="\t")

    #Treatement of probabilitys
    y_predict_probs = [d['y_predict_prob'] for d in out]
    y_predict_probs_aux = [val for sublist in y_predict_probs for val in sublist]

    df_predict_probs = pd.DataFrame(pd.eval('y_predict_probs_aux'))
    df_predict_probs.index = pd.eval('index')
    df_predict_probs.sort_index(inplace=True)
    df_predict_prob_RandomForestClassifier = pd.concat(pd.eval('[df_predict_prob_RandomForestClassifier,df_predict_probs]'), axis =1)
    df_predict_prob_RandomForestClassifier.index = pd.eval('df_predict_probs.index')

    name = "prob/RandomForest_"+label+".csv"
    df_predict_prob_RandomForestClassifier.to_csv(name, sep="\t")

    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")


    ############ ExtraTreesClassifier ############ 
    model_ExtraTreesClassifier = ExtraTreesClassifier(n_jobs=cores, n_estimators=100, max_depth=20, min_samples_split=5, random_state=0)
    print("\nModel:", model_ExtraTreesClassifier)
    logging.info("\nModel: %s", model_ExtraTreesClassifier)    

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal)(train_index, test_index, model_ExtraTreesClassifier, folds, X, Y, "ExtraTrees", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]

    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)

    df_predict_ExtraTreesClassifier[name] = pd.eval('list(aux[name])')
    name = "predict/ExtraTrees_"+label+".csv"
    df_predict_ExtraTreesClassifier.to_csv(name, sep="\t")

    #Treatement of probabilitys
    y_predict_probs = [d['y_predict_prob'] for d in out]
    y_predict_probs_aux = [val for sublist in y_predict_probs for val in sublist]

    df_predict_probs = pd.DataFrame(pd.eval('y_predict_probs_aux'))
    df_predict_probs.index = pd.eval('index')
    df_predict_probs.sort_index(inplace=True)
    df_predict_prob_ExtraTreesClassifier = pd.concat(pd.eval('[df_predict_prob_ExtraTreesClassifier,df_predict_probs]'), axis =1)
    df_predict_prob_ExtraTreesClassifier.index = pd.eval('df_predict_probs.index')

    name = "prob/ExtraTrees_"+label+".csv"
    df_predict_prob_ExtraTreesClassifier.to_csv(name, sep="\t")

    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")


    ############ AdaBoostClassifier ############ 
    model_AdaBoostClassifier = AdaBoostClassifier(n_estimators=100, random_state=0)
    print("\nModel:", model_AdaBoostClassifier)
    logging.info("\nModel: %s", model_AdaBoostClassifier)

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal)(train_index, test_index, model_AdaBoostClassifier, folds, X, Y, "AdaBoost", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]

    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)

    df_predict_AdaBoostClassifier[name] = pd.eval('list(aux[name])')
    name = "predict/AdaBoost_"+label+".csv"
    df_predict_AdaBoostClassifier.to_csv(name, sep="\t")

    #Treatement of probabilitys
    y_predict_probs = [d['y_predict_prob'] for d in out]
    y_predict_probs_aux = [val for sublist in y_predict_probs for val in sublist]

    df_predict_probs = pd.DataFrame(pd.eval('y_predict_probs_aux'))
    df_predict_probs.index = pd.eval('index')
    df_predict_probs.sort_index(inplace=True)
    df_predict_prob_AdaBoostClassifier = pd.concat(pd.eval('[df_predict_prob_AdaBoostClassifier,df_predict_probs]'), axis =1)
    df_predict_prob_AdaBoostClassifier.index = pd.eval('df_predict_probs.index')

    name = "prob/AdaBoost_"+label+".csv"
    df_predict_prob_AdaBoostClassifier.to_csv(name, sep="\t")

    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")


    return pd.eval('df_scoress'), pd.eval('df_predict_SGDClassifier'), pd.eval('df_predict_GaussianNB'), pd.eval('df_predict_MultinomialNB'), pd.eval('df_predict_DecisionTreeClassifier'), pd.eval('df_predict_RandomForestClassifier'), pd.eval('df_predict_ExtraTreesClassifier'), pd.eval('df_predict_AdaBoostClassifier'), pd.eval('df_predict_prob_SGDClassifier'), pd.eval('df_predict_prob_GaussianNB'), pd.eval('df_predict_prob_MultinomialNB'), pd.eval('df_predict_prob_DecisionTreeClassifier'), pd.eval('df_predict_prob_RandomForestClassifier'), pd.eval('df_predict_prob_ExtraTreesClassifier'), pd.eval('df_predict_prob_AdaBoostClassifier')



############ Create Model with Cross Validation just with Trees ############

@profile
def DiferentsFolds_Trees(X, Y, folds, cores, label, countkingdom, label_encoder):
    
    try:
        dir = 'predict'       
        os.makedirs(dir)
        dir = 'prob'       
        os.makedirs(dir)
        
    except OSError:
        pass


    scores = pd.DataFrame()

   
    df_predict_DecisionTreeClassifier = pd.DataFrame()
    df_predict_RandomForestClassifier = pd.DataFrame()
    df_predict_ExtraTreesClassifier = pd.DataFrame()
    
    df_predict_prob_DecisionTreeClassifier = pd.DataFrame()
    df_predict_prob_RandomForestClassifier = pd.DataFrame()
    df_predict_prob_ExtraTreesClassifier = pd.DataFrame()

    for i in range(2, folds+1):
        print("\n\n__________________________________Fold ",i,"__________________________________")
        logging.info("\n\n__________________________________Fold %s__________________________________", i)
        scores, df_predict_DecisionTreeClassifier, df_predict_RandomForestClassifier, df_predict_ExtraTreesClassifier, df_predict_prob_DecisionTreeClassifier, df_predict_prob_RandomForestClassifier, df_predict_prob_ExtraTreesClassifier = ModelSelection_Trees(X, Y, i, cores, scores, label, df_predict_DecisionTreeClassifier, df_predict_RandomForestClassifier, df_predict_ExtraTreesClassifier,  countkingdom, df_predict_prob_DecisionTreeClassifier, df_predict_prob_RandomForestClassifier, df_predict_prob_ExtraTreesClassifier, label_encoder)

    scores.columns = ["model", "fold", "time_train", "time_test", "score_train", "score_test"]
    nameScores = "scores_"+label+".csv"
    scores.to_csv(nameScores, sep="\t")

    return pd.eval('scores'), pd.eval('df_predict_DecisionTreeClassifier'), pd.eval('df_predict_RandomForestClassifier'), pd.eval('df_predict_ExtraTreesClassifier'), pd.eval('df_predict_prob_DecisionTreeClassifier'), pd.eval('df_predict_prob_RandomForestClassifier'), pd.eval('df_predict_prob_ExtraTreesClassifier')


@profile
def ModelSelection_Trees(X, Y, folds, cores, df_scoress, label, df_predict_DecisionTreeClassifier, df_predict_RandomForestClassifier, df_predict_ExtraTreesClassifier, countkingdom, df_predict_prob_DecisionTreeClassifier, df_predict_prob_RandomForestClassifier, df_predict_prob_ExtraTreesClassifier, label_encoder):
    

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    skf.get_n_splits(X, Y)



    ############ DecisionTreeClassifier ############ 
    model_DecisionTreeClassifier = DecisionTreeClassifier(random_state=0, max_depth=20, min_samples_split=5)
    print("\nModel:", model_DecisionTreeClassifier)
    logging.info("\nModel: %s", model_DecisionTreeClassifier)

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal)(train_index, test_index, model_DecisionTreeClassifier, folds, X, Y, "DecisionTree", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]
    
    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    print(y_predicts_inv)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)

    df_predict_DecisionTreeClassifier[name] = pd.eval('list(aux[name])')
    name = "predict/DecisionTree_"+label+".csv"
    df_predict_DecisionTreeClassifier.to_csv(name, sep="\t")

    #Treatement of probabilitys
    y_predict_probs = [d['y_predict_prob'] for d in out]
    y_predict_probs_aux = [val for sublist in y_predict_probs for val in sublist]

    df_predict_probs = pd.DataFrame(pd.eval('y_predict_probs_aux'))
    df_predict_probs.index = pd.eval('index')
    df_predict_probs.sort_index(inplace=True)
    df_predict_prob_DecisionTreeClassifier = pd.concat(pd.eval('[df_predict_prob_DecisionTreeClassifier,df_predict_probs]'), axis =1)
    df_predict_prob_DecisionTreeClassifier.index = pd.eval('df_predict_probs.index')

    #name = "prob/DecisionTree_"+label+".csv"
    #df_predict_prob_DecisionTreeClassifier.to_csv(name, sep="\t")

    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")



    ############ RandomForestClassifier ############ 
    model_RandomForestClassifier = RandomForestClassifier(n_jobs=cores, n_estimators=100, max_depth=20, min_samples_split=5, random_state=0)
    print("\nModel:", model_RandomForestClassifier)
    logging.info("\nModel: %s", model_RandomForestClassifier)

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal)(train_index, test_index, model_RandomForestClassifier, folds, X, Y, "RandomForest", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]

    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)

    df_predict_RandomForestClassifier[name] = pd.eval('list(aux[name])')
    name = "predict/RandomForest_"+label+".csv"
    df_predict_RandomForestClassifier.to_csv(name, sep="\t")

    #Treatement of probabilitys
    y_predict_probs = [d['y_predict_prob'] for d in out]
    y_predict_probs_aux = [val for sublist in y_predict_probs for val in sublist]

    df_predict_probs = pd.DataFrame(pd.eval('y_predict_probs_aux'))
    df_predict_probs.index = pd.eval('index')
    df_predict_probs.sort_index(inplace=True)
    df_predict_prob_RandomForestClassifier = pd.concat(pd.eval('[df_predict_prob_RandomForestClassifier,df_predict_probs]'), axis =1)
    df_predict_prob_RandomForestClassifier.index = pd.eval('df_predict_probs.index')

    #name = "prob/RandomForest_"+label+".csv"
    #df_predict_prob_RandomForestClassifier.to_csv(name, sep="\t")

    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")


    ############ ExtraTreesClassifier ############ 
    model_ExtraTreesClassifier = ExtraTreesClassifier(n_jobs=cores, n_estimators=100, max_depth=20, min_samples_split=5, random_state=0)
    print("\nModel:", model_ExtraTreesClassifier)
    logging.info("\nModel: %s", model_ExtraTreesClassifier)    

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal)(train_index, test_index, model_ExtraTreesClassifier, folds, X, Y, "ExtraTrees", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]

    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)

    df_predict_ExtraTreesClassifier[name] = pd.eval('list(aux[name])')
    name = "predict/ExtraTrees_"+label+".csv"
    df_predict_ExtraTreesClassifier.to_csv(name, sep="\t")

    #Treatement of probabilitys
    y_predict_probs = [d['y_predict_prob'] for d in out]
    y_predict_probs_aux = [val for sublist in y_predict_probs for val in sublist]

    df_predict_probs = pd.DataFrame(pd.eval('y_predict_probs_aux'))
    df_predict_probs.index = pd.eval('index')
    df_predict_probs.sort_index(inplace=True)
    df_predict_prob_ExtraTreesClassifier = pd.concat(pd.eval('[df_predict_prob_ExtraTreesClassifier,df_predict_probs]'), axis =1)
    df_predict_prob_ExtraTreesClassifier.index = pd.eval('df_predict_probs.index')

    #name = "prob/ExtraTrees_"+label+".csv"
    #df_predict_prob_ExtraTreesClassifier.to_csv(name, sep="\t")

    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")


    return pd.eval('df_scoress'), pd.eval('df_predict_DecisionTreeClassifier'), pd.eval('df_predict_RandomForestClassifier'), pd.eval('df_predict_ExtraTreesClassifier'), pd.eval('df_predict_prob_DecisionTreeClassifier'), pd.eval('df_predict_prob_RandomForestClassifier'), pd.eval('df_predict_prob_ExtraTreesClassifier')


@profile
def CrossVal(train_index, test_index, model, folds, X, Y, modelName, countkingdom):  
    

    datasIn_train, datasIn_test = X.loc[train_index], X.loc[test_index]
    datasOut_train, datasOut_test = Y[train_index], Y[test_index]

    y_predict = np.zeros(len(datasOut_test))
    y_predict_prob = np.zeros((len(datasOut_test),countkingdom.shape[0]))

    ############ Train the model ############ 
    time_start_train = time.clock()
    model.fit(datasIn_train, datasOut_train.ravel())
    time_elapsed_train = (time.clock() - time_start_train)

    score_train = model.score(datasIn_test, datasOut_test)

    ############ Predict with the model ############ 
    time_start_test = time.clock()
    y_predict = model.predict(datasIn_test)
    time_elapsed_test = (time.clock() - time_start_test)

    y_predict_prob = model.predict_proba(datasIn_test)

    score_test = model.score(datasIn_train, datasOut_train)

    l = [modelName, folds, time_elapsed_train, time_elapsed_test, score_train, score_test]
    s = pd.Series(l)

    return dict(s = s, y_predict = y_predict, y_predict_prob = y_predict_prob, index = test_index)



############ Create Model with Cross Validation just with Trees ############

@profile
def DiferentsFolds_Trees_Pred(X, Y, folds, cores, label, countkingdom, label_encoder):
    
    try:
        dir = 'predict'       
        os.makedirs(dir)
        
    except OSError:
        pass


    scores = pd.DataFrame()

   
    df_predict_DecisionTreeClassifier = pd.DataFrame()
    df_predict_RandomForestClassifier = pd.DataFrame()
    df_predict_ExtraTreesClassifier = pd.DataFrame()
    

    for i in range(2, folds+1):
        print("\n\n__________________________________Fold ",i,"__________________________________")
        logging.info("\n\n__________________________________Fold %s__________________________________", i)
        scores, df_predict_DecisionTreeClassifier, df_predict_RandomForestClassifier, df_predict_ExtraTreesClassifier = ModelSelection_Trees_Pred(X, Y, i, cores, scores, label, df_predict_DecisionTreeClassifier, df_predict_RandomForestClassifier, df_predict_ExtraTreesClassifier,  countkingdom, label_encoder)

    scores.columns = ["model", "fold", "time_train", "time_test", "score_train", "score_test"]
    nameScores = "scores_"+label+".csv"
    scores.to_csv(nameScores, sep="\t")

    return pd.eval('scores'), pd.eval('df_predict_DecisionTreeClassifier'), pd.eval('df_predict_RandomForestClassifier'), pd.eval('df_predict_ExtraTreesClassifier')


@profile
def ModelSelection_Trees_Pred(X, Y, folds, cores, df_scoress, label, df_predict_DecisionTreeClassifier, df_predict_RandomForestClassifier, df_predict_ExtraTreesClassifier, countkingdom, label_encoder):
    

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    skf.get_n_splits(X, Y)



    ############ DecisionTreeClassifier ############ 
    model_DecisionTreeClassifier = DecisionTreeClassifier(random_state=0, max_depth=20, min_samples_split=5)
    print("\nModel:", model_DecisionTreeClassifier)
    logging.info("\nModel: %s", model_DecisionTreeClassifier)

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal_Pred)(train_index, test_index, model_DecisionTreeClassifier, folds, X, Y, "DecisionTree", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]
    
    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)

    df_predict_DecisionTreeClassifier[name] = pd.eval('list(aux[name])')
    name = "predict/DecisionTree_"+label+".csv"
    df_predict_DecisionTreeClassifier.to_csv(name, sep="\t")


    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")



    ############ RandomForestClassifier ############ 
    model_RandomForestClassifier = RandomForestClassifier(n_jobs=cores, n_estimators=100, max_depth=20, min_samples_split=5, random_state=0)
    print("\nModel:", model_RandomForestClassifier)
    logging.info("\nModel: %s", model_RandomForestClassifier)

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal_Pred)(train_index, test_index, model_RandomForestClassifier, folds, X, Y, "RandomForest", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]

    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)

    df_predict_RandomForestClassifier[name] = pd.eval('list(aux[name])')
    name = "predict/RandomForest_"+label+".csv"
    df_predict_RandomForestClassifier.to_csv(name, sep="\t")


    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")


    ############ ExtraTreesClassifier ############ 
    model_ExtraTreesClassifier = ExtraTreesClassifier(n_jobs=cores, n_estimators=100, max_depth=20, min_samples_split=5, random_state=0)
    print("\nModel:", model_ExtraTreesClassifier)
    logging.info("\nModel: %s", model_ExtraTreesClassifier)    

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal_Pred)(train_index, test_index, model_ExtraTreesClassifier, folds, X, Y, "ExtraTrees", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]

    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)

    df_predict_ExtraTreesClassifier[name] = pd.eval('list(aux[name])')
    name = "predict/ExtraTrees_"+label+".csv"
    df_predict_ExtraTreesClassifier.to_csv(name, sep="\t")

    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")


    return pd.eval('df_scoress'), pd.eval('df_predict_DecisionTreeClassifier'), pd.eval('df_predict_RandomForestClassifier'), pd.eval('df_predict_ExtraTreesClassifier')


############ Create Model with Cross Validation just with Trees ############

@profile
def DiferentsFolds_RF_Pred(X, Y, folds, cores, label, countkingdom, label_encoder):
    
    try:
        dir = 'predict'       
        os.makedirs(dir)
        
    except OSError:
        pass


    scores = pd.DataFrame()

   
    df_predict_RandomForestClassifier = pd.DataFrame()
    

    for i in range(8, folds+1):
        print("\n\n__________________________________Fold ",i,"__________________________________")
        logging.info("\n\n__________________________________Fold %s__________________________________", i)
        scores, df_predict_RandomForestClassifier = ModelSelection_RF_Pred(X, Y, i, cores, scores, label, df_predict_RandomForestClassifier, countkingdom, label_encoder)

    scores.columns = ["model", "fold", "time_train", "time_test", "score_train", "score_test"]
    nameScores = "scores_"+label+".csv"
    scores.to_csv(nameScores, sep="\t")

    return pd.eval('scores'), pd.eval('df_predict_RandomForestClassifier')


@profile
def ModelSelection_RF_Pred(X, Y, folds, cores, df_scoress, label, df_predict_RandomForestClassifier, countkingdom, label_encoder):
    

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    skf.get_n_splits(X, Y)


    ############ RandomForestClassifier ############ 
    model_RandomForestClassifier = RandomForestClassifier(n_jobs=cores, n_estimators=100, max_depth=20, min_samples_split=5, random_state=0)
    print("\nModel:", model_RandomForestClassifier)
    logging.info("\nModel: %s", model_RandomForestClassifier)

    #Run the cross in Paralleled
    out = Parallel(n_jobs=folds, verbose=100, pre_dispatch='all', max_nbytes=None)(delayed(CrossVal_Pred)(train_index, test_index, model_RandomForestClassifier, folds, X, Y, "RandomForest", countkingdom) for train_index, test_index in skf.split(X, Y))

    #Treatement of predicts
    y_predicts = [d['y_predict'] for d in out]
    y_predicts_aux = [val for sublist in y_predicts for val in sublist]
    
    index = [d['index'] for d in out]
    index = [val for sublist in index for val in sublist]

    y_predicts_inv = label_encoder.inverse_transform(y_predicts_aux)
    name = "predict_"+str(folds)

    aux = pd.DataFrame()
    aux[name] = pd.eval('y_predicts_inv')
    aux.index = pd.eval('index')
    aux.sort_index(inplace=True)

    df_predict_RandomForestClassifier[name] = pd.eval('list(aux[name])')
    name = "predict/RandomForest_"+label+".csv"
    df_predict_RandomForestClassifier.to_csv(name, sep="\t")


    #Treatement of scores
    scoress = [d['s'] for d in out]
    df_scoress_aux = pd.DataFrame(pd.eval('scoress'))
    df_scoress = pd.concat(pd.eval('[df_scoress,df_scoress_aux]'))

    nameScores = "scores_"+label+".csv"
    df_scoress.to_csv(nameScores, sep="\t")


    return pd.eval('df_scoress'), pd.eval('df_predict_RandomForestClassifier')


@profile
def CrossVal_Pred(train_index, test_index, model, folds, X, Y, modelName, countkingdom):  
    

    datasIn_train, datasIn_test = X.loc[train_index], X.loc[test_index]
    datasOut_train, datasOut_test = Y[train_index], Y[test_index]

    y_predict = np.zeros(len(datasOut_test))
    y_predict_prob = np.zeros((len(datasOut_test),countkingdom.shape[0]))

    ############ Train the model ############ 
    time_start_train = time.clock()
    model.fit(datasIn_train, datasOut_train.ravel())
    time_elapsed_train = (time.clock() - time_start_train)

    score_train = model.score(datasIn_test, datasOut_test)

    ############ Predict with the model ############ 
    time_start_test = time.clock()
    y_predict = model.predict(datasIn_test)
    time_elapsed_test = (time.clock() - time_start_test)

    score_test = model.score(datasIn_train, datasOut_train)

    l = [modelName, folds, time_elapsed_train, time_elapsed_test, score_train, score_test]
    s = pd.Series(l)

    return dict(s = s, y_predict = y_predict, index = test_index)


############ Create charts ############

def CreateTableCharts(scores, folds, metric):
    dfFit_time = pd.DataFrame()
    dfScore_time = pd.DataFrame()

    for i in range(folds-1):
        lFit_time = []
        i=i+2

        lFit_time = [scores[scores.fold==i][metric].SGDClassifier.mean(), 
                     scores[scores.fold==i][metric].GaussianNB.mean(), 
                     scores[scores.fold==i][metric].MultinomialNB.mean(),
                     scores[scores.fold==i][metric].DecisionTreeClassifier.mean(),
                     scores[scores.fold==i][metric].RandomForestClassifier.mean(),
                     scores[scores.fold==i][metric].ExtraTreesClassifier.mean(), 
                     scores[scores.fold==i][metric].AdaBoostClassifier.mean()]
        
        dfFit_time[i] = lFit_time
        dfFit_time.index = ["SGDClassifier","GaussianNB","MultinomialNB", "DecisionTreeClassifier", 
                           "RandomForestClassifier", "ExtraTreesClassifier","AdaBoostClassifier"]
        
    return dfFit_time

def CreateChart(df_train, df_test, metric):
    plt.figure()
    plt.style.use('ggplot')

    plt.rcParams['figure.figsize'] = (15,12)
    plt.xlabel('Folds')
    plt.ylabel(metric)
    x=[2,3,4,5,6,7,8,9,10]

    plt.title("Model selection")    

    plt.plot(x, df_train.loc["GaussianNB"], color = "#0099ae", marker='o')
    plt.plot(x, df_test.loc["GaussianNB"], color = "#0099ae", marker='v')
    plt.plot(x, df_train.loc["SGDClassifier"], color = "#d25972", marker='o')
    plt.plot(x, df_test.loc["SGDClassifier"], color = "#d25972", marker='v')
    plt.plot(x, df_train.loc["MultinomialNB"], color = "#fdcb72", marker='o')
    plt.plot(x, df_test.loc["MultinomialNB"], color = "#fdcb72", marker='v')
    plt.plot(x, df_train.loc["DecisionTreeClassifier"], color = "#85be22", marker='o')
    plt.plot(x, df_test.loc["DecisionTreeClassifier"], color = "#85be22", marker='v')
    plt.plot(x, df_train.loc["RandomForestClassifier"], color = "#F9A8F5", marker='o')
    plt.plot(x, df_test.loc["RandomForestClassifier"], color = "#F9A8F5", marker='v')
    plt.plot(x, df_train.loc["ExtraTreesClassifier"], color = "#A08D83", marker='o')
    plt.plot(x, df_test.loc["ExtraTreesClassifier"], color = "#A08D83", marker='v')
    plt.plot(x, df_train.loc["AdaBoostClassifier"], color = "#FF8C00", marker='o')
    plt.plot(x, df_test.loc["AdaBoostClassifier"], color = "#FF8C00", marker='v')
    
    
    GaussianNB = mpatches.Patch(color='#0099ae', label='Gaussian NB')
    SGDClassifier = mpatches.Patch(color='#d25972', label='SGD')
    MultinomialNB = mpatches.Patch(color='#fdcb72', label='Multinomial NB')
    DecisionTreeClassifier = mpatches.Patch(color='#85be22', label='Decision Tree')
    RandomForestClassifier = mpatches.Patch(color='#F9A8F5', label='Random Forest')
    ExtraTreesClassifier = mpatches.Patch(color='#A08D83', label='Extra Trees')
    AdaBoostClassifier = mpatches.Patch(color='#FF8C00', label='Ada Boost')

    first_legend = plt.legend(handles=[GaussianNB, SGDClassifier, MultinomialNB, DecisionTreeClassifier, RandomForestClassifier,
                       ExtraTreesClassifier, AdaBoostClassifier], loc='best')

    ax = plt.gca().add_artist(first_legend)
    
    train = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=8, label='Train')
    test = mlines.Line2D([], [], color='black', marker='v', linestyle='None',
                          markersize=8, label='Test')
    
    plt.legend(handles=[train, test], loc=1)
    if(metric!="time"):
        plt.ylim(-0.05, 1.05)
    
    plt.show()
    name = metric+".png"
    plt.savefig(name, dpi=300, type="png")
