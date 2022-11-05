import numpy as np
import pandas as pd
import pickle
from models import VectorSpaceModel, LMJM
from typing import Type
from data_storage import ClinicalTrials, Query
from os.path import exists
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, average_precision_score
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

class LETOR:
    
    def __init__(self, dataset, C = 1, class_weight = None, poly_expand = 1,
                 sample_weight = None):
        #Scalers are a list of tuples (mean, std)
        self.scalers = []
        
        #Data standardization
        for col in ['s11', 's12', 's21', 's22', 's31', 's32', 's41', 's42',
                    's51', 's52']:
            self.scalers.append((dataset[col].mean(), dataset[col].std()))
            dataset[col] = (dataset[col] - dataset[col].mean()) / dataset[col].std()
            
        self.classifier = LogisticRegression(C = C, class_weight = class_weight,
                                             n_jobs = -1, random_state = 2137)
        #Transform the dataset to numpy array
        dataset_numpy = dataset.to_numpy()
        self.x = dataset_numpy[:, 0:10]
        self.x = np.array(self.x, dtype = 'float64')
        
        if poly_expand > 1:
            poly = PolynomialFeatures(poly_expand)
            self.x = poly.fit_transform(self.x)
        
        self.y = dataset_numpy[:, 10]
        self.y = np.array(self.y, dtype = 'int32')
        
        if sample_weight == None:
            self.sample_weight = None
            
        else:
            self.sample_weight = self.y * sample_weight + 1
        
    def obtainCVmetrics(self):
        
        CVmetrics = np.zeros((4, 2))
        CVconfusionMatrix_train = np.zeros((2, 2))
        CVconfusionMatrix_valid = np.zeros((2, 2))
        
        kf = StratifiedKFold(5, shuffle=True, random_state=2137)
        
        for idx_train, idx_valid in kf.split(self.y, self.y):
            k_train_x = self.x[idx_train]
            k_train_y = self.y[idx_train]
            
            valid_x = self.x[idx_valid]
            valid_y = self.y[idx_valid]
            
            self.classifier.fit(k_train_x, k_train_y,
                                sample_weight=None)
            
            pred_train = self.classifier.predict(k_train_x)
            pred_train_prob = self.classifier.predict_proba(k_train_x)
            
            pred_valid = self.classifier.predict(valid_x)
            pred_valid_prob = self.classifier.predict_proba(valid_x)
            
            #Add precision to metrics
            CVmetrics[0, 0] += precision_score(k_train_y, pred_train)
            CVmetrics[0, 1] += precision_score(valid_y, pred_valid)
            
            #Add AP to metrics
            CVmetrics[1, 0] += average_precision_score(k_train_y,
                                                       pred_train_prob[:, 1],
                                                       average = None)
            CVmetrics[1, 1] += average_precision_score(valid_y,
                                                       pred_valid_prob[:, 1],
                                                       average = None)
            
            #Add recall to metrics
            CVmetrics[2, 0] += recall_score(k_train_y, pred_train)
            CVmetrics[2, 1] += recall_score(valid_y, pred_valid)


            #Add accuracy to metrics
            CVmetrics[3, 0] += accuracy_score(k_train_y, pred_train)
            CVmetrics[3, 1] += accuracy_score(valid_y, pred_valid)
            
            #Add confusion matrix
            CVconfusionMatrix_train += confusion_matrix(k_train_y, pred_train)
            CVconfusionMatrix_valid += confusion_matrix(valid_y, pred_valid)
            
        CVmetrics /= 5
        
        return (CVmetrics, CVconfusionMatrix_train, CVconfusionMatrix_valid)

def generateAndSaveDataset(queries: list[Type[Query]],
                           docs: Type[ClinicalTrials]):
    
    dataset = pd.DataFrame(data=None, columns = ['s11', 's12', 's21', 's22',
                                                 's31', 's32', 's41', 's42',
                                                 's51', 's52', 'relevance'])
    
    if (exists('vsm_models.obj') and exists('lmjm_models.obj')):    
        #load the objects from pickle
        file_data = open('vsm_models.obj', 'rb') 
        VSM_models = pickle.load(file_data)
        file_data.close()
    
        file_data = open('lmjm_models.obj', 'rb') 
        LMJM_models = pickle.load(file_data)
        file_data.close()

    else:    
        #Models for brief title section
        VSM1 = VectorSpaceModel(docs,lemmatize = True, lowercase = True,
                                stopwords='english', section = 'brief_title')
        LMJM1 = LMJM(docs, arg_lambda = 0.32, lemmatize = True,
                                lowercase = True, stopwords='english',
                                section = 'brief_title')
    
        #Models for description section
        VSM2 = VectorSpaceModel(docs,lemmatize = True, lowercase = True,
                                stopwords='english', section = 'description')
        LMJM2 = LMJM(docs, arg_lambda = 0.36, lemmatize = False,
                                lowercase = True, stopwords=None,
                                section = 'description')
    
        #Models for brief summary section
        VSM3 = VectorSpaceModel(docs,lemmatize = True, lowercase = True,
                                stopwords='english', section = 'brief_summary')
        LMJM3 = LMJM(docs, arg_lambda = 0.26, lemmatize = True,
                                lowercase = True, stopwords='english',
                                section = 'brief_summary')
    
        #Models for incl criteria section
        VSM4 = VectorSpaceModel(docs,lemmatize = True, lowercase = True,
                                stopwords='english', section = 'incl_criteria')
        LMJM4 = LMJM(docs, arg_lambda = 0.18, lemmatize = True,
                                lowercase = True, stopwords='english',
                                section = 'incl_criteria')
    
        #Models for excl criteria section
        VSM5 = VectorSpaceModel(docs,lemmatize = True, lowercase = True,
                                stopwords='english', section = 'excl_criteria')
        LMJM5 = LMJM(docs, arg_lambda = 0.18, lemmatize = True,
                                lowercase = True, stopwords='english',
                                section = 'excl_criteria')
    
        VSM_models = [VSM1, VSM2, VSM3, VSM4, VSM5]
        LMJM_models = [LMJM1, LMJM2, LMJM3, LMJM4, LMJM5]
        
        file_data = open('vsm_models.obj', 'wb') 
        pickle.dump(VSM_models, file_data)
        file_data.close()

        file_data = open('lmjm_models.obj', 'wb') 
        pickle.dump(LMJM_models, file_data)
        file_data.close()


    for query in queries:
        vsm_scores = np.array([vsm.get_query_scores(query.query_content) for vsm
                      in VSM_models])
        vsm_scores = vsm_scores.transpose()
        
        lmjm_scores = np.array([lmjm.get_query_scores(query.query_content) for lmjm
                      in LMJM_models])
        lmjm_scores = lmjm_scores.transpose()

        df_vsm = pd.DataFrame(vsm_scores, columns = ['s11', 's21', 's31', 's41',
                                                     's51'])
        df_lmjm = pd.DataFrame(lmjm_scores, columns = ['s12', 's22', 's32',
                                                       's42','s52'])
        #Combine results from vsm and lmjm models
        df_combined = pd.concat([df_vsm, df_lmjm], axis=1)
        
        
        #Binraize relevance
        gt_series = pd.Series([int(query.ground_truth[doc_id]) > 0 if 
                               doc_id in query.ground_truth
                               else 0 for doc_id in docs.ids], dtype=int)
        #Add ground truth
        df_combined.insert(len(df_combined.columns), 'relevance', gt_series)        
        
        #Add index to df_combined
        query_series = pd.Series([query.query_id for i in range(0, len(docs.ids))])
        doc_series = pd.Series(docs.ids)
        df_combined = df_combined.set_index([query_series, doc_series])
        
        #Combine with dataset dataframe
        dataset = pd.concat([dataset, df_combined])
    
    
    #Handle NaNs - change them to min col value
    min_values = dataset.min()
    dataset = dataset.fillna({'s12': min_values['s12'],
                              's22': min_values['s22'],
                              's32': min_values['s32'],
                              's42': min_values['s42'],
                              's52': min_values['s52']})

    #Save to file
    file_data = open('features.obj', 'wb') 
    pickle.dump(dataset, file_data)
    file_data.close()
    
    
def loadDataset():
    
    if exists('features.obj'):    
        #load the objects from pickle
        file_data = open('features.obj', 'rb') 
        dataset = pickle.load(file_data)
        file_data.close()
        
        return dataset
    
    else:
        print("Dataset file does not exist. Run generateAndSaveDataset() first")
      