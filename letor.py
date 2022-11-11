import numpy as np
import pandas as pd
import pickle
import rank_metric as metrics
import spacy

from models import VectorSpaceModel, LMJM
from typing import Type
from data_storage import ClinicalTrials, Query
from os.path import exists
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, average_precision_score
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import accuracy_score, ndcg_score
from sklearn.tree import DecisionTreeClassifier


def getAP(predictions, prob_predictions, relevance):
    predictions = np.array(predictions)
    prob_predictions = np.array(prob_predictions)
    relevance = np.array(relevance)
    
    rel_pred = predictions * relevance
    n = len(predictions)
    top_indices = np.argpartition(prob_predictions, -n)[-n:]
    top_indices = np.flip(top_indices[np.argsort(prob_predictions[top_indices])])
    top_rel_pred = rel_pred[top_indices]
    number_of_relevant_docs = sum([r > 0 for r in relevance])
    
    ap = metrics.average_precision(top_rel_pred, number_of_relevant_docs)
    return ap

class LETOR:
    
    def __init__(self, dataset, docs, C = 1, class_weight = None,
                 sample_weight = None, max_d = None):
        self.trials = docs
        #Scalers are a list of tuples (mean, std)
        self.scalers = []
        
        #Data standardization
        for col in ['s11', 's12', 's21', 's22', 's31', 's32', 's41', 's42']:
            self.scalers.append((dataset[col].mean(), dataset[col].std()))
            dataset[col] = (dataset[col] - dataset[col].mean()) / dataset[col].std()
            
        #self.classifier = LogisticRegression(C = C, class_weight = class_weight,
        #                                     n_jobs = -1, random_state = 2137,
        #                                     max_iter = 200)
        
        self.classifier = DecisionTreeClassifier(max_depth = max_d, random_state = 420,
                                                 class_weight=class_weight)
        
        #Transform the dataset to numpy array
        dataset_numpy = dataset.to_numpy()
        self.x = dataset_numpy[:, 0:8]
        self.x = np.array(self.x, dtype = 'float64')
        
        self.relevance = dataset_numpy[:, 8]
        self.relevance = np.array(self.relevance, dtype = 'int32')
        
        if sample_weight == None:
            self.sample_weight = np.ones(len(self.relevance))
            self.y = np.array([y > 0 for y in self.relevance], dtype='int32')
            
        elif sample_weight == 'relevance':
            self.sample_weight = np.array([1 if y < 2 else 2 for y in self.relevance])
            self.y = np.array([y > 0 for y in self.relevance], dtype='int32')
    
    
    def fitModel(self):
        self.classifier.fit(self.x, self.y,
                            sample_weight=self.sample_weight)
    
    
    def get_top_query_results(self, query, gender_filter = True,
                              age_filter = True):
        
        file_data = open('vsm_models.obj', 'rb') 
        VSM_models = pickle.load(file_data)
        file_data.close()
    
        file_data = open('lmjm_models.obj', 'rb') 
        LMJM_models = pickle.load(file_data)
        file_data.close()
        
        #weights = self.classifier.coef_[0]
        weights = self.classifier.feature_importances_
        
        #Obtain scores from different models
        vsm_1_s = VSM_models[0].get_query_scores(query.query_content)
        vsm_2_s = VSM_models[1].get_query_scores(query.query_content)
        vsm_3_s = VSM_models[2].get_query_scores(query.query_content)
        vsm_4_s = VSM_models[3].get_query_scores(query.query_content)
        
        lmjm_1_s = LMJM_models[0].get_query_scores(query.query_content)
        lmjm_2_s = LMJM_models[1].get_query_scores(query.query_content)
        lmjm_3_s = LMJM_models[2].get_query_scores(query.query_content)
        lmjm_4_s = LMJM_models[3].get_query_scores(query.query_content)
        
        #Standardize the scores and mutiply by weights
        vsm_1_s = weights[0]*(vsm_1_s - self.scalers[0][0])/self.scalers[0][1]
        lmjm_1_s = weights[1]*(lmjm_1_s - self.scalers[1][0])/self.scalers[1][1]
        
        vsm_2_s = weights[2]*(vsm_2_s - self.scalers[2][0])/self.scalers[2][1]
        lmjm_2_s = weights[3]*(lmjm_2_s - self.scalers[3][0])/self.scalers[3][1]
        
        vsm_3_s = weights[4]*(vsm_3_s - self.scalers[4][0])/self.scalers[4][1]
        lmjm_3_s = weights[5]*(lmjm_3_s - self.scalers[5][0])/self.scalers[5][1]
        
        vsm_4_s = weights[6]*(vsm_4_s - self.scalers[6][0])/self.scalers[6][1]
        lmjm_4_s = weights[7]*(lmjm_4_s - self.scalers[7][0])/self.scalers[7][1]

        
        #Obtain final scores
        s = vsm_1_s + vsm_2_s + vsm_3_s + vsm_4_s + lmjm_1_s + lmjm_2_s + lmjm_3_s + lmjm_4_s
        s = np.array(s)
        
        #Find indices of top scores, sorted
        n = len(s)
        top_indices = np.argpartition(s, -n)[-n:]
        top_indices = np.flip(top_indices[np.argsort(s[top_indices])])
        
        #Find the ids of these clinical trials
        top_ids = VSM_models[0].id_list[top_indices]
        
        ids_to_filter = []
        
        #Filter the results by gender
        if gender_filter:
            nlp = spacy.load("en_core_web_sm")
            
            #Vocab lists for identifying gender
            male_words = ["male", "males", "boy", "boys", "guy", "guys", "man",
                          "men", "gentleman", "gentlemen", "sir"]
            female_words = ["female", "females", "girl", "girls", "woman", "women",
                            "lady", "ladies", "miss", "madam", "gentlewoman",
                            "gentlewomen"]
            
            male_vocab = spacy.vocab.Vocab(strings = male_words)
            female_vocab = spacy.vocab.Vocab(strings = female_words)
            
            query_nlp = nlp(query.query_content)
            
            male_matches = sum([token.lemma_ in male_vocab for token
                                in query_nlp])
            
            female_matches = sum([token.lemma_ in female_vocab for token
                                in query_nlp])
            
            #logical evaluation of the gender
            if (male_matches > 0):
                if (female_matches == 0):
                    gender = "Male"
                else:
                    gender = None
            else:
                if (female_matches > 0):
                    gender = "Female"
                else:
                    gender = None
            
            #Filter out the docs with non-matching gender           
            for idx, doc_id in enumerate(top_ids):
                if (gender != None and self.trials.gender[doc_id] != "Both"):
                    if self.trials.gender[doc_id] != gender:
                        ids_to_filter.append(idx)

            
        #Filter the results by age   
        if age_filter:
            age = None
            nlp = spacy.load("en_core_web_sm")
            query_nlp = nlp(query.query_content)   
            
            #Vocab for identifying if DATE refers to the age
            age_words_year = ["year", "years", "yo"]
            age_vocab_year = spacy.vocab.Vocab(strings = age_words_year)
            
            age_words_month = ["month", "months", "mo"]
            age_vocab_month = spacy.vocab.Vocab(strings = age_words_month)
            
            #Look for age in the query content
            for ent in query_nlp.ents:
                if (ent.label_ == "DATE"):
                    if ent.text.isnumeric():
                        age = int(ent.text)
                        break
                    else:
                        if sum([token.text in age_vocab_year for token
                                in nlp(ent.text)]) > 0:
                            for token in nlp(ent.text):
                                if token.text.isnumeric():
                                    age = int(token.text)
                                    break
                                
                        elif sum([token.text in age_vocab_month for token
                                in nlp(ent.text)]) > 0:
                            for token in nlp(ent.text):
                                if token.text.isnumeric():
                                    age = int(token.text)/12
                                    break
                        
            #If the age was found            
            if age != None:
                
                for idx, doc_id in enumerate(top_ids):
                    
                    if ((self.trials.min_age[doc_id] not in [None, "N/A"]) and (self.trials.max_age[doc_id] not in [None, "N/A"])):
                        
                        if self.trials.min_age[doc_id].split()[1] in ["Years",
                                                                      "Year"]:
                            min_age = int(self.trials.min_age[doc_id].split()[0])
                        else:
                            min_age = int(self.trials.min_age[doc_id].split()[0])/12
                        
                        
                        if self.trials.max_age[doc_id].split()[1] in ["Years",
                                                                      "Year"]:
                            max_age = int(self.trials.max_age[doc_id].split()[0])
                        else:
                            max_age = int(self.trials.max_age[doc_id].split()[0])/12
                        
                        
                        if (age < min_age or age > max_age):
                                ids_to_filter.append(idx)
                                
                        
                    elif (self.trials.min_age[doc_id] in [None, "N/A"] and 
                          self.trials.max_age[doc_id] not in [None, "N/A"]):
                        
                        if self.trials.max_age[doc_id].split()[1] in ["Years",
                                                                      "Year"]:
                            max_age = int(self.trials.max_age[doc_id].split()[0])
                        else:
                            max_age = int(self.trials.max_age[doc_id].split()[0])/12
                        
                        if age > max_age:
                            ids_to_filter.append(idx)
                            
                    
                    elif (self.trials.min_age[doc_id] not in [None, "N/A"] and 
                          self.trials.max_age[doc_id] in [None, "N/A"]):
                        
                        if self.trials.min_age[doc_id].split()[1] in ["Years",
                                                                      "Year"]:
                            min_age = int(self.trials.min_age[doc_id].split()[0])
                        else:
                            min_age = int(self.trials.min_age[doc_id].split()[0])/12
                        
                        if age < min_age:
                            ids_to_filter.append(idx)
        
        ids_to_filter = sorted(ids_to_filter)
        ids_deleted = top_ids[ids_to_filter]
        indexes = np.unique(ids_deleted, return_index=True)[1]
        ids_deleted = [ids_deleted[index] for index in sorted(indexes)]
        
        top_ids = np.delete(top_ids, ids_to_filter)
        top_ids = np.append(top_ids, ids_deleted)
                               
        return top_ids
        
    def obtainCVmetrics(self):
        
        CVmetrics = np.zeros((5, 2))
        CVconfusionMatrix_train = np.zeros((2, 2))
        CVconfusionMatrix_valid = np.zeros((2, 2))
        
        kf = StratifiedKFold(5, shuffle=True, random_state=213)
        
        for idx_train, idx_valid in kf.split(self.relevance, self.relevance):
            k_train_x = self.x[idx_train]
            k_train_y = self.y[idx_train]
            
            valid_x = self.x[idx_valid]
            valid_y = self.y[idx_valid]
            
            self.classifier.fit(k_train_x, k_train_y,
                                sample_weight=self.sample_weight[idx_train])
            
            pred_train = self.classifier.predict(k_train_x)
            pred_train_prob = self.classifier.predict_proba(k_train_x)
            
            pred_valid = self.classifier.predict(valid_x)
            pred_valid_prob = self.classifier.predict_proba(valid_x)
            
            #Add precision to metrics
            CVmetrics[0, 0] += precision_score(k_train_y, pred_train)
            CVmetrics[0, 1] += precision_score(valid_y, pred_valid)
            
            #Add AP to metrics
            CVmetrics[1, 0] += getAP(pred_train,
                                     pred_train_prob[:, 1],
                                     self.relevance[idx_train])
            CVmetrics[1, 1] += getAP(pred_valid,
                                     pred_valid_prob[:, 1],
                                     self.relevance[idx_valid])
            
            #Add recall to metrics
            CVmetrics[2, 0] += recall_score(k_train_y, pred_train)
            CVmetrics[2, 1] += recall_score(valid_y, pred_valid)


            #Add accuracy to metrics
            CVmetrics[3, 0] += accuracy_score(k_train_y, pred_train)
            CVmetrics[3, 1] += accuracy_score(valid_y, pred_valid)
            
            #Add NDCG5 to metrics
            n_relevant_train = sum([r > 0 for r in self.relevance[idx_train]])
            n_relevant_valid = sum([r > 0 for r in self.relevance[idx_valid]])
            
            CVmetrics[4, 0] += ndcg_score([self.relevance[idx_train]],
                                          [pred_train_prob[:, 1]], k=n_relevant_train)
            CVmetrics[4, 1] += ndcg_score([self.relevance[idx_valid]],
                                          [pred_valid_prob[:, 1]], k=n_relevant_valid)
            
            
            #Add confusion matrix
            CVconfusionMatrix_train += confusion_matrix(k_train_y, pred_train)
            CVconfusionMatrix_valid += confusion_matrix(valid_y, pred_valid)
            
        CVmetrics /= 5
        
        return (CVmetrics, CVconfusionMatrix_train, CVconfusionMatrix_valid)


def generateAndSaveDataset(queries: list[Type[Query]],
                           docs: Type[ClinicalTrials]):
    
    dataset = pd.DataFrame(data=None, columns = ['s11', 's12', 's21', 's22',
                                                 's31', 's32', 's41', 's42',
                                                 'relevance'])
    
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
        LMJM2 = LMJM(docs, arg_lambda = 0.1, lemmatize = True,
                                lowercase = True, stopwords=None,
                                section = 'description')
    
        #Models for brief summary section
        VSM3 = VectorSpaceModel(docs,lemmatize = True, lowercase = True,
                                stopwords='english', section = 'brief_summary')
        LMJM3 = LMJM(docs, arg_lambda = 0.26, lemmatize = True,
                                lowercase = True, stopwords=None,
                                section = 'brief_summary')
    
        #Models for incl criteria section
        VSM4 = VectorSpaceModel(docs,lemmatize = True, lowercase = True,
                                stopwords='english', section = 'criteria')
        LMJM4 = LMJM(docs, arg_lambda = 0.08, lemmatize = True,
                                lowercase = True, stopwords=None,
                                section = 'criteria')
    
    
        VSM_models = [VSM1, VSM2, VSM3, VSM4]
        LMJM_models = [LMJM1, LMJM2, LMJM3, LMJM4]
        
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

        df_vsm = pd.DataFrame(vsm_scores, columns = ['s11', 's21', 's31',
                                                     's41'])
        df_lmjm = pd.DataFrame(lmjm_scores, columns = ['s12', 's22', 's32',
                                                       's42'])
        #Combine results from vsm and lmjm models
        df_combined = pd.concat([df_vsm, df_lmjm], axis=1)
        
        
        #Get relevance
        gt_series = pd.Series([int(query.ground_truth[doc_id]) if 
                               doc_id in query.ground_truth
                               else "NR" for doc_id in docs.ids])
        #Add relevance to dataFrame
        df_combined.insert(len(df_combined.columns), 'relevance', gt_series)        
        
        #Add index to df_combined
        query_series = pd.Series([query.query_id for i in range(0, len(docs.ids))])
        doc_series = pd.Series(docs.ids)
        df_combined = df_combined.set_index([query_series, doc_series])
        
        #Combine with dataset dataframe
        dataset = pd.concat([dataset, df_combined])   
    
    dataset = dataset[dataset['relevance'] != "NR"]

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
      