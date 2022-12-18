import numpy as np
import pandas as pd
import pickle
import torch
import spacy

from data_storage import ClinicalTrials, Query
import data_loading as d_load
from metrics import Evaluate
from os.path import exists
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from transformers import *
from transformers import BertTokenizer, BertModel
from typing import Type

class BERT_LETOR:
    def __init__(self, train_dataset, test_dataset, docs, C = 1,
                 class_weight = None, sample_weight = None):
        self.trials = docs
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        #Extract training embeddings
        train_emb = train_dataset['embedding'].to_numpy()
        train_emb = np.array([np.array(emb) for emb in train_emb])
        
        #Extract test embeddings
        test_emb = test_dataset['embedding'].to_numpy()
        test_emb = np.array([np.array(emb) for emb in test_emb])
        
        #self.scaler = StandardScaler().fit(train_emb)
        
        #self.train_x = self.scaler.transform(train_emb)
        self.train_x = normalize(train_emb)
        self.train_y = train_dataset['relevance'].to_numpy().astype('int')
        
        #self.test_x = self.scaler.transform(test_emb)
        self.test_x = normalize(test_emb)
        self.test_y = test_dataset['relevance'].to_numpy().astype('int')
        
        self.classifier = LogisticRegression(C = C, class_weight = class_weight,
                                             n_jobs = -1, random_state = 2137,
                                             max_iter = 500)
        
        if sample_weight == None:
            self.sample_weight = np.ones(len(self.train_y))
            self.train_y = np.array([y > 0 for y in self.train_y], dtype='int32')
            
        elif sample_weight == 'relevance':
            self.sample_weight = np.array([1 if y < 2 else 2 for y in self.train_y])
            self.train_y = np.array([y > 0 for y in self.train_y], dtype='int32')
    
    
    def fitModel(self):
        self.classifier = self.classifier.fit(self.train_x, self.train_y,
                                              sample_weight=self.sample_weight)
    
    
    def get_top_query_results(self, query, test_set = True, gender_filter = True,
                              age_filter = True):
        
        if test_set:
            dataset = self.test_dataset
        else:
            dataset = self.train_dataset
            
        query_id = query.query_id
        dataset = dataset.query('query_id == @query_id')
        
        embeddings = dataset['embedding'].to_numpy()
        embeddings = np.array([np.array(emb) for emb in embeddings])
        
        scaled = normalize(embeddings)
        #scaled = self.scaler.transform(embeddings)
        s = scaled.dot(np.array(self.classifier.coef_[0]))
        
        doc_ids = dataset['doc_id'].to_numpy()
        
        #Find indices of top scores, sorted
        n = len(s)
        top_indices = np.argpartition(s, -n)[-n:]
        top_indices = np.flip(top_indices[np.argsort(s[top_indices])])
        
        #Find the ids of these clinical trials
        top_ids = doc_ids[top_indices]
        
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
    
    
def get_CV_metrics(bert_letor_model, train_queries, sample_weight):        
    query_ids = list(set(bert_letor_model.train_dataset['query_id'].to_numpy()))
    query_ids = np.array(query_ids)
    
    kf = KFold(5, shuffle=True, random_state=2137)
    
    valid_metrics = np.zeros(4)
    
    for train_id_idx, valid_id_idx in kf.split(query_ids):           
        train_query_ids = query_ids[train_id_idx]
        valid_query_ids = query_ids[valid_id_idx]
        
        k_train_dataset = bert_letor_model.train_dataset.query('query_id in @train_query_ids')
        k_valid_dataset = bert_letor_model.train_dataset.query('query_id in @valid_query_ids')
        
        k_train_emb = k_train_dataset['embedding'].to_numpy()
        k_train_emb = np.array([np.array(emb) for emb in k_train_emb])
        k_train_emb = normalize(k_train_emb)
        #k_train_emb = bert_letor_model.scaler.transform(k_train_emb)
        
        
        k_train_y = np.array(k_train_dataset['relevance'])
        k_train_y = k_train_y.astype('int')
        
        
        if sample_weight == None:
            sample_weights = np.ones(len(k_train_y))
            k_train_y = np.array([y > 0 for y in k_train_y], dtype='int32')
            
        elif sample_weight == 'relevance':
            sample_weights = np.array([1 if y < 2 else 2 for y in k_train_y])
            k_train_y = np.array([y > 0 for y in k_train_y], dtype='int32')

        classifier = bert_letor_model.classifier.fit(k_train_emb, k_train_y,
                                                     sample_weight = sample_weights)
        
        
        df_metrics = pd.DataFrame(data=None, columns = ['P@10', 'recall', 'AP', 'NDCG5', 'MRR', 
                                               '11-point-precision', '11-point-recall'])
        
        for valid_query_id in valid_query_ids:
            
            query = None
            for q in train_queries:
                if q.query_id == valid_query_id:
                    query = q
                    break
            
            dataset = k_valid_dataset.query('query_id == @valid_query_id')
            
            embeddings = dataset['embedding'].to_numpy()
            embeddings = np.array([np.array(emb) for emb in embeddings])
            scaled = normalize(embeddings)
            
            #scaled = bert_letor_model.scaler.transform(embeddings)
            s = scaled.dot(np.array(classifier.coef_[0]))
            
            doc_ids = dataset['doc_id'].to_numpy()
            
            #Find indices of top scores, sorted
            n = len(s)
            top_indices = np.argpartition(s, -n)[-n:]
            top_indices = np.flip(top_indices[np.argsort(s[top_indices])])
            
            #Find the ids of these clinical trials
            search_results = doc_ids[top_indices]
            
            eval = Evaluate(query, search_results)
            
            df_metrics.loc[query.query_id] = [eval.p10, eval.recall, eval.ap, eval.ndcg5,
                                      eval.mrr, eval.precision_11point, eval.recall_11point]
            
        p_10 = df_metrics['P@10'].mean()
        MAP = df_metrics['AP'].mean()
        ndcg_5 = df_metrics['NDCG5'].mean()
        recall_100 = df_metrics['recall'].mean()
        
        valid_metrics += [p_10, MAP, ndcg_5, recall_100]
        
    return valid_metrics/5    
    

def generateAndSaveDataset():
    #Create empty dataframes for storing embeddings
    train_dataset = pd.DataFrame(data=None, columns = ['query_id', 'doc_id',
                                                       'embedding', 'relevance'])
    
    test_dataset = pd.DataFrame(data=None, columns = ['query_id', 'doc_id',
                                                      'embedding', 'relevance'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Definitions for BERT
    model_path = 'dmis-lab/biobert-v1.1'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path,  output_hidden_states=True, output_attentions=True)  
    model = AutoModel.from_pretrained(model_path, config=config)
    
    #Load trials and patients data
    if (exists('train_data.obj') and 
    exists('test_data.obj') and 
    exists('trials_data.obj')):
    
        #load the objects from pickle
        file_data = open('train_data.obj', 'rb') 
        train = pickle.load(file_data)
        file_data.close()
    
        file_data = open('test_data.obj', 'rb') 
        test = pickle.load(file_data)
        file_data.close()
    
        file_data = open('trials_data.obj', 'rb') 
        trials = pickle.load(file_data)
        file_data.close()

    else:
    
        train, test = d_load.load_and_split_queries()
        trials = d_load.load_evaluated_trials()

        #Save the data objects to files
        file_data = open('train_data.obj', 'wb') 
        pickle.dump(train, file_data)
        file_data.close()

        file_data = open('test_data.obj', 'wb') 
        pickle.dump(test, file_data)
        file_data.close()

        file_data = open('trials_data.obj', 'wb') 
        pickle.dump(trials, file_data)
        file_data.close()
    
    
    train_relevance = np.array([])
    train_pairs = np.array([])
    train_q_ids = np.array([])
    train_doc_ids = np.array([])
    
    #Fill the training dataset
    for query in train:
        for doc in trials.ids:                      
            if doc in query.ground_truth:
                train_relevance = np.append(train_relevance, int(query.ground_truth[doc]))
            else:
                train_relevance = np.append(train_relevance, 0)
                
            train_pairs = np.append(train_pairs, (query.query_content, trials.description[doc]))
            train_q_ids = np.append(train_q_ids, query.query_id)
            train_doc_ids = np.append(train_doc_ids, doc)
            
    embeddings_persistent = np.zeros((len(train_pairs), 768))
    
    for batch_idx in range(0, len(train_pairs), 32):

        # Get the current batch of samples
        batch_data = train_pairs[batch_idx:batch_idx + 32]

        inputs = tokenizer.batch_encode_plus(batch_data, 
                                       return_tensors='pt',  # pytorch tensors
                                       add_special_tokens=True,  # Add CLS and SEP tokens
                                       max_length = 512, # Max sequence length
                                       truncation = True, # Truncate if sequences exceed the Max Sequence length
                                       padding = True) # Add padding to forward sequences with different lengths
        
        # Forward the batch of (query, doc) sequences
        with torch.no_grad():
            inputs.to(device)
            outputs = model(**inputs)

        # Get the CLS embeddings for each pair query, document
        batch_cls = outputs['hidden_states'][-1][:,0,:]
        
        # Store the extracted CLS embeddings from the batch on the memory-mapped ndarray
        embeddings_persistent[batch_idx:batch_idx + 32] = batch_cls.cpu()
        
        print("Batch %d out of %0.1f done." % (batch_idx, len(train_pairs)/32))
        
    train_dataset['query_id'] = train_q_ids
    train_dataset['doc_id'] = train_doc_ids
    train_dataset['embedding'] = embeddings_persistent
    train_dataset['relevance'] = train_relevance
    
    file_data = open('train_dataset.obj', 'wb') 
    pickle.dump(train_dataset, file_data)
    file_data.close()
    
    
    test_relevance = np.array([])
    test_pairs = np.array([])
    test_q_ids = np.array([])
    test_doc_ids = np.array([])
    
    #Fill the test dataset
    for query in test:
        for doc in trials.ids:                      
            if doc in query.ground_truth:
                test_relevance = np.append(test_relevance, int(query.ground_truth[doc]))
            else:
                test_relevance = np.append(test_relevance, 0)
                
            test_pairs = np.append(test_pairs, (query.query_content, trials.description[doc]))
            test_q_ids = np.append(test_q_ids, query.query_id)
            test_doc_ids = np.append(test_doc_ids, doc)
            
    embeddings_persistent = np.zeros((len(test_pairs), 768))
    
    for batch_idx in range(0, len(test_pairs), 32):

        # Get the current batch of samples
        batch_data = test_pairs[batch_idx:batch_idx + 32]

        inputs = tokenizer.batch_encode_plus(batch_data, 
                                       return_tensors='pt',  # pytorch tensors
                                       add_special_tokens=True,  # Add CLS and SEP tokens
                                       max_length = 512, # Max sequence length
                                       truncation = True, # Truncate if sequences exceed the Max Sequence length
                                       padding = True) # Add padding to forward sequences with different lengths
        
        # Forward the batch of (query, doc) sequences
        with torch.no_grad():
            inputs.to(device)
            outputs = model(**inputs)

        # Get the CLS embeddings for each pair query, document
        batch_cls = outputs['hidden_states'][-1][:,0,:]
        
        # Store the extracted CLS embeddings from the batch on the memory-mapped ndarray
        embeddings_persistent[batch_idx:batch_idx + 32] = batch_cls.cpu()   
        
    test_dataset['query_id'] = test_q_ids
    test_dataset['doc_id'] = test_doc_ids
    test_dataset['embedding'] = embeddings_persistent
    test_dataset['relevance'] = test_relevance

    #Save datasets to files
    
    file_data = open('test_dataset.obj', 'wb') 
    pickle.dump(test_dataset, file_data)
    file_data.close()


def loadDatasets():
    if exists('train_dataset.obj') and exists('test_dataset.obj'):  
        
        #load the objects from pickle
        file_data = open('train_dataset.obj', 'rb') 
        train_dataset = pickle.load(file_data)
        file_data.close()
        
        file_data = open('test_dataset.obj', 'rb') 
        test_dataset = pickle.load(file_data)
        file_data.close()
        
        return train_dataset, test_dataset
    
    else:
        print("Dataset files do not exist. Run generateAndSaveDataset() first")