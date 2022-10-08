import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from data_storage import ClinicalTrials, Query


def list_evaluated_trials() -> list[str]:
    '''
    This function loads ids of trials that were evaluated

    :return: List of evaluated clinical trials' ids

    '''
    ids = []
    
    f = open('qrels-clinical_trials.txt','r')
    
    while True:
        line = f.readline()
        
        if not line:
            break
            
        _, _, clinical_trial_id, _ = line.split()
    
        if clinical_trial_id not in ids:
            ids.append(clinical_trial_id)
    f.close()   
    
    return ids


def list_evaluated_queries() -> list[str]:
    '''
    This function loads ids of trials that were evaluated

    :return: List of evaluated clinical trials' ids

    '''
    ids = []
    
    f = open('qrels-clinical_trials.txt','r')
    
    while True:
        line = f.readline()
        
        if not line:
            break
            
        query_id, _, _, _ = line.split()
    
        if query_id not in ids:
            ids.append(query_id)
    f.close()   
    
    return ids


def load_evaluated_trials():
    '''
    :return: An object of class ClinicalTrials corresponding to
    the ids of evaluated clinical trials

    '''
    return ClinicalTrials(list_evaluated_trials())


def load_and_split_queries():
    '''
    Function to load and split query data \n
    :returns: two lists - train and test filled with objects of class Query
    
    '''
    
    query_list = []
    evaluated_queries = list_evaluated_queries()

    Queries_file = "topics-2014_2015-summary.topics"

    with open(Queries_file, 'r') as queries_reader:
        txt = queries_reader.read()
        
    root = ET.fromstring(txt)
    
    #A list of tuples (query_id, query_content)
    query_list = [(query.find('NUM').text, query.find('TITLE').text) for query in root.iter('TOP') if (query.find('NUM').text in evaluated_queries)]
    
    #Shuffle and split queries to train and test set, set seed for reproductibility
    train_queries, test_queries, _ , _ = train_test_split(query_list, query_list,
                                                          test_size = 0.2, random_state = 1)
    
    #Transform the generated query lists to objects of Query class
    train = [Query(query_id, content) for (query_id, content) in train_queries]
    test = [Query(query_id, content) for (query_id, content) in test_queries]
    
    #Load the ground truths from file
    ground_truth = load_queries_gt()
    
    #For each of the queries, add ground truth
    for train_q in train:
        train_q.add_ground_truth(ground_truth[train_q.query_id])
    
    for test_q in test:
        test_q.add_ground_truth(ground_truth[test_q.query_id])
    
    return train, test


def load_queries_gt():
    '''
    Function used to load ground truths for the evaluated queries.
    :returns: A dictionary {query_id: list[(doc_id, score)]}
    
    '''
    
    gt = {}
    
    f = open('qrels-clinical_trials.txt','r')
    
    while True:
        line = f.readline()
        
        if not line:
            break
            
        query_id, _, clinical_trial_id, score = line.split()
        
        if query_id in gt:          
            gt[query_id].append((clinical_trial_id, score))
        
        else:
            gt[query_id] = [(clinical_trial_id, score)]
        
    f.close()
    
    return gt
    