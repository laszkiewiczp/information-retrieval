import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Type
from data_storage import ClinicalTrials

class VectorSpaceModel:
    '''
    Class to store a vector space model. The model is fit to 
    data stored in a ClinicalTrials object
    '''
    
    def __init__(self, TrialsData: Type[ClinicalTrials]):
        #Corpus is a list of brief titles
        corpus = list(TrialsData.brief_titles.values())

        #Rearrange id_list so that the order matches brief titles
        id_list = list(TrialsData.brief_titles.keys())

        #Learn a vocab of unigrams and bigrams
        index = TfidfVectorizer(ngram_range=(1, 2), analyzer='word',
                            stop_words=None)
        index.fit(corpus)

        #Compute the corpus representation
        X = index.transform(corpus)
        
        self.corpus = corpus
        self.id_list = np.array(id_list)
        self.index = index
        self.X = X
            
    def get_top_query_results(self, query: str, n = 100) -> list[str]:
        '''
        Function used to find search results for a given query
        '''
        #Transform query to vector form
        query_tfidf = self.index.transform([query])
        
        #Calculate the scores
        doc_scores = np.array(1 - pairwise_distances(self.X, query_tfidf, metric='cosine')[:, 0])
        
        #Find indices of top scores, sorted
        top_indices = np.argpartition(doc_scores, -n)[-n:]
        top_indices = np.flip(top_indices[np.argsort(doc_scores[top_indices])])
        
        #Find the ids of these clinical trials
        top_ids = self.id_list[top_indices]
        
        return top_ids