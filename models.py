import numpy as np
import spacy

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Type
from data_storage import ClinicalTrials

class VectorSpaceModel:
    '''
    Class to store a vector space model. The model is fit to 
    data stored in a ClinicalTrials object
    '''
    
    def __init__(self, TrialsData: Type[ClinicalTrials],
                 tokenize: bool = False):
        #Corpus is a list of brief titles
        corpus = list(TrialsData.brief_titles.values())
        
        if tokenize:
            nlp = spacy.load("en_core_web_sm")
            for idx, document in enumerate(corpus):
                doc = [token.lemma_ + ' ' for token in nlp(document)]
                doc_text = ''.join(doc)
                corpus[idx] = doc_text

        #Rearrange id_list so that the order matches brief titles
        id_list = list(TrialsData.brief_titles.keys())

        #Learn a vocab of unigrams and bigrams
        index = TfidfVectorizer(ngram_range=(1, 1), analyzer='word',
                            stop_words=None, lowercase = False)
        index.fit(corpus)

        #Compute the corpus representation
        X = index.transform(corpus)
        
        self.corpus = corpus
        self.id_list = np.array(id_list)
        self.index = index
        self.X = X
        self.tokenize = tokenize
            
    def get_top_query_results(self, query: str) -> list[str]:
        '''
        Function used to find search results for a given query
        '''
        if self.tokenize:
            nlp = spacy.load("en_core_web_sm")
            lemmas = [token.lemma_ + ' ' for token in nlp(query)]
            query = ''.join(lemmas)
            
        #Transform query to vector form
        query_tfidf = self.index.transform([query])
        
        #Calculate the scores
        doc_scores = np.array(1 - pairwise_distances(self.X, query_tfidf, metric='cosine')[:, 0])
        
        #Find indices of top scores, sorted
        n = len(doc_scores)
        top_indices = np.argpartition(doc_scores, -n)[-n:]
        top_indices = np.flip(top_indices[np.argsort(doc_scores[top_indices])])
        
        #Find the ids of these clinical trials
        top_ids = self.id_list[top_indices]
        
        return top_ids