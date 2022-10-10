import numpy as np
import spacy

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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


class LMJM:
    '''
    Class to store a LM with Jelineck-Mercer smoothing. Will be fit to 
    ClinicalTrials object
    
    '''
    
    def __init__(self, TrialsData: Type[ClinicalTrials], arg_lambda: float):
        #Corpus is a list of brief titles
        corpus = list(TrialsData.brief_titles.values())
        
        #Rearrange id_list so that the order matches brief titles
        id_list = list(TrialsData.brief_titles.keys())
        
        #Learn a vocab of unigrams
        vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='word',
                            stop_words=None, lowercase = True)
        
        X = vectorizer.fit_transform(corpus)
        word_indices = vectorizer.get_feature_names_out()
        
        #Compute word counts
        word_counts = np.sum(X, axis = 0)
        
        #Compute doc lengths
        doc_lengths = np.sum(X, axis = 1)
        
        #Term probability in corpus
        term_corpus_p = word_counts / np.sum(X)
        
        #Term probability in the document
        term_doc_p = X / doc_lengths
        
        self.X = X
        self.id_list = np.array(id_list)
        self.vectorizer = vectorizer
        self.word_indices = word_indices
        self.term_corpus_p = term_corpus_p
        self.term_doc_p = term_doc_p
        self.arg_lambda = arg_lambda
        
    def get_top_query_results(self, query: str) -> list[str]:
        #Transform query to vectorized form
        query_vectorized = self.vectorizer.transform([query]).toarray()[0]
        p = np.zeros(self.term_doc_p.shape[0])
        
        document_t_p = np.power(self.arg_lambda * self.term_doc_p,
                                    query_vectorized)
        
        corpus_t_p = np.power((1 - self.arg_lambda) * self.term_corpus_p,
                              query_vectorized)
        
        sums = document_t_p + corpus_t_p
        
        for idx, single_sum in enumerate(sums):
            new_single_sum = [x for x in single_sum.tolist()[0] if x != 2]
            p[idx] = np.product(new_single_sum)
        
        #Find indices of top scores, sorted
        n = len(p)
        top_indices = np.argpartition(p, -n)[-n:]
        top_indices = np.flip(top_indices[np.argsort(p[top_indices])])
        
        #Find the ids of these clinical trials
        top_ids = self.id_list[top_indices]
        
        return top_ids