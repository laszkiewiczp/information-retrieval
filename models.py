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
                 section: str = 'brief_title',
                 lemmatize: bool = False, unigrams: bool = True,
                 bigrams: bool = False, lowercase: bool = False,
                 stopwords = None):
        
        if unigrams:
            lower_range = 1
        else:
            lower_range = 1
        
        if bigrams:
            upper_range = 2
        else:
            upper_range = 1
        
        if section == 'brief_title':
            corpus = list(TrialsData.brief_titles.values())
            id_list = list(TrialsData.brief_titles.keys())
        elif section == 'description':
            corpus = list(TrialsData.description.values())
            id_list = list(TrialsData.description.keys())
        elif section == 'brief_summary':
            corpus = list(TrialsData.brief_summary.values())
            id_list = list(TrialsData.brief_summary.keys())
        elif section == 'criteria':
            corpus = list(TrialsData.criteria.values())
            id_list = list(TrialsData.criteria.keys())
        else:
            print('Wrong document section.')
            return
                    
        if lemmatize:
            nlp = spacy.load("en_core_web_sm")
            for idx, document in enumerate(corpus):
                doc = [token.lemma_ + ' ' for token in nlp(document)]
                doc_text = ''.join(doc)
                corpus[idx] = doc_text

        #Learn a vocab of unigrams and bigrams
        index = TfidfVectorizer(ngram_range=(lower_range, upper_range),
                                analyzer='word', stop_words=stopwords,
                                lowercase = lowercase)
        index.fit(corpus)

        #Compute the corpus representation
        X = index.transform(corpus)

        self.corpus = corpus
        self.id_list = np.array(id_list)
        self.index = index
        self.X = X
        self.lemmatize = lemmatize
            
    def get_top_query_results(self, query: str) -> list[str]:
        '''
        Function used to find search results for a given query
        '''
        if self.lemmatize:
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
    
    def get_query_scores(self, query: str) -> list[float]:
        '''
        Function used to get scores for each doc for a given query
        '''
        if self.lemmatize:
            nlp = spacy.load("en_core_web_sm")
            lemmas = [token.lemma_ + ' ' for token in nlp(query)]
            query = ''.join(lemmas)
            
        #Transform query to vector form
        query_tfidf = self.index.transform([query])
        
        #Calculate the scores
        doc_scores = np.array(1 - pairwise_distances(self.X, query_tfidf, metric='cosine')[:, 0])
        
        return doc_scores


class LMJM:
    '''
    Class to store a LM with Jelineck-Mercer smoothing. Will be fit to 
    ClinicalTrials object
    
    '''
    
    def __init__(self, TrialsData: Type[ClinicalTrials], arg_lambda: float,
                 section: str = 'brief_title',
                 unigrams: bool = True, bigrams: bool = False,
                 lowercase: bool = False, stopwords = None,
                 lemmatize: bool = False):
        
        if unigrams:
            lower_range = 1
        else:
            lower_range = 1
        
        if bigrams:
            upper_range = 2
        else:
            upper_range = 1
        
        if section == 'brief_title':
            corpus = list(TrialsData.brief_titles.values())
            id_list = list(TrialsData.brief_titles.keys())
        elif section == 'description':
            corpus = list(TrialsData.description.values())
            id_list = list(TrialsData.description.keys())
        elif section == 'brief_summary':
            corpus = list(TrialsData.brief_summary.values())
            id_list = list(TrialsData.brief_summary.keys())
        elif section == 'criteria':
            corpus = list(TrialsData.criteria.values())
            id_list = list(TrialsData.criteria.keys())
        else:
            print('Wrong document section.')
            return
        
        if lemmatize:
            nlp = spacy.load("en_core_web_sm")
            for idx, document in enumerate(corpus):
                doc = [token.lemma_ + ' ' for token in nlp(document)]
                doc_text = ''.join(doc)
                corpus[idx] = doc_text

        #Learn a vocab of unigrams
        vectorizer = CountVectorizer(ngram_range=(lower_range, upper_range),
                                     analyzer='word', stop_words=stopwords,
                                     lowercase = lowercase)
        
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
        self.term_corpus_p = np.array(term_corpus_p)
        self.term_doc_p = np.array(term_doc_p)
        self.arg_lambda = arg_lambda
        self.lemmatize = lemmatize
        
    def get_top_query_results(self, query: str) -> list[str]:        
        if self.lemmatize:
            nlp = spacy.load("en_core_web_sm")
            lemmas = [token.lemma_ + ' ' for token in nlp(query)]
            query = ''.join(lemmas)
           
        #Transform query to vectorized form
        query_vectorized = self.vectorizer.transform([query]).toarray()[0]
        non_zero_idx = np.where(query_vectorized != 0)
        query_non_zero = query_vectorized[non_zero_idx]
        
        #Calculate probabilities for different documents
        document_t_p = np.power(self.arg_lambda * self.term_doc_p[:, non_zero_idx],
                                    query_non_zero)
        
        corpus_t_p = np.power((1 - self.arg_lambda) * self.term_corpus_p[:, non_zero_idx],
                              query_non_zero)
        
        sums = np.log(document_t_p + corpus_t_p)
        
        p = [np.sum(single_sum) for single_sum in sums]
        p = np.array(p)
        
        #Find indices of top scores, sorted
        n = len(p)
        top_indices = np.argpartition(p, -n)[-n:]
        top_indices = np.flip(top_indices[np.argsort(p[top_indices])])
        
        #Find the ids of these clinical trials
        top_ids = self.id_list[top_indices]
        
        return top_ids
    
    def get_query_scores(self, query: str) -> list[float]:
        '''
        Function used to get scores for each doc for a given query
        '''
        if self.lemmatize:
            nlp = spacy.load("en_core_web_sm")
            lemmas = [token.lemma_ + ' ' for token in nlp(query)]
            query = ''.join(lemmas)
            
        #Transform query to vectorized form
        query_vectorized = self.vectorizer.transform([query]).toarray()[0]
        non_zero_idx = np.where(query_vectorized != 0)
        query_non_zero = query_vectorized[non_zero_idx]
        
        #Calculate probabilities for different documents
        document_t_p = np.power(self.arg_lambda * self.term_doc_p[:, non_zero_idx],
                                    query_non_zero)
        
        corpus_t_p = np.power((1 - self.arg_lambda) * self.term_corpus_p[:, non_zero_idx],
                              query_non_zero)
        
        sums = np.log(document_t_p + corpus_t_p)
        
        p = [np.sum(single_sum) for single_sum in sums]
        p = np.array(p)
        
        return p