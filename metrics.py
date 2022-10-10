import rank_metric as metrics
import matplotlib.pyplot as plt
import numpy as np

class Evaluate:
    
    def __init__(self, query, search_result):
        self.gt = {k: v for k, v in query.ground_truth.items() if v != '0'}
        self.gt_doc_ids = list(self.gt.keys())
        self.gt_ratings = list(self.gt.values())
        
        self.search_result = search_result
    
        number_of_relevant_docs = sum(x != '0' for x in self.gt.values())        
        
        if number_of_relevant_docs == 0:            
            self.p10 = 0
            self.recall = 0
            self.ap = 0
            self.ndcg5 = 0
            self.mrr = 0
            
        else:    
            #Get P@10
            top10 = self.search_result[:10]
            true_positive = np.intersect1d(top10, self.gt_doc_ids)
            self.p10 = np.size(true_positive)/10
            
            #Get recall@100
            true_positive = np.intersect1d(self.search_result[:100], self.gt_doc_ids)
            self.recall = np.size(true_positive) / number_of_relevant_docs
            
            total_retrieved_docs = len(self.search_result)
            relev_judg_results = np.zeros(total_retrieved_docs)
            
            for idx, doc_id in enumerate(self.gt_doc_ids):
                relev_judg_results += np.array([result_id == doc_id for result_id in 
                                                self.search_result]) * int(self.gt_ratings[idx])
            
            self.p10 = metrics.precision_at_k(relev_judg_results, 10)
            self.ndcg5 = metrics.ndcg_at_k(relev_judg_results, k = 5, method = 1)
            self.ap = metrics.average_precision(relev_judg_results, number_of_relevant_docs)
            self.mrr = metrics.mean_reciprocal_rank(relev_judg_results)
            
            
        [dummyA, rank_rel, dummyB] = np.intersect1d(self.search_result, self.gt_doc_ids, return_indices=True)
        rank_rel = np.sort(rank_rel) + 1
                
        if number_of_relevant_docs == 0:
            return [np.zeros(11, ), [], number_of_relevant_docs]
        
        recall = np.arange(1, number_of_relevant_docs + 1) / len(self.gt)
        precision = np.arange(1, number_of_relevant_docs + 1) / rank_rel
        
        precision_interpolated = np.maximum.accumulate(precision)
        self.recall_11point = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.precision_11point = np.interp(self.recall_11point, recall, precision)

        if False:
            print(number_of_relevant_docs)
            print(rank_rel)
            print(recall)
            print(precision)
            plt.plot(recall, precision, color='b', alpha=1)  # Raw precision-recall
            plt.plot(recall, precision_interpolated, color='r', alpha=1)  # Interpolated precision-recall
            plt.plot(self.recall_11point, self.precision_11point, color='g', alpha=1)  # 11-point interpolated precision-recall