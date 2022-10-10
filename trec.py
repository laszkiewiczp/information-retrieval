import rank_metric as metrics
import pandas as pd
import numpy as np

class TrecEvaluation:

    def __init__(self, queries, qrels):

        # fields: docid, rel
        self.queries = queries
        self.relevance_judgments = pd.read_csv(qrels, sep='\t', names=["query_id", "dummy", "docid", "rel"])

        self.judged_docs = np.unique(self.relevance_judgments['docid'])

        self.num_docs = len(self.judged_docs)

    def eval(self, result, query_id):
        
        total_retrieved_docs = result.count()[0]

        aux = self.relevance_judgments.loc[self.relevance_judgments['query_id'] == int(query_id)]

        rel_docs = aux.loc[aux['rel'] != 0]
        query_rel_docs = rel_docs['docid']
        relv_judg_list = rel_docs['rel']
        total_relevant = relv_judg_list.count()
        
        if total_relevant == 0:
            return [0, 0, 0, 0, 0]

        # P@10
        top10 = result['_id'][:10]
        true_pos = np.intersect1d(top10,query_rel_docs)
        p10 = np.size(true_pos) / 10
        
        true_pos = np.intersect1d(result['_id'],query_rel_docs)
        recall = np.size(true_pos) / total_relevant

        # Compute vector of results with corresponding relevance level 
        relev_judg_results = np.zeros((total_retrieved_docs,1))
        for index, doc in rel_docs.iterrows():
            relev_judg_results = relev_judg_results + ((result['_id'] == doc.docid)*doc.rel).to_numpy()
        
        # Normalized Discount Cummulative Gain
        p10 = metrics.precision_at_k(relev_judg_results[0], 10)
        ndcg5 = metrics.ndcg_at_k(r = relev_judg_results[0], k = 5, method = 1)
        ap = metrics.average_precision(relev_judg_results[0], total_relevant)
        mrr = metrics.mean_reciprocal_rank(relev_judg_results[0])
        
        return [p10, recall, ap, ndcg5, mrr]

    def evalPR(self, scores, query_id):

        #aux = self.relevance_judgments.loc[self.relevance_judgments['topic_turn_id'] == (topic_turn_id)]
        aux = self.relevance_judgments.loc[self.relevance_judgments['query_id'] == int(query_id)]

        idx_rel_docs = aux.loc[aux['rel'] != (0)]

        [dummyA, rank_rel, dummyB] = np.intersect1d(scores['_id'], idx_rel_docs['docid'], return_indices=True)
        rank_rel = np.sort(rank_rel) + 1
        total_relv_ret = rank_rel.shape[0]
        if total_relv_ret == 0:
            return [np.zeros(11, ), [], total_relv_ret]

        recall = np.arange(1, total_relv_ret + 1) / idx_rel_docs.shape[0]
        precision = np.arange(1, total_relv_ret + 1) / rank_rel

        precision_interpolated = np.maximum.accumulate(precision)
        recall_11point = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        precision_11point = np.interp(recall_11point, recall, precision)

        if False:
            print(total_relv_ret)
            print(rank_rel)
            print(recall)
            print(precision)
            plt.plot(recall, precision, color='b', alpha=1)  # Raw precision-recall
            plt.plot(recall, precision_interpolated, color='r', alpha=1)  # Interpolated precision-recall
            plt.plot(recall_11point, precision_11point, color='g', alpha=1)  # 11-point interpolated precision-recall

        return [precision_11point, recall_11point, total_relv_ret]

