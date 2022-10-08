import xml.etree.ElementTree as ET
import tarfile


class ClinicalTrials:
    '''
    This class is used to store clinical trials data
    '''
    
    def __init__ (self, trial_ids):        
        self.ids = trial_ids
        
        #Create a dictionary mapping ids to brief titles
        self.brief_titles =  {}
        
        #Extract tar file data
        tar = tarfile.open("clinicaltrials.gov-16_dec_2015.tgz", "r:gz")
        
        for tarinfo in tar:
            
            if tarinfo.size > 500:               
                txt = tar.extractfile(tarinfo).read().decode("utf-8", "strict")
                root = ET.fromstring(txt)

                for doc_id in root.iter('nct_id'):
                    
                    if (doc_id.text not in self.ids):
                        continue
                        
                    else:
                        current_id = doc_id.text
                        for brief_title in root.iter('brief_title'):
                            self.brief_titles[current_id] = brief_title.text
                    
        tar.close()


class Query:
    '''
    This class is used to store query data
    '''
    def __init__ (self, query_id, query_content):
        self.query_id = query_id
        self.query_content = query_content
        self.ground_truth ={}
    
    def add_ground_truth (self, ground_truths):
        '''
        Function to add ground truth to a given query\n
        :ground_truths: A list of tuples (document_id, score)
        
        '''
        for gt in ground_truths:
            self.ground_truth[gt[0]] = gt[1]