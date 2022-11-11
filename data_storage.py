import xml.etree.ElementTree as ET
import tarfile


class ClinicalTrials:
    '''
    This class is used to store clinical trials data
    '''
    
    def __init__ (self, trial_ids):        
        self.ids = trial_ids
        
        #Create a dictionary mapping ids to different sections of the doc
        self.brief_titles =  {}
        self.description = {}
        self.brief_summary = {}
        self.criteria = {}
        self.gender = {}
        self.min_age = {}
        self.max_age = {}
        
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
                        
                        self.description[current_id] = ''    
                        for detailed_description in root.iter('detailed_description'):
                            description = ''
                            for child in detailed_description:
                                description += child.text.strip()
                                
                            self.description[current_id] = description
                        
                        self.brief_summary[current_id] = ''
                        for brief_summary in root.iter('brief_summary'):
                            summary = ''
                            for child in brief_summary:
                                summary +=  child.text.strip()  
                                
                            self.brief_summary[current_id] = summary

                        self.criteria[current_id] = ''
                        for criteria in root.iter('criteria'):
                            crit = ''
                            for child in criteria:
                                crit += child.text.strip()
                            
                            self.criteria[current_id] = crit
                        
                        self.gender[current_id] = None
                        for gender in root.iter('gender'):
                            self.gender[current_id] = gender.text
                        
                        self.min_age[current_id] = None
                        for minimum_age in root.iter('minimum_age'):
                            self.min_age[current_id] = minimum_age.text
                        
                        self.max_age[current_id] = None
                        for maximum_age in root.iter('maximum_age'):
                            self.max_age[current_id] = maximum_age.text
                        
                        if self.description[current_id] == '':
                            self.description[current_id] = self.brief_titles[current_id] +  self.brief_summary[current_id]
                        
                        if self.brief_summary[current_id] == '':
                            self.brief_summary[current_id] = self.description[current_id] + self.brief_titles[current_id]
                        
                        if self.criteria[current_id] == '':
                            self.criteria[current_id] = self.brief_titles[current_id]
                    
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