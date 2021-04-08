from interface import implements, Interface
from LanguageModels.LanguageModelTranslator import LanguageModelTranslator
import gensim
from gensim import corpora
from gensim import matutils
import numpy as np

class BagOfWords(implements(LanguageModelTranslator)):
    
    def __init__(self,):
        pass


    def featurize(self, data, preprocessor, mode='multilabel', remove_unlableled=True, text_key='abstract', primary_label_key='label1', secondary_label_key='label2', neg_sample_class=11, remove_neg_samples=False):
        data_cleaned = []
        labels = []
    
        for row in data:
            if remove_unlableled == True and row[primary_label_key] == '':
                continue

            if remove_neg_samples and int(row[primary_label_key]) == neg_sample_class:
                continue
                
            if mode.lower() == 'multilabel':
                labels.append([int(row[primary_label_key]), None if row[secondary_label_key] == '' else int(row[secondary_label_key])])
            elif mode.lower() == 'multiclass':
                labels.append([int(row[primary_label_key])])
            elif mode.lower() == 'binary':
                labels.append([0 if int(row[primary_label_key]) != neg_sample_class else 1])
            else:
                print("ERROR: unreconized mode %s. Choose one of multilabel, multiclass, or binary (0=in conference, 1=not in conference)" % mode)
            
            data_cleaned.append(row)
    
        dictionary = corpora.Dictionary()        
        tokenized_abstracts = [preprocessor.clean(row[text_key]) for row in data_cleaned]
        BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_abstracts]
        features = matutils.corpus2dense(BoW_corpus, num_terms=len(dictionary.token2id)).T

        features = np.array(features)
        labels = np.array(labels)
        
        return features, labels

    def convert(self, words_list):
        pass
    
    def load_model(self):
        pass