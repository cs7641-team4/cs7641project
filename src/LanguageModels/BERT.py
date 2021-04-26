#from interface import implements, Interface

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from .LanguageModelTranslator import LanguageModelTranslator
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
#class BERT(implements(LanguageModelTranslator)):
class BERT(LanguageModelTranslator):
    def __init__(self, path='bert-base-uncased', cuda=False):
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertModel.from_pretrained(path)

        self.cuda = cuda

        self.load_model()
        pass

    def featurize(self, data, preprocessor, mode='multilabel', remove_unlableled=True, text_key='abstract', primary_label_key='label1', secondary_label_key='label2', neg_sample_class=11, remove_neg_samples=False):
        features = []
        labels = []
        for row in tqdm(data):

            if remove_unlableled == True and row[primary_label_key] == '':
                continue

            if remove_neg_samples and int(row[primary_label_key]) == neg_sample_class:
                continue

            abstract = row[text_key]
            abstract_clean = preprocessor.clean(abstract)
            features.append(self.convert(abstract_clean, method='average'))

            if mode.lower() == 'multilabel':
                labels.append([int(row[primary_label_key]), None if row[secondary_label_key] == '' else int(row[secondary_label_key])])
            elif mode.lower() == 'multiclass':
                labels.append([int(row[primary_label_key])])
            elif mode.lower() == 'binary':
                labels.append([0 if int(row[primary_label_key]) != neg_sample_class else 1])
            else:
                print("ERROR: unreconized mode %s. Choose one of multilabel, multiclass, or binary (0=in conference, 1=not in conference)" % mode)
            

        features = np.array(features)
        labels = np.array(labels)
        return features, labels


    def convert(self, words_list, method='average'):
        tokens = self.tokenizer.encode(' '.join(words_list))
        input_ids = torch.tensor(tokens).unsqueeze(0)

        if self.cuda:
            input_ids.cuda()
        outputs = self.model(input_ids)
        last_hidden_states = outputs[0]
        return last_hidden_states.squeeze(0).mean(axis=0).cpu().detach().numpy()
    
    def load_model(self):
        pass