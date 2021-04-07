from interface import implements, Interface

class LanguageModelTranslator(Interface):

    def featurize(self, data, preprocessor, mode='multilabel', remove_unlableled=True, text_key='abstract', primary_label_key='label1', secondary_label_key='label2', neg_sample_class=11, remove_neg_samples=False):
        pass

    def convert(self, words_list):
        pass
    
    def load_model(self):
        pass