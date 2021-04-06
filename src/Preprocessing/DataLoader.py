import csv

class DataLoader():
    def __init__(self, fname = 'data/EMNLP2020.csv'):
        self.fname = fname
    
    def load(self):
        data = []
        with open (self.fname, 'r', encoding = 'utf-8') as f: 
            input_file = csv.DictReader(f) 
            for row in list(input_file):
                data.append(row)
        
        return data
