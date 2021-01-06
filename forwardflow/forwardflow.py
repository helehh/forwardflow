from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import numpy as np

class ForwardFlow:
    def __init__(self, embedding):
        self.embedding = KeyedVectors.load_word2vec_format(embedding, binary=False)
        
    def score(self, words):        
        #for each word calculate instantanious forward flow - average semantic distance from all previous words
        #we start counting from the second element since there 
        #must exist at least two words to calculate forward word
        
        word_flow_scores = []
        for i in range(1, len(words)):
            dists = self.embedding.distances(words[i], words[0:i])
            word_flow_scores.append(np.mean(dists))
        
        return np.mean(word_flow_scores)