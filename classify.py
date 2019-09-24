import json
import codecs
import numpy as np

class Classify():
    embeddings = dict();
    def __init__(self):
        vocab = codecs.open('/app/models/text.vocab','r').read().split('\n');
        label = codecs.open('/app/models/label.vocab','r').read().split('\n');
        maxlen = 12;
        embedding_size = 50;
        self.loadEmbeddings()

    def loadEmbeddings(self):
        f = open('/app/models/embeddings.vec');
        c = 0;
        for line in f:
            values = line.split();
            word = int(values[0]);
            coefs = np.asarray(values[1:], dtype='float32');
            self.embeddings[word] = coefs;
            c = c + 1;
        f.close();

    def processQuery(self):
        return "test"