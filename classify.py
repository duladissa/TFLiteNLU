import json
import codecs
import numpy as np
import tensorflow as tf

class Classify():
    maxlen = 10
    embedding_size = 50
    embeddings = dict()
    interpreter = {}
    vocab = {}
    label = {}

    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path='/app/models/converted_model.tflite')
        self.vocab = codecs.open('/app/models/text.vocab','r').read().split('\n');
        self.label = codecs.open('/app/models/label.vocab','r').read().split('\n');
        self.loadEmbeddings()

    def loadEmbeddings(self):
        f = open('/app/models/classifier_embd.vec');
        c = 0;
        for line in f:
            values = line.split();
            word = int(values[0]);
            coefs = np.asarray(values[1:], dtype='float32');
            self.embeddings[word] = coefs;
            c = c + 1;
        f.close();

    def classify(self, text):
        processed = self.processQuery(text)
        return self.inference(processed)

    def processQuery(self, text):
        inmb = [self.vocab.index(word)+1 if word in self.vocab else 0 for word in text.split()]
        inmb = inmb[:self.maxlen]
        if len(inmb) < self.maxlen:
            inmb = [0 for i in range(self.maxlen-len(inmb))] + inmb
        for c , w in enumerate(inmb):
            if w in self.embeddings:
                wemd = self.embeddings[w]
            else:
                wemd = np.zeros(self.embedding_size)
            if c == 0:
                f_ = wemd
            else:
                f_ = np.hstack((f_,wemd))
        return f_.reshape(self.maxlen, self.embedding_size)

    def inference(self, processed):
        try:
            self.interpreter.allocate_tensors()
        except ValueError:
            assert False

        MINI_BATCH_SIZE = 1
        test = np.array([processed]).astype(np.float32)
        for i in range(len(test)):
            input_index = (self.interpreter.get_input_details()[0]['index'])
            self.interpreter.set_tensor(input_index, test[i * MINI_BATCH_SIZE: (i + 1) * MINI_BATCH_SIZE])
            self.interpreter.invoke()
            output_index = (self.interpreter.get_output_details()[0]['index'])
            result = self.interpreter.get_tensor(output_index)
            self.interpreter.reset_all_variables()
            prediction = np.argmax(result)
            return self.label[prediction]