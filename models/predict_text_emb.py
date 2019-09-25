import json
import numpy as np
import codecs
import tensorflow as tf
from bpe import bpe
import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'




sent = 'when is the showtime'
#sent = 'are there any email'
#sent = 'what is the showtime'
#sent = 'showtime'
sent = 'when is the showtimes'
#sent = 'show me alarms'
#sent = ''




vocab = codecs.open('text.vocab','r').read().split('\n')
label = codecs.open('label.vocab','r').read().split('\n')

maxlen = 10
embedding_size = 50


embeddings = dict()
f = open('classifier_embd.vec')
c = 0
for line in f:
    values = line.split()
    word = int(values[0])
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings[word] = coefs
    c = c + 1
f.close()
print('Loaded %s word vectors.' % len(embeddings))

def process_query(query):
    query = query.lower()
    #query = bpe.process_line(query)
    print (query,'<--- in')
    inmb = [vocab.index(word)+1 if word in vocab else 0 for word in query.split()]
    print (inmb)
    inmb = inmb[:maxlen]
    if len(inmb) < maxlen:
        inmb = [0 for i in range(maxlen-len(inmb))] + inmb
    print (inmb)
    for c , w in enumerate(inmb):
        if w in embeddings:
            wemd = embeddings[w]
        else:
            wemd = np.zeros(embedding_size)
        if c == 0:
            f_ = wemd
        else:
            f_ = np.hstack((f_,wemd))
    #print (f_)
    return f_.reshape(maxlen, embedding_size)
        

#sent = 'when is the showtime for lionking'
sent_org = sent
sent = process_query(sent)

interpreter = tf.lite.Interpreter(model_path='converted_model.tflite')

try:
  interpreter.allocate_tensors()
except ValueError:
  assert False

MINI_BATCH_SIZE = 1
correct_case = 0
test = np.array([sent]).astype(np.float32)
for i in range(len(test)):
  input_index = (interpreter.get_input_details()[0]['index'])
  interpreter.set_tensor(input_index, test[i * MINI_BATCH_SIZE: (i + 1) * MINI_BATCH_SIZE])
  interpreter.invoke()
  output_index = (interpreter.get_output_details()[0]['index'])
  result = interpreter.get_tensor(output_index)
  interpreter.reset_all_variables()
  prediction = np.argmax(result)
  print ()
  print (sent_org)
  print (result)
  print (result[0][prediction])
  print (label[prediction])
