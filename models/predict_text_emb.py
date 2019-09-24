import json
import numpy as np
import codecs
import tensorflow as tf

vocab = codecs.open('text.vocab','r').read().split('\n')
label = codecs.open('label.vocab','r').read().split('\n')

maxlen = 12
embedding_size = 50


embeddings = dict()
f = open('embeddings.vec')
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
    inmb = [vocab.index(word)+1 if word in vocab else 0 for word in query.split()]
    print (inmb)
    inmb = inmb[:maxlen]
    if len(inmb) < maxlen:
        inmb = [0 for i in range(maxlen-len(inmb))] + inmb
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
        

sent = 'question and '
sent = process_query(sent)

interpreter = tf.lite.Interpreter(model_path='model.tflite')

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
  print (label[prediction])
  print (result[0][prediction])
  print (label[prediction])
