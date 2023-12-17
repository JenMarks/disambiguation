# Below we import the needed dependencies
import numpy as np
import matplotlib.pyplot as plt
import spacy
import os
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

def preprocess_data(text_string):
  """
  This function receives one big text string document and returns:
  1) sentences_tokenized: (list(list(string))) This is the list of 
  tokenized sentences of the  text_string document.
  2) sentences_xxx: (list(list(string))) This list is the same as 
  sentences_tokenized with  tokens <a> and <the> replaced by <XXX>
  3) tags: (list(list(string))) This parameter contains the labels of each
  token, all tokens are labeled <0> except "XXX" which is labeled <1> if it was 
  originally an <a>, and <2> if it was originally <the>
  """
  nlp = spacy.load('en')
  doc = nlp(text_string)
  sentences_tokenized = [[token.string.strip() for token in nlp(sent.string.strip())] for sent in doc.sents]
  sentences_xxx = []
  tags = []
  for sentence in sentences_tokenized:
    sentence_xxx = [word if word not in ["a","the"] else "XXX" for word in sentence]
    tag = [0 if word not in ["a","the"] else {"a":1,"the":2}[word] for word in sentence]
    sentences_xxx.append(sentence_xxx)
    tags.append(tag)
  return (sentences_tokenized,sentences_xxx,tags)

def data_stats(sentences):
  """
  This function receives:
  sentences: (list(list(string))) A list of tokenized sentences
  
  Returns:
  1) max_sentence_len: (int) max_length of a sentence
  2) len(words): (int) Number of words in the vocabulary
  3) list(words): (list) A list of words in the vocabulary
  """
  max_sentence_len = 0 
  words = set()
  for sentence in sentences:
    if len(sentence) > max_sentence_len:
      max_sentence_len = len(sentence)
    for word in sentence:
      words.add(word)
  return (max_sentence_len,len(words),list(words))

# We first read the data set located in the folder data
training_data_set = ""
for file in os.listdir("data"):
  if file.endswith(".txt"):
    with open(file,'r') as f:
      training_data_set = training_data_set +" "+ f.read()

# We then read the test document (CharlesDickensTwoTales_orig.txt)
with open("CharlesDickensTwoTales_orig.txt",'r') as f:
  test_data = f.read()

# Now we call <preprocess_data> on the above files to turn 
#them into list of sentences
(tr_sentences,tr_sentences_xxx,tr_tags) = preprocess_data(training_data_set)
(test_sentences,test_sentences_xxx,test_tags) = preprocess_data(test_data)

(max_sentence_len, vocab_num, vocab) = data_stats(tr_sentences_xxx)

padded_sentence_len = 200

print("Maximum sentence length is: {}".format(max_sentence_len))
print("Number of items in the vocabulary is: {}".format(vocab_num))
print("We chose the padded sentence length to be: {}".format(padded_sentence_len))

word_id = {word:id_num for id_num, word in enumerate(vocab)}

X = [[word_id[word] for word in sentence] for sentence in tr_sentences_xxx]
X = pad_sequences(maxlen=max_sentence_len, sequences=X, padding="post", value=vocab_num - 1)

X_test = [[word_id.get(word,int(len(word_id)/2)) for word in sentence] for sentence in test_sentences_xxx]

Y = pad_sequences(maxlen=max_sentence_len, sequences=tr_tags, padding="post", value=0)
Y = [to_categorical(instance, num_classes=3) for instance in Y]

Y_test = pad_sequences(maxlen=max_sentence_len, sequences=test_tags, padding="post", value=0)

input_layer = Input(shape=(padded_sentence_len,))
model = Embedding(input_dim=vocab_num, output_dim=50, input_length=padded_sentence_len)(input_layer)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(3, activation="softmax"))(model)

model = Model(input_layer, out)

adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, np.array(Y_train), batch_size=32, epochs=10, validation_split=0.1, verbose=1)

plt.plot(history.history["acc"],label='Training Accuracy',color=(1,0,0))
plt.plot(history.history["val_acc"],label='Validation Accuracy',color=(0,0,1))
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

def acc_compute(y_preds,y_trues):
  """
  This function only evaluates the accuracy of the model in label predicition 
  for XXX
  """
  total
  correct
  for y_pred,y_true in zip(y_preds,y_trues):
    for i in range(200):
      if y_true[i] != 0:
        if y_true[i] == y_pred[i]:
          correct +=1
          total +=1
        else:
          total +=1
          
  return (correct, total)

y_test_pred = np.argmax(model.predict(X_test),axis =-1)
(correct, total) = acc_compute(y_test_pred,Y_test)
print("Number of correctly classified XXX: {} from a total of: {} producing an accuracy of: {} percent.".format(correct,total,correct*100.0/total))