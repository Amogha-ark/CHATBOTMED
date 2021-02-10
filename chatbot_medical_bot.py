import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
with open('C:/Users/Amogha Rao K/.PyCharmCE2019.1/config/scratches/intents.json') as f:
    data = json.load(f)
# print(data)
train_x = []
train_y = []
labels=[]
responses=[]

for i in data['intents']:

    for pattren in i['patterns']:
        train_x.append(pattren)
        train_y.append(i['tag'])
    responses.append(i['responses'])
    if i['tag'] not in labels:
        labels.append(i['tag'])


num_clas = len(labels)
# print(num_clas)
# print(train_x)
# print(train_y)
encodela = LabelEncoder()
encodela.fit(train_y)
train_y = encodela.transform(train_y)
# print(train_y)
oov_token = "<00V>"
tokenizer = text.Tokenizer(oov_token=oov_token)
tokenizer.fit_on_texts(train_x)
X = tokenizer.texts_to_sequences(train_x)
max_len = len(max(X,key=len))
# print(len(X))
X = sequence.pad_sequences(X, maxlen=max_len, padding='post',truncating='post')
# print(X[0],X[1])
# print(tokenizer.word_index)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(100,16,input_length=max_len))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(num_clas,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])

model.summary()
print(len(X))
model.fit(X,np.array(train_y),epochs=130,batch_size=10)

def string_to_integer(s):
    arr = tokenizer.texts_to_sequences(s)
    arr  =sequence.pad_sequences(arr,maxlen=max_len,padding='post',truncating='post')
    return arr

while True:
    s =input("You : ")
    if s=='quit':
        break
    ls = [s]
    prediction = model.predict(string_to_integer(ls))
    res = np.argmax(prediction)
    tag = encodela.inverse_transform([res])
    print(prediction[0][np.argmax(prediction)])
    if prediction[0][np.argmax(prediction)] > 0.45:
        print(tag)
        for i in data['intents']:
            if i['tag'] == tag:
                print("chatBot: "+ np.random.choice(i['responses']))
    else:
        print(prediction[0][np.argmax(prediction)])
        for i in data['intents']:
            if i['tag'] == tag:
                print("chatBot: "+ np.random.choice(i['responses']))
