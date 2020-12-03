from flask import Flask,request
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pickle
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
import nltk
import json
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer

app=Flask(__name__)

def load_model():
    loaded_model=tf.keras.models.load_model('text_to_emoji.h5')
    nltk.download('stopwords')#download stop world coppus
    return loaded_model

def load_tokenizer():
    with open('tokenizer.json') as f:
        data = json.load(f)
        loaded_tokenizer = tokenizer_from_json(data)
    return loaded_tokenizer

def load_label_encoder():
    file = open("le.obj",'rb')
    le_loaded = pickle.load(file)
    file.close()
    return le_loaded

def removeNumbers(t): 
    for i in range(10):
        t=t.replace(str(i),'')
    return t

def remove_stop_words(x,stop):
    word=''
    result=''
    counter=0
    for car in x:
        if car!=' ':
            word+=car
        else:
            if(word not in stop):
                result+=word
                if(counter+1!=len(x)):
                    result+=' '
            word=''
        counter+=1
    return result

def process_tweet(tweet):
    tweet=tweet.lower()
    tweet=removeNumbers(tweet)
    stop = stopwords.words('english')
    tweet = remove_stop_words(tweet,stop)
    return tweet

def text_to_vect(t1,loaded_tokenizer):
    tk = loaded_tokenizer.texts_to_sequences([t1])
    for i in range(len(tk)):
        for j in range(22-len(tk[i])):
            tk[i].insert(0,0)
    tk=np.array(tk)    
    return tk 

def predict(loaded_model,tk,le_loaded):
    pred = loaded_model.predict(tk)
    return le_loaded.inverse_transform([np.argmax(pred[0])])

is_first_time=False
model=tf.keras.Model()
tokenizer = Tokenizer(num_words=5000, split=" ")
le=preprocessing.LabelEncoder()

@app.route("/api",methods=["POST"])
def login():
    global is_first_time
    global model
    global tokenizer
    global le
    tweet=request.get_json()["tweet"]
    if(is_first_time==False):
        is_first_time=True
        model=load_model()
        tokenizer=load_tokenizer()
        le=load_label_encoder()
    tweet=process_tweet(tweet)
    tk=text_to_vect(tweet,tokenizer)
    pred=predict(model,tk,le)
    prediction=json.dumps(pred.tolist())[2:-2]
    print(prediction)
    return {"emoji":prediction}

@app.route("/api",methods=["Get"])
def loginGet():
    return {"status code":"200"}

if __name__ == '__main__':
    app.run(debug=True)