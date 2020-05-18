#!/usr/bin/env python
# coding: utf-8

import os
import bz2
import re

import librosa.display
import numpy as np
import os
import pandas as pd
import glob as gb
from keras.utils import np_utils
import matplotlib.pyplot as plt
import tempfile
import tensorflow as tf



from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model

model = load_model('final2.h5')
print("Final Models Loaded")
global graph
graph = tf.get_default_graph()

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')



@app.route('/voice', methods = ['POST'])
def voice_upload():
    if request.method == "POST":
        audio = request.files.get('audio-blob')
        print(audio)
		
        audio.save(os.path.join("C:/Users/dependra/Flask", audio.filename))

        data, sampling_rate = librosa.load(audio.filename,duration=1)
        mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=60)
        pad_width = 60 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        X = mfcc.reshape(1,1,60,60)
        with graph.as_default():
            predictions = model.predict(X)
        
    selected_option = int(np.argmax(predictions))
    print('Selected Option = ' , selected_option)
    response = {'selected_option': selected_option, 'status': 'success'}
    return jsonify(response)




if __name__ == "__main__":
    app.run()