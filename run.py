import os
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.svm import OneClassSVM as OCSVM
from pandas import DataFrame
import numpy as np
import pandas
import pickle
import json

# datas = pandas.read_csv('./PCs900_LosAlamos.csv', sep=',', chunksize=30000)
# label = pandas.read_csv('./Labels.csv', sep=',')
# events = pandas.read_csv('./Events.csv', sep=',')
#
# label = label['Label']
# label = label.replace(['Normal'], 1)
# label = label.replace(['Malicious'], -1)
# # label.describe()
# j = 0
# test = 0
#
# for i in datas:
#     j = j + 1
#     #     if j == 5 : continue
#     if j == 3:
#         train = i
#     if j == 5:
#         test = i
#         break
#
# y = label[60000:90000]
# X = train
#
from werkzeug.utils import secure_filename

s=0
set= pandas.read_csv('./PCs900_LosAlamos.csv', sep=',', chunksize=2)
for i in set :
    if s == 0 :
        test_set = i
        break

ensemble = pickle.load(open('saved_model', 'rb'))

app = Flask(__name__)

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/train1', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file :
                # and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('./', filename))
            return 'OK'
    return 'OK'

@app.route('/')
def render_hello():
    return render_template('hello.html')

@app.route('/test')
def render_test():
    return render_template('test.html')

@app.route('/train')
def render_train():
    return render_template('train.html')

@app.route('/number/<number>')
def hello_name(number):
    global label
    return "Hello {}!".format(label[int(number)])


@app.route('/api/predict/event/<event_id>', methods=['GET', 'POST'])
def predict_event(event_id):
    global ensemble
    # content = request.json
    # raw_req = request.form['event']
    raw_req = request.files['file1']
    print(raw_req)
    req = json.loads(raw_req)
    # event = DataFrame([[content['Time'],content['SU'],content['DU'],content['SC'],content['DC'],content['AT'],content['LT'],content['AO'],content['SF']]], columns=['Time',	'SU',	'DU',	'SC',	'DC',	'AT',	'LT',	'AO','SF'])
    # event = DataFrame.from_dict(content)
    event = DataFrame.from_dict(req)    # , orient='index')
    # event.reset_index(level=0, inplace=True)
    # print('Value : '+req)
    print(event)
    new_set = test_set.append(event , ignore_index=True)
    print(new_set)
    preds = ensemble.predict(new_set)
    print(preds)
    if preds[-1]==-1:
        return jsonify({"event_id":event_id, "result":'Malicious'})
    else:
        return jsonify({"event_id":event_id, "result":'Normal'})

@app.route('/api/predict/events/', methods=['GET', 'POST'])
def predict_events():
    global ensemble
    content = request.form['event']
    # event = request.form['event']

    # event = DataFrame([[content['Time'],content['SU'],content['DU'],content['SC'],content['DC'],content['AT'],content['LT'],content['AO'],content['SF']]], columns=['Time',	'SU',	'DU',	'SC',	'DC',	'AT',	'LT',	'AO','SF'])
    event = pandas.read_json(content)
    # event = DataFrame.from_dict(content)
                                # , orient='index')
    new_set = test_set.append(event , ignore_index=True)
    # print(new_set)
    preds = ensemble.predict(new_set)
    preds = DataFrame(preds)
    preds = preds.replace(1.0,'Normal')
    preds = preds.replace(-1.0,'Malicious')
    # print(preds)
    preds = preds[2:]
    preds = preds.reset_index(drop=True)
    # print(preds)
    js = preds.to_json()
    # return jsonify(preds)
    return js

@app.route('/api/train/new2/', methods=['GET', 'POST'])
def train_new2():
    global ensemble
    content = request.json
    label = content['labels']
    event = content['events']
    label_df = DataFrame([[1,label]],columns=['Index','Label'] )
    # label_df = DataFrame.from_dict(label)
    event_df = DataFrame.from_dict(event)

    # print(event_df)
    # print(label_df)
    with open('events_new.csv', 'a') as f:
        event_df.to_csv(f, header=False) # True if first event
    with open('labels_new.csv', 'a') as f:
        label_df.to_csv(f, header=False) # True if first event
    # new_set = test_set.append(event , ignore_index=True)
    # preds = ensemble.predict(new_set)
    # preds = DataFrame(preds)
    # preds = preds.replace(1.0,'Normal')
    # preds = preds.replace(-1.0,'Malicious')
    # js = preds.to_json()
    # print(preds)
    return "OK"

@app.route('/api/train/new/', methods=['GET', 'POST'])
def train_new():
    global ensemble
    content_raw = request.form['json']
    content = pandas.read_json(content_raw)
    print(content)
    event = content['events']
    label = content['labels']
    # label_df = DataFrame([[1,label]],columns=['Index','Label'] )
    label_df = DataFrame.from_dict(label)
    event_df = DataFrame.from_dict(event)

    # print(event_df)
    # print(label_df)
    with open('events_new.csv', 'a') as f:
        event_df.to_csv(f, header=False) # True if first event
    with open('labels_new.csv', 'a') as f:
        label_df.to_csv(f, header=False) # True if first event
    # new_set = test_set.append(event , ignore_index=True)
    # preds = ensemble.predict(new_set)
    # preds = DataFrame(preds)
    # preds = preds.replace(1.0,'Normal')
    # preds = preds.replace(-1.0,'Malicious')
    # js = preds.to_json()
    # print(preds)
    return "Success"

@app.route('/api/train/all/', methods=['GET', 'POST'])
def train_all():
    global ensemble
    content = request.json

    # event = DataFrame([[content['Time'],content['SU'],content['DU'],content['SC'],content['DC'],content['AT'],content['LT'],content['AO'],content['SF']]], columns=['Time',	'SU',	'DU',	'SC',	'DC',	'AT',	'LT',	'AO','SF'])

    event = DataFrame.from_dict(content)
                                # , orient='index')
    # event.reset_index(level=0, inplace=True)

    new_set = test_set.append(event , ignore_index=True)
    # print(new_set)
    preds = ensemble.predict(new_set)
    preds = DataFrame(preds)
    preds = preds.replace(1.0,'Normal')
    preds = preds.replace(-1.0,'Malicious')
    js = preds.to_json()
    print(preds)
    # return jsonify(preds)
    return js


def load_data():
    datas = pandas.read_csv('./PCs900_LosAlamos.csv', sep=',', chunksize=30000)
    label = pandas.read_csv('./Labels.csv', sep=',')
    events = pandas.read_csv('./Events.csv', sep=',')

    label = label['Label']
    label = label.replace(['Normal'], 1)
    label= label.replace(['Malicious'], -1)
    # label.describe()
    j=0
    test = 0

    for i in datas :
        j=j+1
    #     if j == 5 : continue
        if j == 3 :
            train = i
        if j == 5 :
            test = i
            break

    return label, events, test, train

def train_model(ensemble, X, y) :
    seed = 2017
    np.random.seed(seed)


    # --- Build ---
    # Passing a scoring function will create cv scores during fitting
    # the scorer should be a simple function accepting to vectors and returning a scalar
    ensemble = SuperLearner(scorer=accuracy_score, random_state=seed, verbose=2)

    # Build the first layer
    # ensemble.add([RandomForestClassifier(random_state=seed), SVC()])

    ensemble.add([IsolationForest(), LOF(novelty=True)])


    # Attach the final meta estimator
    # ensemble.add_meta(LogisticRegression())

    ensemble.add_meta(OCSVM())

    # Fit ensemble
    ensemble.fit(X, y)

def test_model(ensemble, test, label):
    # Predict
    preds = ensemble.predict(test)

    # score = accuracy_score(preds, label[120000:150000])
    #
    # predictions = DataFrame(preds)
    # predictions[0].value_counts()
    # tn, fp, fn, tp = confusion_matrix(label[120000:150000],preds).ravel()
    # tn, fp, fn, tp = confusion_matrix(label[120000:150000],predictions[0]).ravel()
    # return preds, tn, fp, fn, tp
    return preds

def predict_event(ensemble, test):
    # Predict
    preds = ensemble.predict(test)

    # score = accuracy_score(preds, label[120000:150000])
    #
    # predictions = DataFrame(preds)
    # predictions[0].value_counts()
    # tn, fp, fn, tp = confusion_matrix(label[120000:150000],preds).ravel()
    # tn, fp, fn, tp = confusion_matrix(label[120000:150000],predictions[0]).ravel()
    # return preds, tn, fp, fn, tp
    return preds

def save_model(ensemble, name='saved_model'):
    with open(name,'wb') as f:
        pickle.dump(ensemble,f)

def load_model(name='saved_model'):
    import pickle
    loaded_model = pickle.load(open(name, 'rb'))
    ensemble = loaded_model
    return ensemble

    



if __name__ == '__main__':

    app.run()
