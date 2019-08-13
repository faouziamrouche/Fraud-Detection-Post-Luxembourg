import os
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, send_from_directory
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
from werkzeug.utils import secure_filename


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

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024


s=0
set= pandas.read_csv('./PCs900_LosAlamos.csv', sep=',', chunksize=2)
for i in set :
    if s == 0 :
        test_set = i
        break

ensemble = pickle.load(open('saved_model', 'rb'))

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def render_hello():
    return render_template('hello.html')

@app.route('/test')
def test_launch():
    return render_template('test.html')

@app.route('/train/add')
def train_add():
    return render_template('add training.html')

@app.route('/train/launch')
def train_launch():
    return render_template('launch training.html')

@app.route('/train/download_model')
def train_download_model():
    return send_from_directory('./','./saved_model')

@app.route('/api/predict/event/<event_id>', methods=['GET', 'POST'])
def predict_event(event_id):
    global ensemble
    #content = request.json
    raw_req = request.form['event']
    #raw_req = request.files['file1']
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

@app.route('/api/predict/events/', methods=['PUT','GET', 'POST'])
def predict_events():
    global ensemble
    if request.method == 'POST':
        if 'event' in request.form:
            content = request.form['event']
            print(request.form)
            event = pandas.read_json(content)
            new_set = test_set.append(event , ignore_index=True)
            preds = ensemble.predict(new_set)
            preds = DataFrame(preds)
            preds = preds.replace(1.0,'Normal')
            preds = preds.replace(-1.0,'Malicious')
            preds = preds[2:]
            preds = preds.reset_index(drop=True)
            js = preds.to_json()
            return js
        else:
            if 'file' not in request.files:
                #flash('No file part')
                return redirect('/')
            else:
                file1 = request.files['file']
                file = file1.read()
                req = json.loads(file)
                event = DataFrame.from_dict(req)  # , orient='index')
                new_set = test_set.append(event, ignore_index=True)
                preds = ensemble.predict(new_set)
                preds = DataFrame(preds)
                preds = preds.replace(1.0, 'Normal')
                preds = preds.replace(-1.0, 'Malicious')
                preds = preds[2:]
                preds = preds.reset_index(drop=True)
                # print(preds)
                js = preds.to_json()
                filename = secure_filename(file1.filename)
                file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                #flash('File successfully uploaded')
                return js
    else:
        return jsonify(isError=False,
                       message="Success",
                       statusCode=200), 200

@app.route('/api/train/new/', methods=['GET', 'POST'])
def train_new():
    global ensemble
    if 'events' in request.form:
        events_raw = request.form['events']
        labels_raw = request.form['labels']
        events = pandas.read_json(events_raw)
        labels = pandas.read_json(labels_raw)
        event_df = test_set.append(events , ignore_index=True)
        event_df = event_df[2:]
        event_df = event_df.reset_index(drop=True)
        print(labels)
        print(event_df)

        #label = content['labels']

        # label_df = DataFrame([[1,label]],columns=['Index','Label'] )
        #label_df = DataFrame.from_dict(label)
        #event_df = DataFrame.from_dict(event)
        #event_df['index'] = event_df.index
        #event_df = event_df.pivot_table(columns='index')
        #event_df=event_df['events']
        #print(event_df)
        #event_df = event_df.reset_index(drop=True)
        with open('events_new.csv', 'a') as f:
            event_df.to_csv(f, header=True, index=False) # True if first event

        with open('labels_new.csv', 'a') as f:
            labels.to_csv(f, header=True, index=False) # True if first event
        # new_set = test_set.append(event , ignore_index=True)
        # preds = ensemble.predict(new_set)
        # preds = DataFrame(preds)
        # preds = preds.replace(1.0,'Normal')
        # preds = preds.replace(-1.0,'Malicious')
        # js = preds.to_json()
        # print(preds)
        return render_template("success.html")
    else:
        return render_template("success.html")

@app.route('/api/train/new2/', methods=['GET', 'POST'])
def train_new2():
    global ensemble
    content = request.json
    label = content['labels']
    event = content['events']
    label_df = DataFrame([[1,label]],columns=['Index','Label'] )
    event_df = DataFrame.from_dict(event)

    with open('events_new.csv', 'a') as f:
        event_df.to_csv(f, header=False) # True if first event
    with open('labels_new.csv', 'a') as f:
        label_df.to_csv(f, header=False) # True if first event

    return "OK"

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
