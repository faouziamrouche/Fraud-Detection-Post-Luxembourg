# GALOIS-Python
ML tool for Post Luxembourg

## Author 
Faouzi Amrouche, SNT (University of Luxembourg)

## How to 
#### Link for the features dataset :
https://www.icloud.com/iclouddrive/04MM6IalLZmSPjLqgZfF82Guw#PCs900%5FLosAlamos

#### Install requirements :
pip install -r requirements.txt

#### Launch API : 
python run.py

#### Test API :
#### 1. With Postman :
1. Install Postman : https://www.getpostman.com
2. Format events with json : Use the test file "test_single_events.json" or "test_multiple_events.json"
3. Send Post request using Postman to : http://127.0.0.1:5000/api/predict/events

#### 1. With Web Client :
1. Launch the browser and go to http://127.0.0.1:5000/
2. Choose Test to predict events and paste the content of "test_single_events.json" or "test_multiple_events.json" in the textfield.
3. Choose Train to add training data and paste the content of "train_multiple_events.json" and "train_multiple_labels.json" in their respective textfields.
4. Click the "Submit" button : the result is returned as a json in case of test.



