from flask import Flask,render_template,request
import pickle


app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')
@app.route('/display', methods=['GET','POST'])
def displayresults():
    review = request.form['review']
    vectoriser = pickle.load(open('/home/pragnyam/mysite/vectorizer', 'rb'))
    model = pickle.load(open('/home/pragnyam/mysite/NaiveBayesClassifier', 'rb'))
    X_test = vectoriser.transform([review])
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    return render_template('index.html',review = review,pred = pred, prob = prob)

