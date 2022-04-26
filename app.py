#Importing dependencies
from flask import Flask
from flask import render_template, request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

#Loading model
model = load_model('Saved_Model')

#class labels
Class_labels = ['Five', 'One', 'Ten', 'Two', 'Twenty']

#Prediction pipeline function
def model_predict(img_path):
    #Preprocessing
    img = Image.open(img_path) #Loading image
    img = img.resize((320,240)) #Resizing
    img = np.asarray(img)
    img = img/255.0 #Normalizing
    img = np.expand_dims(img,axis=0)
    #Taking prediction
    probs = model.predict(img)

    return probs

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index_page():
    return  render_template("index.html")

@app.route('/prediction', methods=['POST'])
def prediction():

    #Getting user image
    img = request.files['imagefile']
    img_path = 'static/' + img.filename
    img.save(img_path)

    #getting prediciton
    probs = model_predict(img_path)[0]
    pred_prob = '{:.2f}'.format(probs[np.argmax(probs)])
    pred = Class_labels[np.argmax(probs)]

    #get predictions
    return render_template('prediction.html', pred_prob=pred_prob, pred=pred, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)
