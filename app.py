#import Flask 
from flask import Flask
import numpy as np
import joblib
from flask import Flask, render_template, request

#create an instance of Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    
    if request.method == "POST":
        #get form data
        Umur = request.form.get('Umur')
        Gaji = request.form.get('Gaji')
     
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(Umur, Gaji)
            #pass prediction to template
            return render_template('predict.html', prediction = prediction)
   
        except ValueError:
            return "Please Enter valid values"
  
        pass
    pass

def preprocessDataAndPredict(Umur, Gaji):
    
    #keep all inputs in array
    test_data = [Umur, Gaji]
    print(test_data)
    
    #convert value data into numpy array
    test_data = np.array(test_data)
    
    #reshape array
    test_data = test_data.reshape(1,-1)
    print(test_data)
    
    #open file
    file = open("knn_iklan_model.pkl","rb")
    
    #load trained model
    trained_model = joblib.load(file)
    
    #predict
    prediction = trained_model.predict(test_data)
    
    return prediction
    
    pass

if __name__ == '__main__':
    app.run(debug=True)