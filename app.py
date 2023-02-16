-from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

model = pickle.load(open('alzheimer2.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['M/F']
    data2 = request.form['Age']
    data3 = request.form['EDUC']
    data4 = request.form['SES']
    data5 = request.form['MMSE']
    data6 = request.form['CDR']
    data7 = request.form['nWBV']
    data8 = request.form['ASF']
    data9 = request.form['eTIV']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9]])
    print(arr)
    pred = model.predict(arr)
    return render_template('result.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
