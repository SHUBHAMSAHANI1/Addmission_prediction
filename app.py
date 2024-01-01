import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import PolynomialFeatures


app = Flask(__name__, static_url_path='/static', static_folder='static')
model = pickle.load(open('admission_prediction.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    int_features = int_features[1:]
    final_features = [np.array(int_features)]

    poly_converter = PolynomialFeatures(degree=2, include_bias=False)

    X_poly = poly_converter.fit_transform(final_features)
    prediction = model.predict(X_poly)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Chance of admit is {}%'.format(output*100))


if __name__ == "__main__":
    app.run(debug=True)
