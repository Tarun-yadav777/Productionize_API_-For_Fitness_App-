from flask import Flask, render_template, request, Response, url_for
import pickle
import numpy as np
from prediction_Validation_Insertion import pred_validation
from predictionFromModel import Prediction
import json
from training_Validation_Insertion import train_validation
from trainingModel import trainModel

app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return render_template('home.html')


@app.route("/predicts", methods=['POST'])
def predict_for_single():
    age = request.form.get('age')
    weight = request.form.get('weight')
    duration = request.form.get('duration')
    heart_rate = request.form.get('heart_rate')
    body_temp = request.form.get('body_temp')
    height = request.form.get('height')
    gender = request.form.get('gender')

    with open('my_model.pkl', 'rb') as f:
        model = pickle.load(f)

    result = model.predict(np.array([[age, weight, duration, heart_rate, body_temp]]))

    result = result.tolist()[0]
    return f"Predicted Values is: {result}"


@app.route("/predictm", methods=['POST'])
def predict_for_multiple():
    try:
        if request.json is not None:
            path = request.json['filepath']
            pred_val = pred_validation(path)
            pred_val.prediction_validation()
            pred = Prediction(path)
            path, json_predictions = pred.predictFromModel()
            return Response("Prediction File created at !!!" + str(path) + 'and few of the predictions are ' + str(
                json.loads(json_predictions)))
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.route("/train", methods=["POST"])
def train_for_client():
    try:
        if request.json is not None:
            path = request.json['filepath']
            train_valObj = train_validation(path)

            train_valObj.train_validation()

            trainModelObj = trainModel()
            trainModelObj.trainingModel()


    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)

    return Response("Training successfull!!")


if __name__ == '__main__':
    app.run()
