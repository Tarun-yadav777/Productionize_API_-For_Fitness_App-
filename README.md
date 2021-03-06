# Productionize_API_-For_Fitness_App-
---

### Table of Contents
You're sections headers will be used to reference location of destination.

- [Overview](#overview)
- [How To Use](#how-to-use)
- [Tool Used](#tool-used)
- [Author Info](#author-info)

---

### Overview

This application provides the service for the prediction of Calories burned for the fitness app through API request.

The details of the prodictionized api is that there are three categories:-<br> 
1.Predict from model trained by me for a single records.<br>
2.Predict from model created by me for multiple records in flies.<br>
3.Train model on your data and get predictions.<br>


<h4>1.Predict from model trained by me for a single records</h4>:- Predict is done from the model which is created by me which give farely around 99.5% accuracy(XGBoost) on test data which is trained on over 15,000 records of data on multiple ML models and XGBoost outperforms every other model with 99.5% accuracy with 0.09 standard deviation.<br>
Demo Picture:-
<img src="https://github.com/Tarun-yadav777/Productionize_API_-For_Fitness_App-/blob/main/static/landing.PNG"  width="500" height="300" /><br>

<h4>A logger is also developed to veiw the Real Time process of the whole Prection process.</h4>
<img src="https://github.com/Tarun-yadav777/Productionize_API_-For_Fitness_App-/blob/main/static/prediction_logs.PNG" /><br>

<h4>2.Predict from model created by me for multiple records in flies</h4>:-Predict is done by sending the API request of the path of the Batch_files and follow the below pipeline:-<br>
                    a.Validate File Name Format<br>
                    b.Validate Format of Data<br>
                    c.Preprocess Data<br>
                    d.Send validated and processed data to DataBase<br>
                    e.Data is fetched from DataBase for Prediction<br>
                    f.Data is provided in Json Format<br>
Demo Picture:-
<img src="https://github.com/Tarun-yadav777/Productionize_API_-For_Fitness_App-/blob/main/static/predictm.PNG"  width="500" height="300" /><br>
                    
<h4>3.Train model on your data and get predictions</h4>:-Training is done by sending the API request of the path of the batch files and follow the below pipeline:-<br>
                    a.Validate File Name Format<br>
                    b.Validate Format of Data<br>
                    c.Preprocess Data<br>
                    d.Send validated and processed data to DataBase<br>
                    e.Data is fetched from DataBase for training<br>
                    f.Training is done on multiple model with Hyper-Parameter tuning<br>
                    g.Best Model is selected for prediction and saved<br>

<h4>A logger is also developed to veiw the Real Time process of the whole Training process.</h4>
<img src="https://github.com/Tarun-yadav777/Productionize_API_-For_Fitness_App-/blob/main/static/training_logs.PNG"  /><br>

Demo Picture:-<br>
<img src="https://github.com/Tarun-yadav777/Productionize_API_-For_Fitness_App-/blob/main/static/train.PNG"  width="500" height="300" /><br>
Demo Picture:-<br>
<img src="https://github.com/Tarun-yadav777/Productionize_API_-For_Fitness_App-/blob/main/static/model.PNG" /><br>

### How To Use
## How to run the project?

1. Clone this repository in your local system.
2. Install all the libraries mentioned in the [requirements.txt] file.
3. Replace the path file request option if you want to send request directly or Open Postman for testing API 
4. Open your terminal/command prompt from your project directory and run the `main.py` file by executing the command `python main.py`.
5. Go to your browser and type `http://127.0.0.1:5000/` in the address bar.
6. Hurray! That's it.


### References
1.Google<br>
2.Kaggle<br>
3.Youtube<br>
4.Stackoverflow<br>
5.iNeuron<br>

---

### Tool Used

![Python](https://img.shields.io/badge/Python-3.8-blueviolet)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![Frontend](https://img.shields.io/badge/Frontend-HTML/CSS/JS-green)


---

### Author's Info

- Twitter - [@taronic777](https://twitter.com/taronic777)
- linkedIn - [Tarun Yadav](https://www.linkedin.com/in/tarun-yadav-47442112b/)

[Back To The Top](#read-me-template)
