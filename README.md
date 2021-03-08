# Productionize_API_-For_Fitness_App-
---

### Table of Contents
You're sections headers will be used to reference location of destination.

- [Demo](#demo)
- [Overview](#overview)
- [How To Use](#how-to-use)
- [Tool Used](#tool-used)
- [Author Info](#author-info)

---

### Demo

![GIF](./static/gif.gif)

---

### Overview

This application provides the service for the prediction of Calories burned for the fitness app through API request.

The details of the prodictionized api is that there are three categories:-<br> 
1.Predict from model trained by me for a single records.<br>
2.Predict from model created by me for multiple records in flies.<br>
3.Train model on your data and get predictions.<br>
<b>1.Predict from model trained by me for a single records</b>:- Predict is done from the model which is created by me which give farely around 99.5% accuracy(XGBoost) on test data which is trained on over 15,000 records of data on multiple ML models and XGBoost outperforms every other model with 99.5% accuracy with 0.09 standard deviation.
<b>2.Predict from model created by me for multiple records in flies</b>:-Predict is done by sending the API request of the path of the Batch_files and follow the below pipeline:-
                    a.Validate File Name Format
                    b.Validate Format of Data
                    c.Preprocess Data
                    d.Send validated and processed data to DataBase
                    e.Data is fetched from DataBase for Prediction
                    f.Data is provided in Json Format
<b>3.Train model on your data and get predictions</b>:-Training is done by sending the API request of the path of the batch files and follow the below pipeline:-
                    a.Validate File Name Format
                    b.Validate Format of Data
                    c.Preprocess Data
                    d.Send validated and processed data to DataBase
                    e.Data is fetched from DataBase for training
                    f.Training is done on multiple model with Hyper-Parameter tuning
                    g.Best Model is selected for prediction and saved

### How To Use
## How to run the project?

1. Clone this repository in your local system.
2. Install all the libraries mentioned in the [requirements.txt] file.
3. Replace the path file request option if you want to send request directly or Open Postman for testing API 
4. Open your terminal/command prompt from your project directory and run the `main.py` file by executing the command `python main.py`.
5. Go to your browser and type `http://127.0.0.1:5000/` in the address bar.
6. Hurray! That's it.


### References
1.Google
2.Kaggle
3.Youtube
4.Stackoverflow
5.iNeuron

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
