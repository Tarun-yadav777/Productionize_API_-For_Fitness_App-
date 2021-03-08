from logger import App_logger
from predictionDataValidation import Prediction_data_validation
from data_loader_prediction import Data_Getter_Prediction
from preprocessing import Preprocessor
from file_methods import File_operation
import pandas as pd


class Prediction:
    def __init__(self, path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = App_logger()
        if path is not None:
            self.pred_data_val = Prediction_data_validation(path)

    def predictFromModel(self):
        try:
            self.pred_data_val.deletePredictionFile()
            self.log_writer.log(self.file_object, 'Start of Prediction')
            data_getter = Data_Getter_Prediction(self.file_object, self.log_writer)
            data = data_getter.get_data()
            preprocessor = Preprocessor(self.file_object, self.log_writer)
            is_null_present = preprocessor.is_null_present(data)
            if (is_null_present):
                data = preprocessor.impute_missing_values(data)

            cols_to_drop = preprocessor.get_columns_with_zero_std_deviation(data)
            data = preprocessor.remove_columns(data, cols_to_drop)

            file_loader = File_operation(self.file_object, self.log_writer)
            model = file_loader.load_model('my_model')

            X, y = preprocessor.separate_label_feature(data, 'Calories')
            result = list(model.predict(X.values))
            result = pd.Series(result, name='Predictions')
            path = "Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv", header=True, mode='a+')
            self.log_writer.log(self.file_object, 'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex

        return path, result.head().to_json(orient="records")

