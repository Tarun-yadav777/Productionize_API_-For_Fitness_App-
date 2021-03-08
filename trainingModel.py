from sklearn.model_selection import train_test_split
from data_loader_training import Data_Getter
from preprocessing import Preprocessor
from tuner import Model_Finder
from file_methods import File_operation
from logger import App_logger



class trainModel:

    def __init__(self):
        self.log_writer = App_logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):

        self.log_writer.log(self.file_object, 'Start of Training')
        try:

            data_getter=Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()
            preprocessor=Preprocessor(self.file_object,self.log_writer)
            X,Y=preprocessor.separate_label_feature(data,label_column_name='Calories')
            is_null_present=preprocessor.is_null_present(X)
            if(is_null_present):
                X=preprocessor.impute_missing_values(X)
            cols_to_drop=preprocessor.get_columns_with_zero_std_deviation(X)
            X=preprocessor.remove_columns(X,cols_to_drop)

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 / 3,
                                                                random_state=355)
            model_finder = Model_Finder(self.file_object, self.log_writer)
            best_model_name, best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test)
            file_op = File_operation(self.file_object, self.log_writer)
            save_model = file_op.save_model(best_model, best_model_name)



            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception:

            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception