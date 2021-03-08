import pandas
from os import listdir
from logger import App_logger


class dataTransformationPredict:

    def __init__(self):
        self.goodDataPath = "Prediction_Raw_Files_Validated/Good_Raw"
        self.logger = App_logger()

    def replaceMissingWithNull(self):
        try:
            log_file = open("Prediction_Logs/dataTransformLog.txt", 'a+')
            onlyfiles = [f for f in listdir(self.goodDataPath)]
            for file in onlyfiles:
                csv = pandas.read_csv(self.goodDataPath + "/" + file)
                csv.fillna('NULL', inplace=True)
                csv.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                self.logger.log(log_file, " %s: File Transformed successfully!!" % file)

        except Exception as e:
            self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
            log_file.close()
            raise e
        log_file.close()
