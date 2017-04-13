import pandas as pd
from sklearn.preprocessing import LabelEncoder
from prepare import PrepareData


class PrepareDataLin(PrepareData):
    def get_test_data(self):
        test = pd.read_csv('dataset/prepared/test.csv')
        test = test.drop(['VARIETY', 'YEAR', 'FAMILY', 'LOCATION', 'CHECK', 'REPNO', 'CLASS_OF'], 1)
        test = test.drop(test.columns[[0]], axis=1)
        test = test.rename(columns={'YIELD': 'Y'})

        return {'x': PrepareData.get_feature(test), 'y': PrepareData.get_label(test)}

    def get_training_data(self):
        training = pd.read_csv('dataset/prepared/train.csv')
        training = training.drop(['VARIETY', 'YEAR', 'FAMILY', 'LOCATION', 'CHECK', 'REPNO', 'CLASS_OF'], 1)
        training = training.drop(training.columns[[0]], axis=1)
        training = training.rename(columns={'YIELD': 'Y'})

        # encode VARIETY
        # le = LabelEncoder()
        # le.fit(training['VARIETY'])
        # training['VARIETY'] = le.transform(training['VARIETY'])

        return {'x': PrepareData.get_feature(training), 'y': PrepareData.get_label(training)}

    def get_data(self):
        return self.get_training_data()
