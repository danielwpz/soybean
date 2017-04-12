class PrepareData:

    def __init__(self):
        pass

    def get_training_data(self):
        pass

    def get_test_data(self):
        pass

    def get_data(self):
        pass

    @staticmethod
    def get_feature(data):
        return data.drop(['Y'], 1).as_matrix()

    @staticmethod
    def get_label(data):
        return data['Y'].as_matrix()
