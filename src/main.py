import numpy as np
import time
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

from prepare.prepare_lin import PrepareDataLin


def eval_model(model, training, test, name, prepare):
    train_x = prepare.get_features(training)
    train_y = prepare.get_label(training)
    test_x = prepare.get_features(test)
    test_y = prepare.get_label(test)

    start_time = time.time()
    model.fit(train_x, train_y)
    print 'Training model %s in %.3f seconds.' % (name, time.time() - start_time)

    train_pred = model.predict(train_x)
    train_mse = mean_squared_error(train_y, train_pred)

    test_pred = model.predict(test_x)
    test_mse = mean_squared_error(test_y, test_pred)

    print '%s:\n\ttrain_mse: %f\n\ttest_mse: %f' % (name, train_mse, test_mse)

    return {'train_mse': train_mse, 'test_mse': test_mse}


prepare = PrepareDataLin()
training_data = prepare.get_training_data()
test_data = prepare.get_test_data()

regr = linear_model.LinearRegression(n_jobs=4)
mlp = MLPRegressor(solver='adam',
                   alpha=0.0001,
                   hidden_layer_sizes=(15, 15),
                   random_state=1,
                   activation="relu",
                   max_iter=500)


def loss_curve(model, param_name, param_values, title):
    def get_mse(v):
        model.set_params(**{param_name: v})
        return eval_model(model, training_data, test_data, title, prepare)

    result = map(get_mse, param_values)
    train_mses = map(lambda x: x['train_mse'], result)
    test_mses = map(lambda x: x['test_mse'], result)

    plt.plot(param_values, train_mses, 'b-', param_values, test_mses, 'g-')
    plt.xlabel(param_name)
    plt.ylabel('MSE')
    plt.title(title)
    plt.show()

# eval_model(regr, training_data, test_data, 'Linear', prepare)
# eval_model(mlp, training_data, test_data, 'MLP', prepare)

# alpha_values = range(1, 10, 2)
# alpha_values = map(lambda x: 0.0001 * x, alpha_values)
# loss_curve(mlp, 'alpha', alpha_values, 'MLP')

# iter_values = range(500, 1000, 100)
# loss_curve(mlp, 'max_iter', iter_values, 'MLP')

# 200 best
layer_values = range(50, 150, 10)
layer_values = map(lambda x: (x, x, x), layer_values)
loss_curve(mlp, 'hidden_layer_sizes', layer_values, 'MLP')

