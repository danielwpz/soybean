import time
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

from prepare.prepare_lin import PrepareDataLin
from prepare.prepare_wang import PrepareDataWang
from util.optimizer import Optimizer


def eval_model(model, train_x, train_y, test_x, test_y, name, prepare):
    start_time = time.time()
    model.fit(train_x, train_y)
    print 'Training model %s in %.3f seconds.' % (name, time.time() - start_time)

    train_pred = model.predict(train_x)
    train_mse = mean_squared_error(train_y, train_pred)

    test_pred = model.predict(test_x)
    test_mse = mean_squared_error(test_y, test_pred)

    print '%s:\n\ttrain_mse: %f\n\ttest_mse: %f' % (name, train_mse, test_mse)

    return {'train_mse': train_mse, 'test_mse': test_mse}


def loss_curve(model, param_name, param_values, title):
    def get_mse(v):
        model.set_params(**{param_name: v})
        return eval_model(model, train_x, train_y, test_x, test_y, title, prepare)

    result = map(get_mse, param_values)
    train_mses = map(lambda x: x['train_mse'], result)
    test_mses = map(lambda x: x['test_mse'], result)

    plt.plot(param_values, train_mses, 'b-', param_values, test_mses, 'g-')
    plt.xlabel(param_name)
    plt.ylabel('MSE')
    plt.title(title)
    plt.show()


prepare = PrepareDataWang()
all_data = prepare.get_data()
test_size = int(0.2 * len(all_data['x']))

train_x = all_data['x'][test_size:]
train_y = all_data['y'][test_size:]
test_x = all_data['x'][:test_size]
test_y = all_data['y'][:test_size]

print 'Data size: %d training, %d testing' % (len(train_x), len(test_x))

regr = linear_model.LinearRegression(n_jobs=4)
mlp = MLPRegressor(solver='adam',
                   alpha=0.0001,
                   hidden_layer_sizes=(15, 15),
                   random_state=1,
                   activation="relu",
                   max_iter=500)

eval_model(regr, train_x, train_y, test_x, test_y, 'Linear', prepare)

mlp_optimizer = Optimizer(mlp)

alpha_values = [0.0001, 0.0003, 0.001, 0.003]
layer_values = [(5, 5), (10, 10), (15, 15), (20, 20)]
# layer_values = [(15, 15, 15)]

loss_curve(mlp, 'hidden_layer_sizes', layer_values, 'MLP')

def get_score(model):
    model.fit(train_x, train_y)

    test_pred = model.predict(test_x)
    test_mse = mean_squared_error(test_y, test_pred)

    return test_mse


# score = mlp_optimizer.step('alpha', alpha_values).step('hidden_layer_sizes', layer_values).solve(get_score)
# print 'BEST score: ', score
# print 'BEST: ', mlp_optimizer.params



# eval_model(regr, training_data, test_data, 'Linear', prepare)
# eval_model(mlp, training_data, test_data, 'MLP', prepare)

# alpha_values = range(1, 10, 2)
# alpha_values = map(lambda x: 0.0001 * x, alpha_values)
# loss_curve(mlp, 'alpha', alpha_values, 'MLP')

# iter_values = range(500, 1000, 100)
# loss_curve(mlp, 'max_iter', iter_values, 'MLP')

# 200 best
(200, 200, 200)
# layer_values = range(50, 150, 10)
# layer_values = map(lambda x: (x, x, x), layer_values)
# loss_curve(mlp, 'hidden_layer_sizes', layer_values, 'MLP')
