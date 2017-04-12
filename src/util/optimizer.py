class Optimizer:
    def __init__(self, model, params=None):
        self.model = model
        if params:
            self.model.set_params(**params)
        self.params = self.model.get_params()

        self.__chain = list()

    def step(self, name, values, skipped=False):
        if not skipped:
            self.__chain.append({
                'pname': name,
                'pvalues': values
            })
        return self

    def solve(self, evaluator):
        for param in self.__chain:
            self.model.set_params(**self.params)    # set previous best param
            results = [(evaluator(self.model.set_params(**{param['pname']: value})), value)
                       for value in param['pvalues']]
            results = sorted(results, lambda a, b: -1 if a[0] < b[0] else 1)

            print param['pname']
            for result in results:
                print result[1], ' : ', result[0]

            # update best params
            self.params[param['pname']] = results[0][1]
            score = results[0][0]

        return score
