from src.util.optimizer import Optimizer


class FakeModel:
    def __init__(self):
        self.score = 10

    def set_params(self, **kwargs):
        return self

    def get_params(self):
        return dict()

    def get_score(self):
        self.score -= 1
        return self.score

def test_optimizer():
    o = Optimizer(FakeModel())
    o.step('depth', [1, 2, 3]).step('height', [3, 4, 5]).solve(lambda m: m.get_score())
    params = o.params
    assert params['depth'] == 3
    assert params['height'] == 5
