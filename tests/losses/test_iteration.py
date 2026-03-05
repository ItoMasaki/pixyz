import torch
from pixyz.losses import IterativeLoss, Parameter, Expectation
from pixyz.distributions import Deterministic, Normal


class TestIterativeLoss:
    def test_print_latex(self):
        t_max = 3
        itr = IterativeLoss(Parameter('t'), max_iter=t_max, timestep_var='t')
        assert itr.loss_text == r"\sum_{t=0}^{" + str(t_max - 1) + "} t"

    def test_time_specific_step_loss(self):
        t_max = 3
        itr = IterativeLoss(Parameter('t'), max_iter=t_max, timestep_var='t')
        assert itr.eval() == sum(range(t_max))

    def test_input_var(self):
        q = Normal(var=['z'], cond_var=['x'], loc='x', scale=1)
        p = Normal(var=['y'], cond_var=['z'], loc='z', scale=1)
        e = Expectation(q, p.log_prob())
        assert set(e.input_var) == set(('x', 'y'))
        assert e.eval({'y': torch.zeros(1), 'x': torch.zeros(1)}).shape == torch.Size([1])

    def test_input_extra_var(self):
        q = Normal(var=['z'], cond_var=['x'], loc='x', scale=1)
        p = Normal(var=['y'], cond_var=['z'], loc='z', scale=1)
        e = Expectation(q, p.log_prob())
        assert set(e.eval({'y': torch.zeros(1), 'x': torch.zeros(1),
                           'w': torch.zeros(1)}, return_dict=True)[1]) == set(('w', 'x', 'y', 'z'))
        assert set(e.eval({'y': torch.zeros(1), 'x': torch.zeros(1),
                           'z': torch.zeros(1)}, return_dict=True)[1]) == set(('x', 'y', 'z'))

    def test_return_dict_keeps_updated_state(self):
        class StateTransition(Deterministic):
            def __init__(self):
                super().__init__(var=['h'], cond_var=['h_prev'])

            def forward(self, h_prev):
                return {'h': h_prev + 1}

        transition = StateTransition()
        step_loss = Expectation(transition, Parameter('h'))
        itr = IterativeLoss(step_loss=step_loss, max_iter=3, update_value={'h': 'h_prev'})
        _, return_dict = itr.eval({'h_prev': torch.tensor(0.0)}, return_dict=True)

        assert return_dict['h_prev'] == torch.tensor(3.0)
