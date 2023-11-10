from .model import Model


class Observation(Model):
    def __init__(self, data, name="obs"):
        self.setup_serket(name=name, learnable=False)

        self.set_forward_msg(data)
