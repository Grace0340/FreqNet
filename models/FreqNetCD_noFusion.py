from models.FreqNetCD import Model as _Base


class Model(_Base):
    def __init__(self, configs):
        configs.freqnet_adaptive_fusion = False
        super().__init__(configs)
