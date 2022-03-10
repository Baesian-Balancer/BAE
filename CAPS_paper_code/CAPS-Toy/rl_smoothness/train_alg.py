import abc

class TrainAlg(abc.ABC):
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def train(self, args):
        pass

    @abc.abstractmethod
    def add_args(self, parser):
        pass

def subclasses():
    # get all the subclasses of TrainAlg dynamically and instantiate them
    import rl_smoothness.algs as algs
    return list(alg() for alg in TrainAlg.__subclasses__())
