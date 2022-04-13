# import modules to build RunBuilder helper classes
from itertools import product
from collections import namedtuple

# helper classes added to manage the hyperparams & training process
# read in the hyper params and return a Run namedtuple containing all the 
# combinations of hyper-parameters
class RunBuilder():
    @staticmethod
    def get_runs(params):
      """
      Takes the params dictionary which we define in the 'params' argument. 
      And takes every parameter with every other parameter.
      Args:
      --------
      - params
      Returns:
      --------
      - runs = gets runs containing the combination of hyper-parameters
      """
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
