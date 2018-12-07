import pickle
from lunar_lander import LunarLander
import pdb
from qlearning_feat import QLearningAlgorithm, simulate, basicFeatureExtractor
import sys


def load_weights_from_pckl( pickle_file ):
    pickle_in = open(pickle_file, "rb")
    return pickle.load(pickle_in)

def test_lander(weight_dict, featureExtractor):
    newLander = LunarLander()
    myrl = QLearningAlgorithm(newLander.actions, newLander.discount, featureExtractor)
    myrl.weights = weight_dict
    myrl.explorationProb = 0.0

    simulate(newLander, myrl, numTrials=100, do_training=False, do_render=True)


pickle_filename = sys.argv[1]
if sys.argv[1] is None or sys.argv[1] == "":
    raise Exception("Need a weights file")

weight_dict = load_weights_from_pckl(pickle_filename)
test_lander(weight_dict, basicFeatureExtractor)
