#run_with_weight.py
from qlearning_for_dummer import *

def import_weights(file):
    with open(file,'rb') as fid:
        ax = pickle.load(fid)

    pdb.set_trace()
    return weights

def main():
    myLander = LunarLander()
    myLander.weights = import_weights('./output/sparse_weight_1202_1548.pkl')
    myrl = QLearningAlgorithm(myLander.actions, myLander.discount, improvedFeatureExtractor)
    simulate(myLander,myrl, 1000, verbose=True, render = True)
if __name__ == '__main__':
	main()
