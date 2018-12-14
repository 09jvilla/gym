from lunarlander_2d import LunarLander
import math, random
from collections import defaultdict
import pdb
import matplotlib.pyplot as plt
from precision.to_precision import to_precision
import keras
from keras.layers import Dense, Activation, Multiply
from keras.layers import Input, Embedding, LSTM, Dense, Lambda
from keras.models import Model
import collections
import numpy as np
from keras.callbacks import History
from keras.utils import plot_model
import argparse
STATE_SIZE = 8
NUM_PAST_STATE = 3

#input_size = 8, breaks at incorporate feedback
  # File "qlearning_deep_multistate.py", line 95, in incorporateFeedback_toNet
  #   next_stateQs = self.getQ(next_states)
  # File "qlearning_deep_multistate.py", line 70, in getQ
  #   return self.deep_net_preds.predict(state)

def build_model( input_size= STATE_SIZE * NUM_PAST_STATE, num_actions=4 ):

    #tensor that takes state as input (8 elements)
    main_input = Input(shape=(input_size,), name='sim_state')
    #connect input to 4 neuron hidden later with relu activation
    x = Dense(4, activation='relu')(main_input)
    #connect hidden layer to output layer with 4 neurons (to represent Q score of each of our 4 actions)
    #currently using relu activation -- not sure if that is the right choice
    all_q_estimates = Dense(num_actions, activation='relu')(x)

    #tensor that takes 4 element, 1 hot vector as input
    #this is a mask vector - there will be a 1 set for the action taken, 0's elsewhere
    auxiliary_input = Input(shape=(num_actions,), name='action_mask')

    #multiply the output of the network by the mask so we only get the Q of the action we care about
    do_mask = Multiply()([all_q_estimates, auxiliary_input])
    #sum up all elements in masked vector (should just give you the Q of action we care about)
    #have to use a "Lambda" layer since there's no keras layer for this - LAME
    masked_loss = Lambda(lambda x: keras.backend.sum(x), output_shape=(1,), name='masked_loss' )(do_mask)

    #model for prediction - gives you q values for all actions
    model_pred = Model(inputs=[main_input], outputs=[all_q_estimates])
    #model for training - gives you q value only for action that ends up getting taken
    model_train = Model(inputs=[main_input, auxiliary_input], outputs=[masked_loss])

    plot_model(model_train, to_file='model_train', show_shapes=True)
    plot_model(model_pred, to_file='model_pred', show_shapes=True)


    #set up our training model with adam optimizer using MSE loss
    model_train.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

    return model_train, model_pred

class QLearningAlgorithm():
    def __init__(self, actions, discount, epochs, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.numIters = 0
        self.epochs = epochs
        self.deep_net_train, self.deep_net_preds = build_model()

        # input size = 24, breaks at getQ
        #   File "qlearning_deep_multistate.py", line 75, in getAction
        #     return np.argmax( self.getQ(state)  )
        #   File "qlearning_deep_multistate.py", line 65, in getQ
        #     return self.deep_net_preds.predict(state)
    def getQ(self, state ):
        # pdb.set_trace()
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        return self.deep_net_preds.predict(state)

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return np.argmax( self.getQ(state)  )

    def incorporateFeedback_toNet( self, minibatch , verbose ):
        #numpy array from initial state
        x_train = np.array( [x[0] for x in minibatch] )
        #starting to create our "target"
        #begin with numpy array of reward values
        y_train = np.array( [x[2] for x in minibatch] )

        #numpy array of next states
        next_states = np.array( [x[3] for x in minibatch] )
        #numpy array of is_dones. 1 if not done, 0 if done
        is_dones = np.logical_not(np.array( [x[4] for x in minibatch] ))
        #get Q values for each of the next states
        next_stateQs = self.getQ(next_states)
        #Get the max Q value from that
        vopt_nextstate = np.max(next_stateQs, axis=1)
        #add to target the vopt_nextstate*reward. remember vopt_nextstate is 0 if
        #the state we were at was endgame. (i.e. all you want is the final reward in this case)
        y_train += vopt_nextstate*is_dones*self.discount()

        ##Create an action mask from actions taken
        #This has the index of the action taken
        taken_actions = np.array( [x[1] for x in minibatch])
        #now make this into a 1 hot vector of size 4, since 4 possible actions
        one_hot = np.zeros((len(minibatch), 4))
        one_hot[np.arange(len(minibatch)), taken_actions] = 1

        #train the model. give it as inputs the state and the action mask (so you only look at the Q value for the action you took
        #then you want it to approximate the "target" value, our y_train
        his = self.deep_net_train.fit({'sim_state': x_train, 'action_mask': one_hot }, {'masked_loss': y_train}, batch_size=len(minibatch), epochs=self.epochs, verbose=False)

        return his.history['loss']

def simulate( Lander, rl, memD, numTrials, maxIters=10000, do_training=True, verbose=True):
    totalRewards = []
    totalLoss = []
    loss_list = []

    for trial in range(0,numTrials):

        #basically puts us back in start state
        #reset returns the start state
        state_0 = Lander.reset()

        # action = rl.getAction(state_0)
        # pdb.set_trace()
        #Choose random actions cuz we don't have enough history
        action = random.choice(rl.actions(state_0))
        state_1, reward, is_done, info = Lander.step(action)
        #
        action = random.choice(rl.actions(state_1))
        state_2, reward, is_done, info = Lander.step(action)
<<<<<<< HEAD
        #
=======
>>>>>>> 0b81cd8d1d942fecf668e891c6dbfc13fdc7cba8

        #pdb.set_trace()
        state = np.concatenate(((state_0, state_1, state_2)), axis=None)
<<<<<<< HEAD
=======
       
>>>>>>> 0b81cd8d1d942fecf668e891c6dbfc13fdc7cba8

        totalDiscount = 1
        totalReward = 0

        for _ in range(0,maxIters):
            # pdb.set_trace()
            #get action from QLearning Algo
            action = rl.getAction(state)
            #simulate action
            #returns new state, reward, boolean indicating done and info
            next_state, reward, is_done, info = Lander.step(action)
            new_state = np.concatenate((state[STATE_SIZE:], next_state), axis=None)

            #Keep track of reward as I go, multiplying by discount factor each time
            totalReward += totalDiscount * reward
            totalDiscount *= Lander.discount()

	    #add to my memory buffer this most recent action next state pair
            #randomly sample 10% of memory size
	    #do rl on that
            if do_training:
                #pdb.set_trace()
                memD.append( (state, action, reward, new_state, is_done) )
                memD_sample = random.sample(memD, round(len(memD)*0.1))
                loss_list = rl.incorporateFeedback_toNet(memD_sample, verbose )

            if is_done:
                #this trial has ended so break out of it
                break

            #advance state
            state = new_state

        totalLoss.extend(loss_list)
        if verbose:
            print("Trial %d (totalReward = %s)" % (trial, totalReward))
            print("Loss: " + str(loss_list))

        totalRewards.append(totalReward)

    if verbose:
        print("Finished simulating.")

    return totalRewards, totalLoss

def train_QL( myLander, numTrials, numEpochs,  memsize ):
    myrl = QLearningAlgorithm(myLander.actions, myLander.discount , numEpochs)

    #init memory
    print("Init memory: ")
    memD = init_memory(myLander, myrl, memsize)

    trainRewards, totalLoss = simulate(myLander, myrl, memD, numTrials)
    return myrl, trainRewards, totalLoss


#it takes about 215 steps to complete the landing, so I'll round to 300
#lets be able to hold at least 10 such simulations
def init_memory(Lander, rl, memsize ):
    prevexploreprob = rl.explorationProb

    #lets just act randomly
    rl.explorationProb = 1.0

    D = collections.deque(maxlen=memsize)
    frames_added = 0

    while(frames_added < memsize ):
        state_0 = Lander.reset()
    
        action = rl.getAction(state_0)
        state_1, reward, is_done, info = Lander.step(action)

        action = rl.getAction(state_1)
        state_2, reward, is_done, info = Lander.step(action)

        state = np.concatenate(((state_0, state_1, state_2)), axis=None)
        nextState = state
    
        while(True):

            action = rl.getAction(state_2)
            state_3, reward, is_done, info = Lander.step(action)

            nextState = np.concatenate((state[STATE_SIZE:], state_3), axis=None)

            D.append( (state, action, reward, nextState, is_done) )
            frames_added += 1

            if is_done:
                break
            else:
                state = nextState

    #reset this back to exploration prob
    rl.explorationProb = prevexploreprob
    return D


def main(args):

    myLander = LunarLander()
    myrl, trainRewards, totalLoss = train_QL( myLander, numTrials=args.num_train_trials, numEpochs=args.num_epochs, memsize=args.memsize )

    print("Training completed. Switching to testing.")

    plt.plot(trainRewards)
    plt.ylabel('trainingReward')
    plt.xlabel('Trial No.')
    plt.savefig('plots/trainprogress_memSz' + str(args.memsize) + '_epochs' + str(args.num_epochs) + '.png')
    #plt.show()

    plt.clf()
    plt.plot(totalLoss)
    plt.savefig('plots/loss_v_time_memSz_' + str(args.memsize) + '_epochs' + str(args.num_epochs) + '.png')
    #plt.show()

    #Now test trained model:
    myrl.explorationProb = 0
    #Can simulate from here:
    simulate(myLander, myrl, memD=None,  numTrials=args.num_test_trials, do_training=False, verbose=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_trials", default='1000', type=int, help="Number of simluations to train for")
    parser.add_argument("--num_test_trials", default='100', type=int, help="Number of simluations to test for")
    parser.add_argument("--num_epochs", default='1', type=int, help="Number of epochs that each train set will train for")
    parser.add_argument("--memsize", default='500', type=int, help="Size of memory buffer")
    args = parser.parse_args()
    main(args)
