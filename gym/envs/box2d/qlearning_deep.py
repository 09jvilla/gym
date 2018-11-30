from lunar_lander import LunarLander
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
from keras import regularizers
from keras.optimizers import Adam

SAMPLE_SIZE=64
TRAIN_FREQ=4
LR=5e-4
EPS=1e-8


def build_model( input_size=8, num_actions=4 ):
 
    #tensor that takes state as input (8 elements)
    main_input = Input(shape=(input_size,), name='lander_state')
    #connect input to 4 neuron hidden later with relu activation
    
    model_type = 1
    if model_type == 0:
        x = Dense(4, activation='relu')(main_input)
    elif model_type == 1:
        x = Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0) , name='Dense1')(main_input)
        x = Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0) , name='Dense2' )(x)
    else:
        raise Exception('Undefined model type')
    
    #connect hidden layer to output layer with 4 neurons (to represent Q score of each of our 4 actions)
    #currently using relu activation -- not sure if that is the right choice
    all_q_estimates = Dense( num_actions, kernel_regularizer=regularizers.l2(0) , name='Q_est' )(x)

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
    
    train_name = "model_train"
    pred_name = "model_pred" 

    plot_model(model_train, to_file=train_name, show_shapes=True)
    plot_model(model_pred, to_file=pred_name, show_shapes=True)

    pdb.set_trace()
    #set up our training model with adam optimizer using MSE loss
    adopt = keras.optimizers.Adam(lr=LR, epsilon=EPS)
    model_train.compile(optimizer=adopt, loss='mean_squared_error' )

    return model_train, model_pred



class QLearningAlgorithm():
    def __init__(self, actions, discount, epochs, allow_explore=True):
        self.actions = actions
        self.discount = discount
        self.allow_explore = allow_explore
        self.numIters = 0
        self.explore_decay=0.0001 
        self.explore_start=0.3
        self.explore_stop=0.01
        self.epochs = epochs
        self.deep_net_train, self.deep_net_preds = build_model()

    def getQ(self, state ):

        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        return self.deep_net_preds.predict(state)

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state, nodecay=False):
        
        if not nodecay:
            self.numIters += 1
        
        randval = random.random()
        exploreval = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp(-self.explore_decay*self.numIters) 
       
        if (not self.allow_explore) or randval>exploreval:
            return np.argmax( self.getQ(state) )
        else:
            return random.choice(self.actions(state))

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
        #pdb.set_trace()
        his = self.deep_net_train.fit({'sim_state': x_train, 'action_mask': one_hot }, {'masked_loss': y_train}, batch_size=len(minibatch), epochs=self.epochs, verbose=False)

        return his.history['loss'][-1]

def simulate( Lander, rl, memD, numTrials, maxIters=600, do_training=True, verbose=True):
    totalRewards = []
    totalLoss = []
    loss = 0

    for trial in range(0,numTrials):

        #basically puts us back in start state
        #reset returns the start state
        state = Lander.reset()

        totalDiscount = 1
        totalReward = 0

        for _ in range(0,maxIters):
            #get action from QLearning Algo
            action = rl.getAction(state)

            #simulate action
            #returns new state, reward, boolean indicating done and info
            nextState, reward, is_done, info = Lander.step(action)
            
            #Lander.render()

            #Keep track of reward as I go, multiplying by discount factor each time
            totalReward += totalDiscount * reward
            totalDiscount *= Lander.discount()

	    #add to my memory buffer this most recent action next state pair
            #randomly sample 10% of memory size
	    #do rl on that
            if do_training:
                memD.append( (state, action, reward, nextState, is_done) )
                if _ % TRAIN_FREQ == 0 :
                    #pdb.set_trace()
                    memD_sample = random.sample(memD, SAMPLE_SIZE)
                    loss = rl.incorporateFeedback_toNet(memD_sample, verbose )
                
            if is_done:
                #this trial has ended so break out of it
                break

            #advance state
            state = nextState

        totalLoss.append(loss)
        if verbose:
            print("Trial %d (totalReward = %s)" % (trial, totalReward))
            print("Loss: " + str(loss))
        
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

    D = collections.deque(maxlen=memsize)
    state = Lander.reset()

    for i in range(0,SAMPLE_SIZE):
        #do random action
        action = rl.getAction(state, nodecay=True)
        nextState, reward, is_done, info = Lander.step(action)

        D.append( (state, action, reward, nextState, is_done) )
        
        if is_done:
            state = Lander.reset()
        else:
            state = nextState

    return D


def main(args):

    myLander = LunarLander()
    myLander.set_discount(args.discount)

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
    parser.add_argument("--num_epochs", default='100', type=int, help="Number of epochs that each train set will train for")
    parser.add_argument("--memsize", default='1000000', type=int, help="Size of memory buffer")
    parser.add_argument("--discount", default='1.0', type=float, help="Discount factor for rewards")
    args = parser.parse_args()
    main(args)
