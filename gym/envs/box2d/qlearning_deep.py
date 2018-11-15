from lunar_lander import LunarLander
import math, random
from collections import defaultdict
import pdb
import matplotlib.pyplot as plt
from precision.to_precision import to_precision
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import collections
import numpy as np


def build_model( input_size=8, num_actions=4 ):
	model = Sequential()
	model.add(Dense(4, input_dim=input_size))
	model.add(Activation('relu'))
	model.add(Dense(num_actions))
	model.add(Activation('relu'))

	model.compile(loss=keras.losses.mean_squared_error, \
					optimizer='adam', \
					metrics=['accuracy'])

	return model


def identityFeatureExtractor(state, action):
    #have to turn state into a list so its hashable
    featureKey = ( tuple(state.tolist()), action )
    featureValue = 1
    #pdb.set_trace()
    return [(featureKey, featureValue)]


def roundedFeatureExtractor(state, action):
	#lets round the velocities and positions

	rounded_list = []

	for i in range(len(state)):
		#the minus 2 is because we don't want to touch the lunar lander's leg info
		if i < (len(state)-2):
			#Used this before to always round to 2 decimal places but replaced with sigfig rounding
			#rounded_list.append(round( state[i], 2) )

			rounded_list.append( float(to_precision( state[i], 2, notation='std')) )
		else:
			rounded_list.append(state[i])

	#pdb.set_trace()
	roundedfeatureKey = ( tuple(rounded_list), action )
	featureValue = 1
	return [(roundedfeatureKey, featureValue)]

class QLearningAlgorithm():
	def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
		self.actions = actions
		self.discount = discount
		self.featureExtractor = featureExtractor
		self.explorationProb = explorationProb
		self.weights = defaultdict(float)
		self.numIters = 0
		self.deep_net = build_model()

    # Return the Q function associated with the weights and features
	"""def getQ(self, state, action):
		score = 0
		for f, v in self.featureExtractor(state, action):
			score += self.weights[f] * v
		return score
	"""

	def getQ(self, state ):

		if state.ndim == 1:
			state = np.expand_dims(state, axis=0)

		return self.deep_net.predict(state)

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
	def getAction(self, state):
		self.numIters += 1
		if random.random() < self.explorationProb:
			return random.choice(self.actions(state))
		else:
			return np.argmax( self.getQ(state)  )

    # Call this function to get the step size to update the weights.
	def getStepSize(self):
		return 1.0 / math.sqrt(self.numIters)

	def incorporateFeedback_toNet( self, minibatch ):
		#need to calculate our loss
		pdb.set_trace()
		x_train = np.array( [x[0] for x in minibatch] )
		y_train = np.array( [x[2] for x in minibatch] )

		#make a bunch of predictions from next states
		next_states = np.array( [x[3] for x in minibatch] )
		is_dones = np.logical_not(np.array( [x[4] for x in minibatch] ))

		next_stateQs = self.getQ(next_states)
		vopt_nextstate = np.max(next_stateQs, axis=1)
		y_train += vopt_nextstate*is_dones*self.discount()

		self.deep_net.fit(x=x_train, y=y_train, batch_size=len(minibatch))

		return



    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
	def incorporateFeedback(self, state, action, reward, newState, is_done):
		# BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)

		#pdb.set_trace()
		#calculate V_opt(s') 
		if is_done:
			vopt_newstate = 0
		else:
			#vopt_newstate = max(self.getQ(newState, newactions) for newactions in self.actions(newState))
			#since now we have a vector of Q value for all possible actions
			vopt_newstate = max( self.getQ( newState ) )

		target = reward + self.discount()*vopt_newstate
		prediction = self.getQ(state)

		features_sa = self.featureExtractor( state, action )
		update_multiplier = self.getStepSize() * ( prediction - target ) 

		for i in range(0, len(features_sa)):
			fname = features_sa[i][0]
			fval = features_sa[i][1]
			self.weights[fname] = self.weights[fname] - update_multiplier*fval


def simulate( Lander, rl, memD, numTrials, maxIters=10000, do_training=True, verbose=True):
	totalRewards = []

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

			#Keep track of reward as I go, multiplying by discount factor each time
			totalReward += totalDiscount * reward
			totalDiscount *= Lander.discount()

			#add to memD
			memD.append( (state, action, reward, nextState, is_done) )

			#do reinforcement learning
			#randomly sample 10% of memory size
			if do_training:
				memD_sample = random.sample(memD, round(len(memD)*0.1))
				rl.incorporateFeedback_toNet(memD_sample)


			if is_done:
				#this trial has ended so break out of it
				break

            #advance state
			state = nextState

		if verbose:
			print("Trial %d (totalReward = %s)" % (trial, totalReward))
		totalRewards.append(totalReward)

	if verbose:
		print("Finished simulating.")

	return totalRewards

def train_QL( myLander, featureExtractor, numTrials=1000 ):
	myrl = QLearningAlgorithm(myLander.actions, myLander.discount, featureExtractor)
	
	#init memory
	print("Init memory: ")
	memD = init_memory(myLander, myrl)



	trainRewards = simulate(myLander, myrl, memD, numTrials)
	return myrl, trainRewards


#it takes about 215 steps to complete the landing, so I'll round to 300
#lets be able to hold at least 10 such simulations
def init_memory(Lander, rl, memsize = 3000):
	prevexploreprob = rl.explorationProb 

	#lets just act randomly
	rl.explorationProb = 1.0

	D = collections.deque(maxlen=memsize)
	state = Lander.reset()

	for i in range(0,memsize):
		#do random action
		action = rl.getAction(state)
		nextState, reward, is_done, info = Lander.step(action)

		D.append( (state, action, reward, nextState, is_done) )
		
		if is_done:
			state = Lander.reset()
		else:
			state = nextState

	#reset this back to exploration prob
	rl.explorationProb = prevexploreprob
	return D



def main():

	myLander = LunarLander()
	myrl, trainRewards = train_QL( myLander, roundedFeatureExtractor, numTrials=10 )

	print("Training completed. Switching to testing.")

	plt.plot(trainRewards)
	plt.ylabel('trainingReward')
	plt.xlabel('Trial No.')
	plt.savefig('trainprogress.png')
	plt.show()

	#Now test trained model:
	myrl.explorationProb = 0
	#Can simulate from here:
	simulate(myLander, myrl, numTrials=100, do_training=False)


if __name__ == '__main__':
	main()
