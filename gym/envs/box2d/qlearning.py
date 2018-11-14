from lunar_lander import LunarLander
import math, random
from collections import defaultdict
import pdb
import matplotlib.pyplot as plt
from precision.to_precision import to_precision

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

    # Return the Q function associated with the weights and features
	def getQ(self, state, action):
		score = 0
		for f, v in self.featureExtractor(state, action):
			score += self.weights[f] * v
		return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
	def getAction(self, state):
		self.numIters += 1
		if random.random() < self.explorationProb:
			return random.choice(self.actions(state))
		else:
			return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
	def getStepSize(self):
		return 1.0 / math.sqrt(self.numIters)

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
			vopt_newstate = max(self.getQ(newState, newactions) for newactions in self.actions(newState))

		target = reward + self.discount()*vopt_newstate
		prediction = self.getQ(state,action)

		features_sa = self.featureExtractor( state, action )
		update_multiplier = self.getStepSize() * ( prediction - target ) 

		for i in range(0, len(features_sa)):
			fname = features_sa[i][0]
			fval = features_sa[i][1]
			self.weights[fname] = self.weights[fname] - update_multiplier*fval


def simulate( Lander, rl, numTrials, maxIters=10000, do_training=True, verbose=True):
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

			if do_training:
				rl.incorporateFeedback(state, action, reward, nextState, is_done)

			#Keep track of reward as I go, multiplying by discount factor each time
			totalReward += totalDiscount * reward
			totalDiscount *= Lander.discount()

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
	trainRewards = simulate(myLander, myrl, numTrials)
	return myrl, trainRewards

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
