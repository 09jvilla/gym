from lunar_lander import LunarLander
import math, random
from collections import defaultdict
import pdb

def identityFeatureExtractor(state, action):
    #have to turn state into a list so its hashable
    featureKey = ( tuple(state.tolist()), action )
    featureValue = 1
    return [(featureKey, featureValue)]

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


def simulate( Lander, rl, numTrials, maxIters=10000, verbose=True):
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

def train_QL( myLander, featureExtractor, numTrials=1000 ):
	myrl = QLearningAlgorithm(myLander.actions, myLander.discount, featureExtractor)
	simulate(myLander, myrl, numTrials)

def main():
	myLander = LunarLander()
	train_QL( myLander, identityFeatureExtractor )

if __name__ == '__main__':
	main()
