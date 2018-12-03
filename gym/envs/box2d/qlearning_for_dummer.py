import time
from lunar_lander_dumb import LunarLander
import math, random
from collections import defaultdict
import numpy as np
import pdb
import matplotlib.pyplot as plt
from precision.to_precision import to_precision
import pickle
STATE_SIZE = 8
NUM_PAST_STATE = 3
def identityFeatureExtractor(state, action):
    #have to turn state into a list so its hashable
    featureKey = ( tuple(state.tolist()), action )
    featureValue = 1
    # pdb.set_trace()
    return [(featureKey, featureValue)]


def roundedFeatureExtractor(state, action):
	#lets round the velocities and positions

	rounded_list = []

	for i in range(len(state)):

		#the minus 2 is because we don't want to touch the lunar lander's leg info
		if i < (len(state)-2):
			# pdb.set_trace()

			#Used this before to always round to 2 decimal places but replaced with sigfig rounding
			#rounded_list.append(round( state[i], 2) )

			rounded_list.append( float(to_precision( state[i], 2, notation='std')) )
		else:
			rounded_list.append(state[i])

	#pdb.set_trace()
	roundedfeatureKey = ( tuple(rounded_list), action )
	featureValue = 1
	# pdb.set_trace()
	return [(roundedfeatureKey, featureValue)]

def improvedFeatureExtractor(state, action):
	#Create more sophisticated version of feature extractor
	#Returns a list of following features:
	# --1: segmented x and whether L/R engine is
	# 		(ex: segment 0 (on the left most corner) , left)
	#--2: Segmented y and whether the main engine is on
	# 		(ex: segment 0 (as high as it could be ), on)
	#--3: Direction of Angle, direction of angular velocity and whether L/R engine is on
	# 		(ex: (Left, clockwise, Right), (Right, counterclockwise, not on))
	#--4: Angle of lunar lander w.r.t the horizon and angular velocity
	# 		(ex: (15rad, 5 rad/s))
	#--5: Angle of lunar lander w.r.t the horizon and angular velocity too fast or not so fast
	# 		(ex: (Angle too big, angular velocity too fast))
	#--6:Angle of lunar lander w.r.t. the horizon and segmented x, segmented y
	# 		(ex: (12 rad, segment 10 (right most), segment 7(towards the bottom)))
	#--7: Indicator function of whether either of the legs are touching the ground, y segment, and whether the main engine is on
	#		(ex: (Touching, Not touching, segment 10, engine on),
	# 		(Not touching, not touching, segment 2, engine off))
	#######################################REF#################################
	# state = [
	# 	(pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
	# 	(pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
	# 	vel.x*(VIEWPORT_W/SCALE/2)/FPS,
	# 	vel.y*(VIEWPORT_H/SCALE/2)/FPS,
	# 	self.lander.angle,
	# 	20.0*self.lander.angularVelocity/FPS,
	# 	1.0 if self.legs[0].ground_contact else 0.0,
	# 	1.0 if self.legs[1].ground_contact else 0.0
	# 	]

	# Action 0,1,2,3 = Nop, fire left engine, main engine, right engine
	###########################################################################
	# Input parm parsing: 'giving meanings' to your states
	x, y, vx, vy, theta, w, touching_l, touching_r = state
	x_segment = int(((x+1)*10+1)/2)
	y_segment = int(((y)*10))
	#indicator function for engines
	#0= nothing is on, 1= fire left engine, 2= fire main engine, 3 = fire right engine
	engine_on_indicator = [0] * 4
	engine_on_indicator[action] = 1
	engine_on_indicator = tuple (engine_on_indicator)
	#Engine indicator related to angular velocity
	left_or_right_fired = True if action == 1 or 2 else False
	# theta CW or CCW ######### kinda need to confirm
	isClockwise, angleTooBig = False, False
	if math.cos(theta) > 0:  isClockwise = True
	# else:  print('theta checking triggered')
	if abs(theta) >0.5:
		angleTooBig = True
		# print('angle too big')
	# else:
		# print('angle not too big')

	# theta' CW or CCW ######### kinda need to confirm
	isMovingClockwise, angVelTooBig = False, False
	if w > 0: isMovingClockwise = True
	if abs(w) > 1 :
		# print('too fast')
		angVelTooBig = True
	# Is lunar lander touching or not touching?
	touching = False
	if touching_l >0.5 or touching_r >0.5:
		touching = True

	result = []

	result.append(((x_segment, engine_on_indicator), action)) #Case 1
	result.append(((y_segment, engine_on_indicator), action)) #Case 2
	result.append(((isClockwise, isMovingClockwise, engine_on_indicator), action)) #Case 3
	result.append(((theta, w), action)) #case 4
	result.append(((angleTooBig, angVelTooBig), action)) #case 5
	result.append(((theta, x_segment, y_segment), action)) #case 6
	result.append(((touching, y_segment, engine_on_indicator),action)) #case 7
	# pdb.set_trace()
	return result
	# pdb.set_trace()

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
			score += self.weights[f] * 1
		return score


    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
	def getAction(self, state):
		self.numIters += 1
		# pdb.set_trace()
		state_segment = state[(NUM_PAST_STATE-1)*STATE_SIZE:NUM_PAST_STATE*STATE_SIZE]
		if random.random() < self.explorationProb:
			return random.choice(self.actions(state_segment))
		else:
			return max((self.getQ(state, action), action) for action in self.actions(state_segment))[1]

    # Call this function to get the step size to update the weights.
	def getStepSize(self):
		return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
	def incorporateFeedback(self, state, action, reward, newState, is_done):
		# BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)

		# pdb.set_trace()
		newState_segment = newState[(NUM_PAST_STATE-1)*STATE_SIZE:NUM_PAST_STATE*STATE_SIZE]

		#calculate V_opt(s')
		if is_done:
			vopt_newstate = 0
		else:
			vopt_newstate = max(self.getQ(newState_segment, newactions) for newactions in self.actions(newState_segment))
			# pdb.set_trace()

		target = reward + self.discount()*vopt_newstate
		# pdb.set_trace()
		prediction = 0
		for i in range (0,NUM_PAST_STATE):
			state_segment = state[i*STATE_SIZE:(i+1)*STATE_SIZE]
			prediction += self.getQ(state_segment, action)
		# print(prediction)
		# pdb.set_trace()
		# 	for f, v in self.featureExtractor(state_segment, action):
        #
        #
		# prediction = self.getQ(state,action)
        # pdb.set_trace()

		features_sa = []

		for i in range (0,NUM_PAST_STATE):
			state_segment = state[i*STATE_SIZE:(i+1)*STATE_SIZE]

			features_sa.extend(self.featureExtractor( state_segment, action ))

		update_multiplier = self.getStepSize() * ( prediction - target )

		for i in range(0, len(features_sa)):
			fname = features_sa[i][0]
			# fval = features_sa[i][1]
			fval = 1
			self.weights[fname] = self.weights[fname] - update_multiplier*fval


def simulate( Lander, rl, numTrials, maxIters=10000, do_training=True, verbose=True, render = False):
	totalRewards = []
	if do_training:
		reward_ls = [0]* numTrials
	for trial in range(0,numTrials):

		#basically puts us back in start state
		#reset returns the start state
		state_0 = Lander.reset()
		#Choose random actions cuz we don't have enough history
		action = random.choice(rl.actions(state_0))
		# pdb.set_trace()
		state_1, reward, is_done, info = Lander.step(action)
		#
		action = random.choice(rl.actions(state_1))
		state_2, reward, is_done, info = Lander.step(action)

		state = np.concatenate(((state_0, state_1, state_2)), axis=None)

		totalDiscount = 1
		totalReward = 0

		for _ in range(0,maxIters):
			#get action from QLearning Algo
			action = rl.getAction(state_2)

			#simulate action
			#returns new state, reward, boolean indicating done and info
			nextState, reward, is_done, info = Lander.step(action)
			new_state = np.concatenate((state[STATE_SIZE:], nextState), axis=None)
			# pdb.set_trace()
			# time.sleep(.5)

			if do_training:
				rl.incorporateFeedback(state, action, reward, new_state, is_done)
				reward_ls [trial]= totalReward

			#Keep track of reward as I go, multiplying by discount factor each time
			totalReward += totalDiscount * reward
			totalDiscount *= Lander.discount()


			if render:
				still_open = Lander.render()
				if still_open == False: break

			if is_done:
				#this trial has ended so break out of it
				break

                        #advance state
			state = new_state
			state_2 = nextState

		if verbose:
			if totalReward > 150:
				print("Trial %d (totalReward = %s)" % (trial, totalReward))
			# if trial % 100 == 0:
			if True:
				print("Trial %d (totalReward = %s)" % (trial, totalReward))

		if numTrials <= 50000 or (trial % 100 == 0):
			totalRewards.append(totalReward)

	if verbose:
		print("Finished simulating.")
    #
    # """
    # ##plots
	# fig = plt.subplot(111)
	# x = np.linspace(0,numTrials)
	# plt.scatter(x, reward_ls)
	# with open(filename, 'wb') as fid:
    # 	pickle.dump(fig, fid)
    #
    # """
	if do_training:
		filename = "./output/trial_reward_list"+ time.strftime("%m%d_%H%M") + ".pkl"
		output = open(filename, 'wb')
		pickle.dump(reward_ls, output)
		output.close()

	return totalRewards

def train_QL( myLander, featureExtractor, numTrials=1000 ):
	myrl = QLearningAlgorithm(myLander.actions, myLander.discount, featureExtractor)
	trainRewards = simulate(myLander, myrl, numTrials, verbose=True, render=False)


	return myrl, trainRewards

def export_weights_sparse(weight_dict):
    filename = "./output/sparse_weight_" + time.strftime("%m%d_%H%M") + ".pkl"
    output = open(filename, 'wb')
    pickle.dump(weight_dict, output)
    output.close()

def main():
	myLander = LunarLander()
	myrl, trainRewards = train_QL( myLander, improvedFeatureExtractor, numTrials=50 )
	export_weights_sparse(myrl.weights)
	# myrl, trainRewards = train_QL( myLander, roundedFeatureExtractor, numTrials=500 )

	print("Training completed. Switching to testing.")

	# plt.plot(trainRewards)
	# plt.ylabel('trainingReward')
	# plt.xlabel('Trial No.')
	# plt.savefig("output/trainprogress"+ time.strftime("%m%d_%H%M") )
	# plt.show()

	#Now test trained model:
	myrl.explorationProb = 0
	#Can simulate from here:
	simulate(myLander, myrl, numTrials=100, do_training=False, render = False)


if __name__ == '__main__':
	main()
