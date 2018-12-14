import argparse
import time
from lunarlander_2d import LunarLander
import math, random
from collections import defaultdict, deque
import pdb
import matplotlib.pyplot as plt
from precision.to_precision import to_precision
import pickle
import numpy as np
STATE_SIZE = 8
NUM_PAST_STATE = 3

startExplore = 1.0
endExplore = 0.1

def identityFeatureExtractor(state, action):
    #have to turn state into a list so its hashable
    featureKey = ( tuple(state.tolist()), action )
    featureValue = 1
    # pdb.set_trace()
    return [(featureKey, featureValue)]

def basicFeatureExtractor(state, action):

    x, y, vx, vy, theta, w, touching_l, touching_r = state
    featList = []

    y_segment = round(y,1)
    if y > 2:
        y_segment = 2

    ##add a key with y segment and action
    featureKey = (y_segment,action)
    featList.append( (featureKey,1) )

    ##add a key with y velocity and action
    featList.append( ( ("velocity",action), vy) )

    ##add a key with y velocity and y position and action
    vy_round = round(vy, 1)
    if vy_round > 2:
        vy_round = 2
    elif vy_round < -2:
        vy_round = -2

    featList.append( ( (y_segment,vy_round,action), 1)   )

    ##add a key with whether legs are touching
    is_touch = 0
    if touching_l > 0 or touching_r > 0:
        is_touch = 1

    featList.append( (("touching", action), is_touch) )

    return featList



def basic2DFeatureExtractor(state, action):

    x, y, vx, vy, theta, w, touching_l, touching_r = state
    featList = []

    y_segment = round(y,1)
    if y > 2:
        y_segment = 2

    x_segment = round(x,1)
    if x > 1:
        x_segment = 1
    if x < -1:
        x_segment = -1

    ##add a key with y segment and action
    featureKey = ( x_segment, y_segment, action)
    featList.append( (featureKey,1) )

    ##add a key with y velocity and action
    featList.append( ( ("y_velocity",action), vy) )
    featList.append( ( ("x_velocity",action), vx) )

    ##add a key with angle and action
    featList.append( (("angle", action), theta) )
    featList.append( (("angular_vel", action), w) )

    ##add a key with whether legs are touching
    is_touch = 0
    if touching_l > 0 or touching_r > 0:
        is_touch = 1

    featList.append( (("touching", action), is_touch) )

    return featList

def heuristics(feature_state):
    #output from the feature extractor
    x, y, vx, vy, _, theta, w, touching = feature_state
    x = x[0]
    y=y[0]
    vx = vx[0]
    vy = vy[0]
    theta = theta[0]
    touching = touching[0]

    est_score = 0
    est_score+= 200*(abs(vy)+0.1)/(y+0.1)
    # if y <55 and abs(vy)>0:
    #     est_score -= y/(vy)**2 * 200
    # if vy==0 :
    #     est_score += 200/(y+0.1)
    # if abs(15-x) <2:
        # est_score -= abs(theta)* 200/ (y+0.1)
    # pdb.set_trace()
    # est_score = x+y #temp
    return est_score

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
    #######################################REF#################################
    # state = [
    #     (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
    #     (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
    #     vel.x*(VIEWPORT_W/SCALE/2)/FPS,
    #     vel.y*(VIEWPORT_H/SCALE/2)/FPS,
    #     self.lander.angle,
    #     20.0*self.lander.angularVelocity/FPS,
    #     1.0 if self.legs[0].ground_contact else 0.0,
    #     1.0 if self.legs[1].ground_contact else 0.0
    #     ]

    # Action 0,1,2,3 = Nop, fire left engine, main engine, right engine
    ###########################################################################
    # Input parm parsing: 'giving meanings' to your states
    x, y, vx, vy, theta, w, touching_l, touching_r = state
    x_segment = int(((x+1)*30+1)/2)
    y_segment = int(y*50) #expand & segment 0.....~1.5
    vx_segment = int((vx * 10) +20 /40)
    vy_segment= int( (vy *20) + 40 // 80)




    # theta CW or CCW ######### kinda need to confirm
    isClockwise, angleTooBig = False, False
    if math.cos(theta) > 0:  isClockwise = True
    # else:  print('theta checking triggered')

    theta_deg = int(math.degrees(theta))
    w_deg = int(w /0.01745)

    touching = False
    if touching_l >0.5 or touching_r >0.5:
        touching = True

    result = []

    result.append((x_segment, 1)) #Case 1
    result.append((y_segment, 1)) #Case 2
    result.append((vx_segment,1))
    result.append((vy_segment,1))
    result.append((isClockwise, 1)) #Case 3
    # result.append((isMovingClockwise, 1)) #Case 4
    result.append((theta_deg, 1)) #case 5
    result.append((w_deg, 1)) #case 6
    # result.append((angleTooBig, 1)) #case 7
    # result.append((angVelTooBig, 1)) #case8
    result.append((touching, 1)) #case 9
    # pdb.set_trace()
    return result
    # pdb.set_trace()

class QLearningAlgorithm():
    def __init__(self, actions, discount, featureExtractor, repeatActions, explorationProb=0.3):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0
        self.explore_decay=0.005
        self.prevAction = -1
        self.repeatActions = repeatActions
        # pdb.set_trace()

    def decay_exploration(self):
        exploreval = endExplore + (startExplore - endExplore)*np.exp(-self.explore_decay*self.numIters)
        return exploreval

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):

        score = 0

        phi = self.featureExtractor(state, action)
        # print(phi)
        est_score = heuristics(phi)
        for f, v in phi:
            score += self.weights[f] * v

        # print('score: ', score, 'heuristics score: ', est_score)
        # if est_score>score:
            # print('heuristics')
        # return est_score
        # return max(score, est_score)
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):

        self.numIters += 1

        exval = self.explorationProb

        if False:
            exval = self.decay_exploration()

        #force same action for repeat actions times straight
        if self.numIters % self.repeatActions != 0 and self.prevAction != -1:
            return self.prevAction

        # pdb.set_trace()
        if random.random() < exval:
            action = random.choice(self.actions(state))
            # print('random action')
        else:
            # state_segment = state[2*STATE_SIZE:]
            # action = self.actions(state_segment)
            # print('=====your features=====')
            Qvals = [ (self.getQ(state, action),action) for action in self.actions(state) ]
            bestScore = max( Qvals )[0]
            bestActs = [Qvals[index][1] for index in range(len(Qvals)) if Qvals[index][0] == bestScore]
            action = random.choice(bestActs)
            # print('bestScore: ', bestScore, 'action -->', action)

        self.prevAction = action
        return action

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        # return 1.0 / math.sqrt(self.numIters)
        return 0.01

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
            # pdb.set_trace()
        else:
            vopt_newstate = max(self.getQ(newState, newactions) for newactions in self.actions(newState))

        target = reward + self.discount()*vopt_newstate
        prediction = self.getQ(state,action)

        features_sa = self.featureExtractor( state, action )
        update_multiplier = self.getStepSize() * ( prediction - target )
        pdb.set_trace()
        new_weights = self.weights.copy()
        for i in range(0, len(features_sa)):
            fname = features_sa[i][0]
            fval = features_sa[i][1]
            new_weights[fname] = new_weights[fname] - update_multiplier*fval
        self.weights =  new_weights.copy()

    def incorporateFeedback_multistate(self, state, action, reward, newState, is_done):
        # pdb.set_trace()
        newState_segment = newState[(NUM_PAST_STATE-1)*STATE_SIZE:NUM_PAST_STATE*STATE_SIZE]

        #calculate V_opt(s')
        if is_done:
            vopt_newstate = 0
            # pdb.set_trace()
        else:
            # print('========new features: ===========')
            vopt_newstate = max(self.getQ(newState_segment, newactions) for newactions in self.actions(newState_segment))
            # print('choose : ', vopt_newstate)
            # pdb.set_trace()

        target = reward + self.discount()*vopt_newstate
        # pdb.set_trace()
        prediction = 0
        # print('======prediction======')
        for i in range (0,NUM_PAST_STATE):
            state_segment = state[i*STATE_SIZE:(i+1)*STATE_SIZE]
            prediction += self.getQ(state_segment, action)

        features_sa = []

        for i in range (0,NUM_PAST_STATE):
            state_segment = state[i*STATE_SIZE:(i+1)*STATE_SIZE]

            features_sa.extend(self.featureExtractor( state_segment, action ))

        update_multiplier = self.getStepSize() * ( prediction - target )

        for i in range(0, len(features_sa)):
            fname = features_sa[i][0]
            # fval = features_sa[i][1]
            pdb.set_trace()
            fval = 1
            self.weights[fname] = self.weights[fname] - update_multiplier*fval


def simulate( Lander, rl, numTrials, maxIters=5000, do_training=True, verbose=True, do_render=False):
    totalRewards = []
    prevRewards = deque(maxlen=10)
    # pdb.set_trace()
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

            if do_render:
                Lander.render()

            if do_training:
                rl.incorporateFeedback(state, action, reward, nextState, is_done)
                # pdb.set_trace()


            #Keep track of reward as I go, multiplying by discount factor each time
            totalReward += totalDiscount * reward
            totalDiscount *= Lander.discount()

            if is_done:
                #this trial has ended so break out of it
                break

                        #advance state
            state = nextState

        if verbose:
            if trial % 1 == 0:
                print("Trial %d (totalReward = %s)" % (trial, totalReward))

        if numTrials <= 50000 or (trial % 100 == 0):
            totalRewards.append(totalReward)

        if do_training:
            prevRewards.append(totalReward)
            past_avg_reward = sum(prevRewards) / len(prevRewards)
            if past_avg_reward >= 200:
                print("Solved game so breaking out of training loop!")
                break

    if verbose:
        print("Finished simulating.")


    return totalRewards

def simulate_multistate( Lander, rl, numTrials, maxIters=10000, do_training=True, verbose=True, render = False):
    totalRewards = []
    # pdb.set_trace()
    if do_training:
        reward_ls = [0]* numTrials
    for trial in range(0,numTrials):
        # if trial > 4800:
             # render = True
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
        state_count= 3

        for _ in range(0,maxIters):
            #get action from QLearning Algo
            state_segment = state[2*STATE_SIZE:]
            action = rl.getAction(state_segment)

            #simulate action
            #returns new state, reward, boolean indicating done and info
            nextState, reward, is_done, info = Lander.step(action)

            # print('=====new state: =====')
            # print(nextState)
            state_count +=1
            new_state = np.concatenate((state[STATE_SIZE:], nextState), axis=None)

            if do_training:
                rl.incorporateFeedback_multistate(state, action, reward, new_state, is_done)
                reward_ls [trial]= totalReward
                pdb.set_trace()


			#Keep track of reward as I go, multiplying by discount factor each time
            totalReward += totalDiscount * reward
            totalDiscount *= Lander.discount()

            if render:
                still_open = Lander.render()
                if still_open == False: break
            if is_done:
                # print('total states: ', state_count)
                state_count = 0
                # if trial > 2500:
                    # pdb.set_trace()
                break

            state = new_state

        if verbose:
            if totalReward > -10:
                print("Trial %d (totalReward = %s)" % (trial, totalReward))
			# if trial % 100 == 0:
            # if True:
                # print("Trial %d (totalReward = %s)" % (trial, totalReward))

        if numTrials <= 50000 or (trial % 100 == 0):
            totalRewards.append(totalReward)

    if verbose:
        print("Finished simulating.")

    if do_training:
        filename = "./output/trial_reward_list"+ time.strftime("%m%d_%H%M") + ".pkl"
        output = open(filename, 'wb')
        pickle.dump(reward_ls, output)
        output.close()

    return reward_ls

def train_QL( myLander, featureExtractor, repeatActions, numTrials ):
    myrl = QLearningAlgorithm(myLander.actions, myLander.discount, featureExtractor, repeatActions)
    print('######Note: currently simulating multistate#######')
    print('######using ', featureExtractor)
    myrl.explorationProb = .3
    # trainRewards = simulate_multistate(myLander, myrl, numTrials, verbose=True, render=True)
    trainRewards = simulate(myLander, myrl, numTrials, verbose=True, do_render = False)
    return myrl, trainRewards

def export_weights_sparse(weight_dict):
    filename = "sparse_weight_" + time.strftime("%m%d_%H%M") + ".pkl"
    output = open(filename, 'wb')
    pickle.dump(weight_dict, output)
    output.close()


def main(args):
    myLander = LunarLander()
    # myrl, trainRewards = train_QL( myLander, basicFeatureExtractor, args.repeatActions, numTrials=args.num_train_trials )
    myrl, trainRewards = train_QL( myLander, improvedFeatureExtractor, args.repeatActions, numTrials=args.num_train_trials )
    # myrl, trainRewards = train_QL( myLander, basic2DFeatureExtractor, args.repeatActions, numTrials=1000 )
    # myrl, trainRewards = train_QL( myLander, roundedFeatureExtractor, numTrials=500 )

    export_weights_sparse(myrl.weights)

    print("Training completed. Switching to testing.")

    plt.plot(trainRewards)
    plt.ylabel('trainingReward')
    plt.xlabel('Trial No.')
    plt.savefig("output/trainprogress"+ time.strftime("%m%d_%H%M") )
    plt.show()

    #Now test trained model:
    myrl.explorationProb = 0
    #Can simulate from here:
    simulate(myLander, myrl, numTrials=args.num_test_trials, do_training=False, do_render=False)
    print('######Note: currently simulating multistate#######')
    # simulate_multistate(myLander, myrl, numTrials=args.num_test_trials, do_training=False, render=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeatActions", default=1, type=int, help="Force simulation to repeat actions this number of times")
    parser.add_argument("--num_train_trials", default='5000', type=int, help="Number of simluations to train for")
    parser.add_argument("--num_test_trials", default='100', type=int, help="Number of simluations to test for")
    args = parser.parse_args()
    main(args)
