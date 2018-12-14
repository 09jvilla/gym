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
NUM_PAST_STATE = 2
HOW_FAR = 10

startExplore = 1.0
endExplore = 0.1

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

def take_derivative_feature_extractor(state, action, weights):
    x0, y0, vx0, vy0, theta0, w0, touching_l_0, touching_r_0 = state[0:STATE_SIZE]
    x1, y1, vx1, vy1, theta1, w1, touching_l_1, touching_r_1 = state[STATE_SIZE:]
    Jr = weights['Jr']
    Jl_m = weights['Jl_m']
    Jl_s = weights['Jl_s']
    radius = weights['radius']
    mg = weights['mg']
    timestep= weights['timestep']
    #I want to see 10 steps ahead

    #second derivative terms
    a0x = vx1 -vx0
    a0y = vy1- vy0 +mg
    # a0 = a0x * a0y * math.cos(theta0) #check whether this is in radian

    alpha0 = w1 - w0

    # a1 = Jl * timestep + a0s
    if action == 2:
        Jx = Jl_m * math.cos(theta1)
        Jy = Jl_m * math.sin(theta1)
    elif action == 1 or action == 3:
        Jx = Jl_s * math.cos(theta1)
        Jy = Jl_s * math.sin(theta1)
    else: #if not doing anything
        Jx = 0
        Jy = 0
        Jr = 0

    # first derivative terms
    vx2 = .5 * Jx * timestep **2 + a0x * timestep + vx1 - w1 * radius
    vy2 = .5 * Jy * timestep **2 + (a0y-mg) * timestep + vy1 + w1 * radius

    # v2 = vx2 * vy2 * math.cos(theta0)

    w2 = 0.5 * Jr * timestep**2 + alpha0 * timestep + w1


    x2 = 1.5 * Jx * timestep **3 + .5 * a0x * timestep **2 + vx1 * timestep + x1 -w1* radius * timestep
    y2 = 1.5 * Jy * timestep **3 + .5 * a0y * timestep **2 + vy1 * timestep + y1 +w1* radius * timestep
    #
    theta2 = 1.5 * Jr * timestep **3 + .5 * alpha0 * timestep **2 + w1 * timestep + theta1

    # pdb.set_trace()
    return [x2, y2, vx2, vy2, theta2, w2]
    # pdb.set_trace()


class QLearningAlgorithm():
    def __init__(self, actions, discount, featureExtractor, repeatActions, explorationProb=0.3):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.weights['Jr'] = round (0.15 * (1.0 - random.random()),3)
        self.weights['Jl_m'] = round(1.5 * (1.0 - random.random()),3)
        self.weights['Jl_s'] = round(1.5 * (1.0 - random.random()),3)
        self.weights['timestep'] = 1
        self.weights['mg'] = round (0.5 * (1.0 - random.random()),3)
        self.numIters = 0
        self.explore_decay=0.005
        self.prevAction = -1
        self.repeatActions = repeatActions


    def decay_exploration(self):
        exploreval = endExplore + (startExplore - endExplore)*np.exp(-self.explore_decay*self.numIters)
        return exploreval

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        x1, y1, vx1, vy1, theta1, w1, touching_l_1, touching_r_1 = state[STATE_SIZE:]
        [x2, y2, vx2, vy2, theta2, w2] = take_derivative_feature_extractor(state, action, self.weights)
        # print('x2 :',x2, 'y2',  y2,'vx2', vx2, 'vy2:', vy2, 'theta2', theta2,'w2', w2)
        timestep = self.weights['timestep']
        radius = self.weights['radius']
        score = ((vy2-vy1)/(vy1+0.1))**2+ ((w2-w1)/(w1+0.1))**2 + (y2-y1)/(y1+0.1)+(theta2-theta1)/(theta1+0.1)

        # pdb.set_trace()
        # return score
        return [x1, x2, y1, y2, vy2, vy1, w2, w1, theta2, theta1, score]
                #0  1   2   3    4    5   6   7      8      9       10

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state,prr):
        state_segment = state[(NUM_PAST_STATE-1)*STATE_SIZE:]
        qvals=[]
        for action in self.actions(state_segment):
            result= self.getQ(state,action)
            print('Q values: action -- ', action)
            result = [round(result[x],2) for x in range(0,len(result))]

            print(result)
            print(self.weights)
            qvals.append(result[-1])
        # qvals = [ self.getQ(state, action)[-1] for action in self.actions(state_segment) ]
        action = qvals.index(min(qvals))
        # if prr:
        #     pdb.set_trace()
        # print(self.weights)
        # print(Qvals)
        # pdb.set_trace()
        action = [i for i, qval in enumerate(qvals) if qval == min(qvals)]
        if len(action)>1:
            chosen_action = 2
        else:
            chosen_action = action[0]
            # print(chosen_action)
        return chosen_action

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        # return 1.0 / math.sqrt(self.numIters)
        return 0.01

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState, is_done):
        x0, y0, vx0, vy0, theta0, w0, _, _ = state[0:STATE_SIZE]
        x1, y1, vx1, vy1, theta1, w1, _, _ = state[STATE_SIZE:]

        Jr = self.weights['Jr']
        Jl_m = self.weights['Jl_m']
        Jl_s = self.weights['Jl_s']
        radius = self.weights['radius']
        mg = self.weights['mg']
        timestep= self.weights['timestep']

        a0x = vx1 -vx0
        a0y = vy1- vy0
        # a0 = a0x * a0y * math.cos(theta0)
        a0 = math.sqrt(a0x**2 + a0y **2)
        alpha0 = w1-w0

        if action == 2:
            Jx = Jl_m * math.cos(theta0)
            Jy = Jl_m * math.sin(theta0)
            J = Jl_m
        elif action == 1 or action == 3:
            Jx = Jl_s * math.cos(theta0)
            Jy = Jl_s * math.sin(theta0)
            J=Jl_s
        else: #if not doing anything
            Jx = 0
            Jy = 0
            Jr = 0
            J = 0

        del_jx =.5 * Jx * timestep **4 + timestep **2 * (a0x * timestep - w1* radius)
        del_jy =.5 * Jy * timestep **4 + timestep **2 * (a0y * timestep + w1* radius)

        del_j = math.sqrt((Jx* del_jy)**2  +  (del_jx * Jy)**2)

        del_jr = 0.5 * Jr * timestep ** 4 + timestep **2 * (alpha0*timestep)

        # del_mg = 2*a

        v1 = math.sqrt(vx1**2 + vy1 **2)

        del_radius = (J *timestep **2 + 2*a0 * timestep) * w1+ 2*w1*radius

        del_mg = 2*a0y * timestep **2 + 2 * mg * timestep**2 - (Jy * timestep **2 + 2* y1 + 2*w1* radius-y1)
        print('del_mg:', del_mg)


        new_weights = self.weights.copy()
        if action ==2:
            new_weights['Jl_m'] = round(J - self.getStepSize() *del_j,2)
        if action ==1 or action ==3:
            new_weights['Jl_s'] = round(J - self.getStepSize() *del_j,2)
        new_weights['Jr'] = round(Jr - self.getStepSize() *del_jr,2)
        new_weights['radius'] = round(radius - self.getStepSize() *del_radius,2)
        new_weights['mg'] = round(mg - self.getStepSize() * del_mg,2)

        self.weights = new_weights.copy()
def printweights(rl):
    print(rl.weights)

def simulate_multistate( Lander, rl, numTrials, maxIters=10000, do_training=True, verbose=True, render = False):
    totalRewards = []
    # pdb.set_trace()
    if do_training:
        reward_ls = [0]* numTrials
    for trial in range(0,numTrials):
        # if trial > 4800:
             # render = True
        #basically puts us back in start state
        #reset returns the start statenewState_segment
        state_0 = Lander.reset()
        #Choose random actions cuz we don't have enough history
        action = 2 # we're sticking with the main engine, babe
        next_state, reward, _, _ = Lander.step(action)
        #we're going to base our prediction off of 10th state further
        # for i in range(0,HOW_FAR):
        state = np.concatenate(((state_0, next_state)), axis=None)

        totalDiscount = 1
        totalReward = reward
        # state_count= 10

        for iter in range(0,maxIters):
            #get action from QLearning Algo
            # state_segment = state[(NUM_PAST_STATE-1)*STATE_SIZE:]
            # action = rl.getAction(state_segment)
            prr = False
            if iter % 10 == 0:
                prr= True
            action = rl.getAction(state, prr)

            #simulate action
            #returns new state, reward, boolean indicating done and info
            nextState, reward, is_done, info = Lander.step(action)

            # print('=====new state: =====')
            # print(nextState)
            # state_count +=1
            new_state = np.concatenate((state[STATE_SIZE:], nextState), axis=None)

            if do_training:
                rl.incorporateFeedback(state, action, reward, new_state, is_done)
                reward_ls [trial]= totalReward
                # pdb.set_trace()


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
    trainRewards = simulate_multistate(myLander, myrl, numTrials, verbose=True, render=True)
    # trainRewards = simulate(myLander, myrl, numTrials, verbose=True, do_render = True)
    return myrl, trainRewards

def export_weights_sparse(weight_dict):
    filename = "sparse_weight_" + time.strftime("%m%d_%H%M") + ".pkl"
    output = open(filename, 'wb')
    pickle.dump(weight_dict, output)
    output.close()


def main(args):
    myLander = LunarLander()
    # myrl, trainRewards = train_QL( myLander, basicFeatureExtractor, args.repeatActions, numTrials=args.num_train_trials )
    myrl, trainRewards = train_QL( myLander, take_derivative_feature_extractor, args.repeatActions, numTrials=args.num_train_trials )
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
    # simulate(myLander, myrl, numTrials=args.num_test_trials, do_training=False, do_render=True)
    print('######Note: currently simulating multistate#######')
    simulate_multistate(myLander, myrl, numTrials=args.num_test_trials, do_training=False, render=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeatActions", default=1, type=int, help="Force simulation to repeat actions this number of times")
    parser.add_argument("--num_train_trials", default='5000', type=int, help="Number of simluations to train for")
    parser.add_argument("--num_test_trials", default='100', type=int, help="Number of simluations to test for")
    args = parser.parse_args()
    main(args)
