from lunar_lander import demo_heuristic_lander, LunarLander


total_reward_array = []

myLunarLander = LunarLander()

dorender = True
num_iters = 100
isdumb= True


for i in range(0,num_iters):
    end_reward = demo_heuristic_lander( myLunarLander, render=dorender, dumb=isdumb )
    total_reward_array.append(end_reward)
    myLunarLander.reset()
    print("Iteration: " + str(i))

print("Average Rewards Over " + str(num_iters) + " trials ")
average_reward = sum(total_reward_array) / len(total_reward_array)
print(average_reward)

