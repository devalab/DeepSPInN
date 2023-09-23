#CODE TO MEASURE TIME EFFICIENCY OF ENVIRONMENT [DEPRECATED]
import time
from .environment import *
import numpy as np
import random
from tqdm import tqdm
import time 

stepTimes = np.array([])
validTimes = np.array([])
rewardTimes = np.array([])
for i in range(10):    
    x = Env([6, 0, 0, 0], np.zeros(1801))
    for j in tqdm(range(10), desc="Action iter"):
        timeStart = time.process_time()
        valid = x.state.valid_actions()
        validTimes = np.append(validTimes, time.process_time() - timeStart)
        if np.sum(valid) == 0:
            print('Nothing Valid Boi')
            break
        action = int(random.choice(np.argwhere(valid)))
        timeStart = time.process_time()
        x.step(action)
        stepTimes = np.append(stepTimes, time.process_time() - timeStart)
        print(x.state)
    print(x)
    timeStart = time.process_time()
    for _ in range(10):
        print("Terminal Reward: ", x.terminal_reward())
    print(x.terminal_reward())
    rewardTimes = np.append(rewardTimes, time.process_time() - timeStart)

print("Average Step time: ", np.average(stepTimes))
print("Average Reward Time:", np.average(rewardTimes))
print("Average Valid Action Time: ", np.average(validTimes))

import ipdb
# ipdb.set_trace()

# For the previous: environment:
# Average Step time:  0.0027921693296088486                                                                                                                                        
# Average Valid Action Time:  0.07053368033165826

# #After refactoring:
# Average Step time:  0.03225421156666663   #step time increased because terminal condition changed                                                                                                                                       
# Average Valid Action Time:  0.029156465826086966
