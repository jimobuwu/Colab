import numpy as np
import random

# reward or punishment
r = np.array([[-1, -1, -1, -1, 0, -1], [-1, -1, -1, 0, -1, 100], [-1, -1, -1, 0, -1, -1], [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, 100], [-1, 0, -1, -1, 0, 100]])

# q table
q = np.zeros([6,6], dtype=np.float32)

gamma = 0.8

step = 0
while step < 1000:
    step += 1
    state = random.randint(0, 4)
    next_state_list = []
    for i in range(6):
        if r[state, i] != -1:
            next_state_list.append(i)
    next_state = next_state_list[random.randint(0, len(next_state_list) -1)]
    # Q = r + gamma * maxQ
    qval = r[state, next_state] + gamma * max(q[next_state])
    q[state, next_state] = qval

for i in range(10):
    print("第{}次验证".format(i+1))
    state = random.randint(0, 5)
    print("机器人处于{}".format(state))
    count = 0
    while state != 5:
        if count > 20:
            print("fail")
            break
        q_max = q[state].max()

        q_max_action = []
        for act in range(6):
            if q[state, act] == q_max:
                q_max_action.append(act)

        next_state = q_max_action[random.randint(0, len(q_max_action) - 1)]
        print("robot go to state:{}".format(next_state))
        state = next_state
        count += 1

