### NOTE: THIS PROGRAM HAVEN'T FINISHED YET. I can't get observation with no action, so how to simulate cart speed is a problem now.

import gym
import sys

def updateActions(vel):
    actions = [0] * 50
    vel_t = vel - 50
# Zero velocity
    if vel_t is 0:
        return actions

# Build action list
    act = 1 if vel_t < 0 else 2
    step = 50 // abs(vel_t)
    for i in range(0, 50, step):
        actions[i] = act
    return actions

def decodeAct(act):
    return act - 1

def main(argv):
    # vel = 50 if len(argv) == 1 else int(argv[1])
    # print(updateActions(vel))

    env = gym.make('CartPole-v1')
    for i_episode in range(20):
        observation = env.reset()
        t = 0
        actions = updateActions(60)
        print(actions)
        while True:
            env.render()
            print(observation)
            action = decodeAct(actions[t])
            observation, reward, done, info = env.step(action=None)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                # break
            t += 1

if __name__ == "__main__":
    main(sys.argv)
