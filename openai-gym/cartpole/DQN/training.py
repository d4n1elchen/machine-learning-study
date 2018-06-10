import gym
from ReinforcementLearn import DQN

env = gym.make("CartPole-v0")

# Print info about the env
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DQN(n_actions=env.action_space.n,
         n_features=env.observation_space.shape[0],
         learning_rate=0.01,
         e_greedy=0.9,
         replace_target_iter_max=1000,
         replace_target_iter_min=100,
         memory_size=200,
         batch_size=30,
         e_greedy_increment=0.001)

step = 0
episode = 0

while True:
    # Reset and get first observation
    observation = env.reset()

    epi_step = 0
    # Start learning until game finished
    while True:
        # Render the scene
        #env.render()

        # Choose action from observation
        action = RL.get_action(observation)

        # Execute the action
        observation_, reward, done, info = env.step(action)

        x, x_dot, theta, theta_dot = observation_
        r1 = (env.env.x_threshold - abs(x))/env.env.x_threshold
        r2 = (env.env.theta_threshold_radians - abs(theta))/env.env.theta_threshold_radians
        reward = r1 + r2

        # Store the transition
        RL.store_transition(observation, action, reward, observation_)

        # (Start after 200 step) and (learning frequency)
        if (step > 200) and (step % 5 == 0):
            RL.learn()

        if done:
            if epi_step > 50:
                print("Episode:{}: Failed. Servive step={}".format(episode, epi_step))
            break

        observation = observation_
        epi_step += 1
        step += 1

    episode += 1
