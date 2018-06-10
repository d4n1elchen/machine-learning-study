import gym
env = gym.make('CartPole-v1')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
print(env.observation_space.high)
#> array([ 2.4,  inf,  0.20943951,  inf])
print(env.observation_space.low)
#> array([-2.4, -inf, -0.20943951, -inf])

x = env.action_space.sample()
assert env.action_space.contains(x)
assert env.action_space.n == 2
