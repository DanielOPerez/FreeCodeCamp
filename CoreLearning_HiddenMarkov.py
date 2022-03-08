import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf



tfd = tfp.distributions #for simplicity

#the first day has a 80% probability of beeing cold
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])

#there is a 70% probability for a cold day to be followed by a cold day
#there is a 80% probability for a hot day to be followed by a hot day
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])

#The temperature is normally distribued with mean 0 and std 5 for a cold day
# and mean 15 and std 10 por a hot day. loc==mean, scale==std
observation_distribution=tfd.Normal(loc=[0.0, 15.0], scale=[5.0, 10.0])


#Model creation using HIdden Markov
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=10)

#Mean values of the temperatures of the "num_steps" days given the probilities
print(model.mean().numpy())
