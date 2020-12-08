import numpy as np



def run_game(env, ind, render, time_limit=2000, kill_limit = -10):
    fitness = 0
    observation = env.reset()
    neurons = None
    for t in range(time_limit):
        if render:
            env.render()
        neurons = ind.feed_sensor_values(observation, neurons)
        result = ind.extract_output_values(neurons)
        action = np.argmax(result)
        observation, reward, done, info = env.step(action)
        fitness += reward
        if done or fitness < kill_limit:
            break
    env.close()
    # print(f"{fitness}")
    return fitness