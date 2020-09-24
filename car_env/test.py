from pid_env import Env

env = Env()

for i in range(10):
    done = False
    state = env.reset()
    while not done:
        action = [0, 0]
        state, reward, done, info = env.step(action)
        env.render()