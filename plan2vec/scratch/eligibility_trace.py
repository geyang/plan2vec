import numpy as np
from params_proto.neo_proto import ParamsProto


class Args(ParamsProto):
    n_states = 10
    alpha = 1e-4
    gamma = 0.99


state_values = np.zeros(n_states)  # initial guess = 0 value
eligibility = np.zeros(n_states)

lamb = 0.95  # the lambda weighting factor
state = env.reset()  # start the environment, get the initial state
# Run the algorithm for some episodes
for t in range(n_steps):
    # act according to policy
    action = policy(state)
    new_state, reward, done = env.step(action)
    # Update eligibilities
    eligibility *= lamb * Args.gamma
    eligibility[state] += 1.0

    # get the td-error and update every state's value estimate
    # according to their eligibilities.
    td_error = reward + Args.gamma * state_values[new_state] - state_values[state]

    # this is the learning step
    state_values = state_values + alpha * td_error * eligibility

    if done:
        state = env.reset()
    else:
        state = new_state

if __name__ == '__main__':
    print('done')
