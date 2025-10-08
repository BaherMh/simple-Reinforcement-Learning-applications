def learn_problem(env, agent, episodes, max_steps, need_render):
    scores = []

    for e in range(episodes):
        state, _ = env.reset()
        score = 0
        for i in range(max_steps):
            if need_render:
                env.render()
            action = agent.act(state)

            next_state, reward, done, _, _ = env.step(action)
            score += reward
            agent.memory.remember(state, action, reward, next_state, int(done))
            state = next_state
            agent.replay()
            if done:
                break
        agent.end_episode()
        print("episode: {}/{}, score: {}".format(e, episodes, score))
        scores.append(score)

    return scores


def result_learning(env, agent, max_steps):
    agent.epsilon = 0.0
    while True:
        state = env.reset()[0]
        score = 0
        for i in range(max_steps):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            score += reward
            state = next_state
            if done:
                break
        print("score: {}".format(score))
