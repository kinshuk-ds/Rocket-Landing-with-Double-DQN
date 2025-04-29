import time
import random
import numpy as np
import json
from agent.dqn_agent import DoubleQAgent
from environment.rocket_landing_env import RocketLandingEnv
import pygame

LEARN_EVERY = 4

def train_agent(n_episodes=2000, load_latest_model=False):
    print(f"Training a DDQN agent on {n_episodes} episodes. Pretrained model = {load_latest_model}")
    env = RocketLandingEnv(render_mode=None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DoubleQAgent(state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_dec=0.995,
                         lr=0.001, mem_size=200000, batch_size=128, epsilon_end=0.01)

    if load_latest_model:
        agent.load_saved_model('ddqn_torch_model.h5')
        print('Loaded most recent: ddqn_torch_model.h5')

    scores = []
    eps_history = []
    start = time.time()
    top_episodes = []

    for i in range(n_episodes):
        seed = random.randint(0, int(1e6))
        state = env.reset(seed=seed)
        done = False
        score = 0
        steps = 0
        episode_actions = []

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.save(state, action, reward, next_state, done)

            episode_actions.append(action)

            state = next_state
            if steps > 0 and steps % LEARN_EVERY == 0:
                agent.learn()
            steps += 1
            score += reward

        if len(top_episodes) < 3:
            top_episodes.append((score, seed, episode_actions.copy()))
            top_episodes.sort(reverse=True, key=lambda x: x[0])  # Sort by score descending
        else:
            if score > top_episodes[-1][0]:
                top_episodes[-1] = (score, seed, episode_actions.copy())
                top_episodes.sort(reverse=True, key=lambda x: x[0])  # Sort again

        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[max(0, i - 100):(i + 1)])

        if (i + 1) % 10 == 0 and i > 0:
            elapsed_time = (time.time() - start) / 60
            expected_total_time = ((elapsed_time / (i + 1)) * n_episodes)
            print(f'Episode {i + 1} in {elapsed_time:.2f} min. Expected total time for {n_episodes} episodes: {expected_total_time:.0f} min. '
                  f'[{score:.2f}/{avg_score:.2f}]')

        if (i + 1) % 100 == 0 and i > 0:
            agent.save_model('ddqn_torch_model.h5')
            with open(f"ddqn_torch_dqn_scores_{int(time.time())}.json", "w") as fp:
                json.dump(scores, fp)
            with open(f"ddqn_torch_eps_history_{int(time.time())}.json", "w") as fp:
                json.dump(eps_history, fp)

    if top_episodes:
        print("\nRendering and saving the best 3 episodes:")
        for idx, (score, seed, actions) in enumerate(top_episodes):
            print(f"Episode {idx + 1} with score: {score:.2f}")
            render_episode(env, actions, seed, idx + 1)
    else:
        print("\nNo episodes were completed during training.")

    return agent

def animate_model(name):
    env = RocketLandingEnv(render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DoubleQAgent(state_size, action_size, gamma=0.99, epsilon=0.0, lr=0.0005,
                         mem_size=200000, batch_size=64, epsilon_end=0.01)
    agent.load_saved_model(name)
    state = env.reset()
    for _ in range(5):
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            env.render()
        state = env.reset()
    env.close()

from moviepy.editor import ImageSequenceClip

def render_episode(env, actions, seed, episode_num):
    env = RocketLandingEnv(render_mode='human')
    state = env.reset(seed=seed)
    done = False
    step = 0
    frames = []

    while not done and step < len(actions):
        action = actions[step]
        state, reward, done, info = env.step(action)
        env.render()

        # Capture the current frame from the Pygame display
        frame = pygame.surfarray.array3d(env.screen)
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(frame)

        time.sleep(0.033)  # Pause for 30 FPS
        step += 1

    env.close()

    # Save the frames as a video file using moviepy
    clip = ImageSequenceClip(frames, fps=30)
    video_filename = f"best_episode_{episode_num}.mp4"
    clip.write_videofile(video_filename, codec="libx264")
    print(f"Saved video: {video_filename}")


agent = train_agent(n_episodes=1000, load_latest_model=True)