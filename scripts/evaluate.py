from environment.rocket_landing_env import RocketLandingEnv
from agent.dqn_agent import DQNAgent
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    'batch_size': 64,
    'gamma': 0.99,
    'epsilon_start': 0.0,      
    'epsilon_min': 0.0,
    'epsilon_decay': 1,  
    'learning_rate': 1e-3,
    'target_update': 10,
    'replay_buffer_capacity': 10000,
    'num_episodes': 10,
    'max_steps_per_episode': 500,
}

def main():
    env = RocketLandingEnv(render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize the agent
    agent = DQNAgent(state_size, action_size, params)

    # Load the trained model onto the device
    agent.policy_net.load_state_dict(torch.load('models/dqn_policy_net.pth', map_location=device))
    agent.policy_net.to(device)
    agent.policy_net.eval()

    num_episodes = params['num_episodes']

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = agent.policy_net(state_tensor)
                action = q_values.max(1)[1].item()

            next_state, reward, done, _ = env.step(action)
            state = next_state

            total_reward += reward

            env.render()

            if done:
                break

        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")
    average_reward = sum(total_rewards) / num_episodes
    print(f"Average Total Reward over {num_episodes} episodes: {average_reward}")
    env.close()

if __name__ == "__main__":
    main()
