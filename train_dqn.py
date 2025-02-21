import numpy as np
import gymnasium as gym
from rl_agent_zoo.components.common import init_seed
from rl_agent_zoo.dqn import DqnAgent


def main():
    init_seed(7)
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DqnAgent(state_size, action_size)

    scores, episodes = [], []
    EPOCHS = 1000
    TARGET_SCORE = 260

    # wandb.init(
    #     project="pistar-lab-lunar-lander",
    #     config={
    #         "seed": 7,
    #         "state_size": state_size,
    #         "action_size": action_size,
    #         "discount_factor": agent.discount_factor,
    #         "learning_rate": agent.learning_rate,
    #         "epsilon": agent.epsilon,
    #         "epsilon_decay": agent.epsilon_decay,
    #         "epsilon_min": agent.epsilon_min,
    #         "batch_size": agent.batch_size,
    #         "train_start": agent.train_start,
    #         "target_update_period": agent.target_update_period,
    #         "epochs": EPOCHS,
    #         "target_score": TARGET_SCORE
    #     }
    # )

    for epoch in range(EPOCHS):
        done = False
        turncated = False
        score = 0

        state, _ = env.reset()
        while not(done or turncated):
            action = agent.get_action(state)
            next_state, reward, done, turncated, info = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            if len(agent.memory) >= agent.train_start:
                agent.train()
            score += reward
            state = next_state

            if agent.step_counter % 100 == 0:
                # wandb.log({"score": score, "epsilon": agent.epsilon, "step_counter": agent.step_counter})
                pass

            if done or turncated:
                agent.done()
                
                scores.append(score)
                episodes.append(epoch)
                avg_score = np.mean(scores[-min(30, len(scores)):])
                print(
                    f'episode:{epoch} '
                    f'score:{score:.3f}, '
                    f'avg_score:{avg_score:.3f}, '
                    f'epsilon:{agent.epsilon:.3f}, '
                    f'step_counter:{agent.step_counter}'
                )
                # wandb.log({"avg_score": avg_score})
    
        if avg_score > TARGET_SCORE:
            print(f"Solved in episode: {epoch + 1}")
            break
    
    def plot(scores, episodes):
        import matplotlib.pyplot as plt
        plt.plot(episodes, scores)
        plt.title('original DQN For LunarLander-v3')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.grid()
        plt.show()

    plot(scores, episodes)
    mean_score, std_score = agent.evaluate(num_episodes=5, render=True)
    print(f"Evaluated Result(Mean Score: {mean_score:.3f}, Std Score: {std_score:.3f})")
    # wandb.log({"mean_score": mean_score, "std_score": std_score})
    
if __name__ == "__main__":
    main()
        
        
    
