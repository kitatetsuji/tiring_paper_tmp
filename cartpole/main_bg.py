import gym
from gym import wrappers
from agent import Agent
import numpy as np
import time
import matplotlib.pyplot as plt
from statistics import mean

def bins(clip_min, clip_max, num):
    """観測した状態を離散値にデジタル変換する"""
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


def digitize_state(observation, num_dizitized):
    """各値を離散値に変換"""
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_dizitized)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_dizitized)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_dizitized)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_dizitized))
    ]
    return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])


def train():
    n = 1
    for x in range(1, n+1):
        num_dizitized = 6 #分割数
        max_number_of_steps = 200 #1試行のstep数
        num_consecutive_iterations = 100 #学習完了評価に使用する平均試行回数
        num_episodes = 800 #総試行回数
        total_reward_vec = np.zeros(num_consecutive_iterations) #各試行の報酬を格納
        final_x = np.zeros((num_episodes, 1)) #学習後、各試行のt=200でのｘの位置を格納
        islearned = 0 #学習が終わったフラグ
        _log_step = [] #行動回数メモリ
        log_step = [] #行動回数メモリ
        env = gym.make('CartPole-v0')
        agent = Agent(num_dizitized**4)

        for episode in range(1, num_episodes + 1): #試行数分繰り返す
            observation = env.reset()
            state = digitize_state(observation, num_dizitized)
            pre_state = state
            action = agent.get_action(episode, state)
            episode_reward = 0
            a_cnt = 0

            for t in range(max_number_of_steps): #1試行のループ
                cart_pos,cart_v,pole_angle,pole_v = observation
                agent.learn_fh(state, pre_state, action, a_cnt) #bg_loop前半
                action = agent.get_action(episode, state) #Policyで行動決定
                a_cnt += 1 #行動回数カウント
                observation, reward, done, info = env.step(action)

                if done:
                    if t < 195:
                        reward = -200 #こけたら罰則
                    else:
                        reward = 1 #立ったまま終了時は罰則はなし
                else:
                    reward = 1 #各ステップで立ってたら報酬追加

                episode_reward += reward #報酬を追加
                next_state = digitize_state(observation, num_dizitized) #t+1での観測状態を、離散値に変換
                pre_state = state
                state = next_state
                agent.learn_sh(pre_state, state, a_cnt, action, reward, episode, cart_pos, cart_v, pole_angle, pole_v) #bg_loop後半

                if done:
                    print('%d Episode finished after %f time steps / mean %f' %
                        (episode, t + 1, total_reward_vec.mean()))
                    total_reward_vec = np.hstack((total_reward_vec[1:],
                                                episode_reward)) #報酬を記録
                    _log_step.append(a_cnt)
                    if episode % 50 == 0:
                        log_step.append(mean(_log_step))
                        _log_step = []
                    if islearned == 1: #学習終わってたら最終のx座標を格納
                        final_x[episode, 0] = observation[0]
                    break
        agent.writer(35)
        print(log_step)

        # 学習曲線を表示
        x = np.arange(1, num_episodes+ 1, 50)
        y = log_step
        plt.figure(figsize=(16, 8))
        plt.plot(x, y, label='Label')
        plt.legend()
        plt.title('Change the number of steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    train()