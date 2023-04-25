# coding:utf-8
# [0]ライブラリのインポート
from operator import ne
import gym  #倒立振子(cartpole)の実行環境
from gym import wrappers  #gymの画像保存
import numpy as np
import time
import csv
from statistics import mean
 
memory = []
 
# [1]Q関数を離散化して定義する関数　------------
# 観測した状態を離散値にデジタル変換する
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]
 
# 各値を離散値に変換
def digitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_dizitized)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_dizitized)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_dizitized)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_dizitized))
    ]
    return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])
 
 
# [2]行動a(t)を求める関数 -------------------------------------
def get_action(next_state, episode):
           #徐々に最適行動のみをとる、ε-greedy法
    epsilon = 0.5 * (1 / (episode + 1e-8))
    if np.random.random() < epsilon:
        next_action = np.random.choice([0, 1])
    else:
        next_action = np.argmax(q_table[next_state])
    return next_action
 
 
# [3]Qテーブルを更新する関数 -------------------------------------
def update_Qtable(q_table, state, action, reward, next_state):
    gamma = 0.9
    alpha = 0.1
    next_Max_Q=max(q_table[next_state][0],q_table[next_state][1] )
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * next_Max_Q)
   
    return q_table
 
# [4]. メイン関数開始 パラメータ設定--------------------------------------------------------
env = gym.make('CartPole-v0')
max_number_of_steps = 200  #1試行のstep数
num_consecutive_iterations = 100  #学習完了評価に使用する平均試行回数
num_episodes = 800  #総試行回数
goal_average_reward = 195  #この報酬を超えると学習終了（中心への制御なし）
# 状態を6分割^（4変数）にデジタル変換してQ関数（表）を作成
num_dizitized = 6  #分割数
q_table = np.zeros((num_dizitized**4, env.action_space.n))
 
total_reward_vec = np.zeros(num_consecutive_iterations)  #各試行の報酬を格納
final_x = np.zeros((num_episodes, 1))  #学習後、各試行のt=200でのｘの位置を格納
islearned = 0  #学習が終わったフラグ
isrender = 0  #描画フラグ
 
def memorize(ep,a_cnt,action,state,reward,q,cart_pos,cart_v,pole_angle,pole_v):

    memory.append({'episode':ep,'a_cnt':a_cnt,'action':action,'state':state,\
                    'reward':reward, 'q':q,\
                    'cart_pos':cart_pos,'cart_v':cart_v,'pole_angle':pole_angle,'pole_v':pole_v
                    })

def writer_csv(n):
    with open("sampleA-{}.csv".format(n), "w", newline="") as f:
        fieldnames = ['episode','a_cnt','action','state','reward','q','cart_pos','cart_v','pole_angle','pole_v']

        dict_writer = csv.DictWriter(f, fieldnames = fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(memory)

log_step = [] #行動回数メモリ
_log_step = [] #行動回数メモリ
# [5] メインルーチン--------------------------------------------------
for episode in range(1, num_episodes+1):  #試行数分繰り返す
    # 環境の初期化
    observation = env.reset()
    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    episode_reward = 0
    a_cnt = 0
    for t in range(max_number_of_steps):  #1試行のループ
        if islearned == 1:  #学習終了したらcartPoleを描画する
            env.render()
            time.sleep(0.1)
            print (observation[0])  #カートのx位置を出力
 
        # 行動a_tの実行により、s_{t+1}, r_{t}などを計算する
        observation, reward, done, info = env.step(action)
        a_cnt += 1 #行動回数カウント
        cart_pos, cart_v, pole_angle, pole_v = observation
 
        # 報酬を設定し与える
        if done:
            if t < 195:
                reward = -200  #こけたら罰則
            else:
                reward = 1  #立ったまま終了時は罰則はなし
        else:
            reward = 1  #各ステップで立ってたら報酬追加
 
        episode_reward += reward  #報酬を追加
 
        # 離散状態s_{t+1}を求め、Q関数を更新する
        next_state = digitize_state(observation)  #t+1での観測状態を、離散値に変換
        if (episode % 50 == 0) or (episode < 100):
            memorize(episode,t,action,state,reward,q_table[state][action],cart_pos, cart_v, pole_angle, pole_v)
        q_table = update_Qtable(q_table, state, action, reward, next_state)

        #  次の行動a_{t+1}を求める 
        action = get_action(next_state, t)    # a_{t+1} 
        
        state = next_state
        
        #終了時の処理
        if done:
            print('%d Episode finished after %f time steps / mean %f' %
                  (episode, t + 1, total_reward_vec.mean()))
            total_reward_vec = np.hstack((total_reward_vec[1:],
                                          episode_reward))  #報酬を記録
            _log_step.append(a_cnt)
            if episode % 50 == 0:
                log_step.append(mean(_log_step))
                _log_step = []
            if islearned == 1:  #学習終わってたら最終のx座標を格納
                final_x[episode, 0] = observation[0]
            break

    if (total_reward_vec.mean() >=
            goal_average_reward):  # 直近の100エピソードが規定報酬以上であれば成功
        #print('Episode %d train agent successfuly!' % episode)
        #islearned = 1
        #np.savetxt('learned_Q_table.csv',q_table, delimiter=",") #Qtableの保存する場合
        if isrender == 0:
            #env = wrappers.Monitor(env, './movie/cartpole-experiment-1') #動画保存する場合
            isrender = 1
    #10エピソードだけでどんな挙動になるのか見たかったら、以下のコメントを外す
    #if episode>10:
    #    if isrender == 0:
    #        env = wrappers.Monitor(env, './movie/cartpole-experiment-1') #動画保存する場合
    #        isrender = 1
    #    islearned=1;
print(log_step)
writer_csv(49)
if islearned:
    np.savetxt('final_x.csv', final_x, delimiter=",")