from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import random

class Maze():
    PATH = 0
    WALL = 1

    def __init__(self, width, height):
        """初期設定"""
        self.maze = []
        self.width = width + 1
        self.height = height + 1
        self.pin = False

        if(self.height < 5 or self.width < 5): #迷路は、幅高さ5以上の奇数で生成する
            print('at least 5')
            exit()
        if (self.width % 2) == 0:
            self.width += 1
        if (self.height % 2) == 0:
            self.height += 1


    def set_out_wall(self):
        """迷路の外周を壁とし、それ以外を通路とする"""
        for _x in range(0, self.width):
            row = []
            for _y in range(0, self.height):
                if (_x == 0 or _y == 0 or _x == self.width-1 or _y == self.height -1):
                    cell = self.WALL
                else:
                    cell = self.PATH
                row.append(cell)
            self.maze.append(row)
        return self.maze


    def set_inner_wall_boutaosi(self):
        """壁を生成する
            #外周の内側に基準となる棒を1セルおき、x,yともに偶数の座標に配置する
            # 棒をランダムな方向に倒して壁とする
            # 1行目の内側の壁以外では上方向に倒してはいけない
            # すでに棒が倒され壁になっている場合、その方向には倒してはいけない"""
        #random.seed(42) #迷路を乱数で固定 
        for _x in range(2, self.width-1, 2):
            for _y in range(2, self.height-1, 2):
                self.maze[_x][_y] = self.WALL

                while True: #棒を倒して壁にする方向を決める
                    if _y == 2:
                        direction = random.randrange(0, 4)
                    else:
                        direction = random.randrange(0, 3)
                    wall_x = _x
                    wall_y = _y
                    if direction == 0:
                        wall_x += 1
                    elif direction == 1:
                        wall_y += 1
                    elif direction == 2:
                        wall_x -= 1
                    else:
                        wall_y -= 1

                    if self.maze[wall_x][wall_y] != self.WALL: #壁にする方向が壁でない場合は壁にする
                        self.maze[wall_x][wall_y] = self.WALL
                        break
        return self.maze


    def set_start_goal(self, start, goal):
        """ スタートとゴールを迷路にいれる"""
        self.maze[start[0]][start[1]] = 'S'
        self.maze[goal[0]][goal[1]] = 'G'
        
        self.maze[1][2] = self.WALL
        return self.maze


    def bg_maze(self,kernel):
        """BG_networkのkernel調整"""
        if kernel > 1:
            kernel -= 1
            new_width = [['1'] * self.width] * kernel
            new_width = np.array(new_width)
            new_width.reshape(self.width, kernel)

            new_height = [['1'] * kernel] * (self.height + 2 * kernel)
            new_height = np.array(new_height)
            new_height.reshape(kernel, self.height + 2 * kernel)
            
            self.maze = np.vstack((new_width, self.maze))
            self.maze = np.vstack((self.maze, new_width)) 
            self.maze = np.hstack((self.maze, new_height))
            self.maze = np.hstack((new_height, self.maze))

        self.maze = np.array(self.maze)
        print(self.maze)
        print()


    def reset(self):
        start = np.where(self.maze == 'S') #スタートのインデックス取得
        start = list(map(list, start))
        self.start = np.array([start[0][0], start[1][0]]) 


    def run(self, agent, n, episode_count):
        """環境内でエージェントを動かす""" 
        self.reset() #スタート位置情報の取得,変更
        a_list = [] #行動回数メモリ
        x_list = np.zeros([5,5]) #位置回数メモリ

        for ep in tqdm(range(1, episode_count+1)):
            state = self.start
            done = False #課題完了の判断
            a_cnt = 0 #行動回数
            x_list[state[0]-2,state[1]-2]+=1 #位置回数カウント
            if ep == 190:
                x_list = np.zeros([5,5])

            while True:
                agent.learn_fh(state) #bg_loop前半
                action = agent.get_action(ep,state) #Policyで行動決定
                a_cnt += 1 #行動か数カウント
                n_state, reward, done, action = self.step(agent, ep, state, action, done) #環境を進める
                pre_state = state
                state = n_state
                x_list[state[0]-2, state[1]-2] += 1 #位置回数カウント
                agent.learn_sh(ep, a_cnt, pre_state, state, action, reward) #bg_loop後半

                if done == True: #終了判定
                    agent.episode_fin(ep) #エピソード終了時の更新
                    break

                #if a_cnt == 100:　#行動回数が上限に達したら終了
                #    done = True
            a_list.append(a_cnt)

        """結果の表示"""
        print(a_list) #各エピソードにおける行動回数を表示
        x = np.arange(1, episode_count+1, 1) #そのグラフを表示
        y = a_list
        plt.figure(figsize=(8, 4))
        plt.plot(x, y)
#         plt.legend()
        plt.title('learning_rate')
        plt.xlabel('Episode')
        plt.ylabel('action_steps')
        plt.grid(True)
        plt.show()

        agent.writer(n) #CSVファイルを記入
        sns.heatmap(x_list, annot=False, square=True, cmap='Greens') #遷移回数のヒートマップを表示
        plt.show()


    def step(self, agent, ep, state, action, done):
        """環境を進める"""
        n_state = state + action
        if self.maze[n_state[0], n_state[1]] == "1": #n_stateが行けないとき再びアクションを選ぶ
            while self.maze[n_state[0], n_state[1]] == "1":
                action = agent.get_action(ep, state)
                n_state = state + action

        if self.maze[n_state[0], n_state[1]] == "G": #n_stateに応じて報酬を決める
            reward=1
            done = True
        else:
            reward = -0.01 # reward = -0.01

        return n_state, reward, done, action