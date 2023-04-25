from environment import Maze
from agent import Agent

def train():
    kernel = 2 #迷路の外壁の厚み（拡張分）
    start = [1, 1] #start位置の指定
    goal = [5, 5] #goal位置の指定

    """迷路を自動生成"""
    env = Maze(5, 5) #迷路の大きさ指定
    env.set_out_wall() #迷路を壁で囲う
    #env.set_inner_wall_boutaosi() #壁を自動生成
    env.set_start_goal(start, goal) #startとgoalを指定
    env.bg_maze(kernel) #kernel分の迷路のマス目拡張

    """環境にエージェント生成・学習"""
    n = 5 #エピソードの試行回数
    episode_count = 50 #エピソード回数
    for e in range(1, n+1):
        epsilon = 0.1
        agent = Agent(env.maze)
        env.run(agent, e, episode_count)
    print('Finish')


if __name__ == "__main__":
    train()