import numpy as np
import random
import csv

class BG_Network():

    def __init__(self, env):
        """環境に応じたネットワーク構築"""
        self.action_num = 4 #行動の選択肢
        self.shape = env.shape #ネットワークの形状
        self.maze = env

        """細胞の設定"""
        self.x = np.zeros(self.shape) #大脳皮質（入力）
        self.stn = np.zeros(self.shape) #視床下核（D5レセプター）
        self.d1 = np.zeros(self.shape) #D1レセプター
        self.d2 = np.zeros(self.shape) #D2レセプター
        self.gpe = np.zeros(self.shape) #淡蒼球外節
        self.snr = np.zeros(self.shape + tuple([self.action_num + 1])) #淡蒼球内節
        self.xs = np.zeros(self.shape + tuple([self.action_num + 1])) #大脳皮質（再入力）
        self.strio = np.zeros(self.shape + tuple([self.action_num + 1])) #ストリオソーム
        self.da = np.zeros(self.shape) #ドーパミン細胞

        """入力"""
        self.input = 1 #入力

        """重みパラメータ"""
        self.wstn = np.ones(self.shape) #大脳皮質 - 視床下核（D5レセプター）
        self.wd1 = np.ones(self.shape) #大脳皮質 - D1レセプター
        self.wd2 = np.ones(self.shape) #大脳皮質 - D2レセプター
        self.wstrio = np.ones(self.shape)*5.5 #大脳皮質 - ストリオソーム

        """サイズパラメータ"""
        self.stn_ratio = 1 #大脳皮質 - STN（D5レセプター）
        self.d1_ratio = 1 #大脳皮質 - D1レセプター
        self.d2_ratio = 1 #大脳皮質 - D2レセプター
        self.lc_stn_ratio = 1 #局所的ハイパー直接路
        self.gl_stn_ratio = 1 #広域的ハイパー直接路
        self.lc_gpe_ratio = 1 #局所的関節路
        self.gl_gpe_ratio = 0.1 #広域的関節路

        """BG_loopツール"""
        self.lc_hyper = np.zeros(self.shape) #局所的ハイパー直接路
        self.gl_hyper = np.zeros(self.shape) #広域的ハイパー直接路
        self.lc_gpe = np.zeros(self.shape) #局所的関節路
        self.gl_gpe = np.zeros(self.shape) #広域的関節路
        self.memory = [] #メモリ

        self.str = np.zeros(self.shape)


    def bg_loop_fh(self, state):
        """位置情報の取得"""
        i = state[0]
        j = state[1]
        s = tuple([i, j]) #現在のマス
        u = tuple([i-1, j]) #上のマス
        r = tuple([i, j+1]) #右のマス
        l = tuple([i, j-1]) #左のマス
        d = tuple([i+1, j]) #下のマス
        
        """ゴール位置の取得"""
        goal = np.where(self.maze=='G')
        goal = list(map(list, goal))
        

        """細胞活動のリセット"""
        self.x = np.zeros(self.shape) #大脳皮質（入力）
        self.stn = np.zeros(self.shape) #視床下核（D5レセプター）
        self.d1 = np.zeros(self.shape) #D1レセプター
        self.d2 = np.zeros(self.shape) #D2レセプター
        self.gpe = np.zeros(self.shape) #淡蒼球外節
        self.snr = np.zeros(self.shape + tuple([self.action_num + 1])) #淡蒼球内節
        self.xs = np.zeros(self.shape + tuple([self.action_num + 1])) #大脳皮質（再入力）
        self.strio = np.zeros(self.shape + tuple([self.action_num + 1])) #ストリオソーム
        self.da = np.zeros(self.shape) #ドーパミン細胞

        """BG_loopツールのリセット"""
        self.lc_hyper = np.zeros(self.shape) #局所的ハイパー直接路
        self.gl_hyper = np.zeros(self.shape) #広域的ハイパー直接路
        self.lc_gpe = np.zeros(self.shape) #局所的淡蒼球外節（局所的関節路）
        self.gl_gpe = np.zeros(self.shape) #広域的淡蒼球外節（広域的関節路）

        self.str = np.zeros(self.shape)

        """入力"""
        self.x[s] = self.input
        self.x[u] = self.input
        self.x[r] = self.input
        self.x[l] = self.input
        self.x[d] = self.input

        if i == goal[0][0] and j == goal[1][0]:
            self.x[u] = 0
            self.x[r] = 0
            self.x[l] = 0
            self.x[d] = 0

        """
        ハイパー直接路:大脳皮質→視床下核→淡蒼球内節
        """
        self.stn = self.x * self.wstn *self.stn_ratio
        self.lc_hyper = self.stn * self.lc_stn_ratio #局所的ハイパー直接路
        self.gl_hyper = self.stn * self.gl_stn_ratio #広域的ハイパー直接路
        self.snr[s+tuple([0])] = self.snr[s+tuple([0])] + self.lc_hyper[s] + self.gl_hyper[u]
        self.snr[s+tuple([1])] = self.snr[s+tuple([1])] + self.lc_hyper[s] + self.gl_hyper[r]
        self.snr[s+tuple([2])] = self.snr[s+tuple([2])] + self.lc_hyper[s] + self.gl_hyper[l]
        self.snr[s+tuple([3])] = self.snr[s+tuple([3])] + self.lc_hyper[s] + self.gl_hyper[d]
        self.snr[s+tuple([4])] = self.snr[s+tuple([4])] + self.lc_hyper[s]

        """
        直接路:大脳皮質→D1→[抑制性]淡蒼球内節
        """
        self.d1 = self.x * self.wd1 * self.d1_ratio
        for p in range(0, 5):
            self.snr[s+tuple([p])] = self.snr[s+tuple([p])] - self.d1[s]

        """
        関節路:大脳皮質→D2→[抑制性]淡蒼球外節→[抑制性]淡蒼球内節
        """
        self.d2 = self.x * self.wd2 * self.d2_ratio
        self.gpe = -self.d2
        self.lc_gpe = self.gpe * self.lc_gpe_ratio #局所的関節路
        self.gl_gpe = self.gpe * self.gl_gpe_ratio #広域的関節路
        self.snr[s+tuple([0])] = self.snr[s+tuple([0])] - self.lc_gpe[s] - self.gl_gpe[u]
        self.snr[s+tuple([1])] = self.snr[s+tuple([1])] - self.lc_gpe[s] - self.gl_gpe[r]
        self.snr[s+tuple([2])] = self.snr[s+tuple([2])] - self.lc_gpe[s] - self.gl_gpe[l]
        self.snr[s+tuple([3])] = self.snr[s+tuple([3])] - self.lc_gpe[s] - self.gl_gpe[d]
        self.snr[s+tuple([4])] = self.snr[s+tuple([4])] - self.lc_gpe[s]

        """
        淡蒼球内接→[抑制性]大脳皮質
        """
        if i==goal[0][0] and j==goal[1][0]: # ゴールのとき別処理
            self.snr = self.snr - (self.lc_stn_ratio)
        else:
            self.snr = self.snr - (self.lc_stn_ratio + self.gl_stn_ratio + self.gl_gpe_ratio)

        self.xs = -self.snr


    def policy(self, ep, state):
        """位置情報と行動情報の取得"""
        s = tuple([state[0], state[1]])
        u = np.array([-1, 0])
        r = np.array([0, 1])
        l = np.array([0, -1])
        d = np.array([1, 0])
        action = [u, r, l, d]

        """ε-greedyに基づく方策"""
        self.epsilon = 0.5 * (1 / (ep + 1e-8))
        if np.random.random() < self.epsilon:
            a = random.choices(action)
            action = a[0]
            return action
        else:
            xx = np.array([self.snr[s+tuple([0])], self.snr[s+tuple([1])], self.snr[s+tuple([2])], self.snr[s+tuple([3])]])
            max_x = [i for i, x in enumerate(xx) if x == max(xx)]
            max_x_index = random.choices(max_x)
            return action[max_x_index[0]]


    def bg_loop_sh(self, ep, a_cnt, pre_state, state, action, reward):
        """位置情報と行動情報の取得"""
        s = tuple([state[0], state[1]]) #更新する大脳基底核

        """
        大脳皮質→ストリオソーム→[抑制性]ドーパミン細胞
        """
        self.strio[s] = self.xs[s] * self.wstrio[s]
        self.da[s] = np.max(reward - self.strio[s])

        self.str[s] = np.max(-self.strio[s])

        """
        ドーパミンによるシナプス可塑性:ドーパミン細胞→ハイパー直接路, 直接路, 間接路, ストリオソーム
        """
        self.wstn[s] += 0.01 * self.da[s] * self.x[s]
        self.wd1[s] += 0.01 * self.da[s] * self.x[s]
        self.wd2[s] += -0.01 * self.da[s] * self.x[s]
        self.wstrio[s] += 0.01 * self.da[s] * self.x[s]

#         self.wstn[s] += 0.004 * self.da[s] * self.x[s] * self.stn[s]
#         self.wd1[s] += 0.004 * self.da[s] * self.x[s] * self.d1[s]
#         self.wd2[s] += -0.004 * self.da[s] * self.x[s] * self.d2[s]
#         self.wstrio[s] += 0.004 * self.da[s] * self.x[s] * self.strio[s+tuple([4])]
        # print('wstn ',self.wstn[s], ' wd1 ', self.wd1[s], ' wd2 ', self.wd2[s], ' wstrio ', self.wstrio[s])

        self.memorize(ep, a_cnt, action, state, reward)


    def episode_fin(self, e):
        """episode毎の処理"""


    def memorize(self, ep, a_cnt, action, state, reward):
        self.memory.append({'episode':ep, 'a_cnt':a_cnt, 'action':action, 's_row':state[0], 's_col':state[1],\
                        'reward':reward, 'DAsum':self.da.sum(),\

                        'x22':self.x[2,2],\
                        'x23':self.x[2,3],\
                        'x24':self.x[2,4],\
                        'x25':self.x[2,5],\
                        'x26':self.x[2,6],\
                        'x27':self.x[2,7],\
                        'x28':self.x[2,8],\
                        'x32':self.x[3,2],\
                        'x33':self.x[3,3],\
                        'x34':self.x[3,4],\
                        'x35':self.x[3,5],\
                        'x36':self.x[3,6],\
                        'x37':self.x[3,7],\
                        'x38':self.x[3,8],\
                        'x42':self.x[4,2],\
                        'x43':self.x[4,3],\
                        'x44':self.x[4,4],\
                        'x45':self.x[4,5],\
                        'x46':self.x[4,6],\
                        'x47':self.x[4,7],\
                        'x48':self.x[4,8],\
                        'x52':self.x[5,2],\
                        'x53':self.x[5,3],\
                        'x54':self.x[5,4],\
                        'x55':self.x[5,5],\
                        'x56':self.x[5,6],\
                        'x57':self.x[5,7],\
                        'x58':self.x[5,8],\
                        'x62':self.x[6,2],\
                        'x63':self.x[6,3],\
                        'x64':self.x[6,4],\
                        'x65':self.x[6,5],\
                        'x66':self.x[6,6],\
                        'x67':self.x[6,7],\
                        'x68':self.x[6,8],\
                        'x72':self.x[7,2],\
                        'x73':self.x[7,3],\
                        'x74':self.x[7,4],\
                        'x75':self.x[7,5],\
                        'x76':self.x[7,6],\
                        'x77':self.x[7,7],\
                        'x78':self.x[7,8],\
                        'x82':self.x[8,2],\
                        'x83':self.x[8,3],\
                        'x84':self.x[8,4],\
                        'x85':self.x[8,5],\
                        'x86':self.x[8,6],\
                        'x87':self.x[8,7],\
                        'x88':self.x[8,8],\

                        'D122':self.d1[2,2],\
                        'D123':self.d1[2,3],\
                        'D124':self.d1[2,4],\
                        'D125':self.d1[2,5],\
                        'D126':self.d1[2,6],\
                        'D127':self.d1[2,7],\
                        'D128':self.d1[2,8],\
                        'D132':self.d1[3,2],\
                        'D133':self.d1[3,3],\
                        'D134':self.d1[3,4],\
                        'D135':self.d1[3,5],\
                        'D136':self.d1[3,6],\
                        'D137':self.d1[3,7],\
                        'D138':self.d1[3,8],\
                        'D142':self.d1[4,2],\
                        'D143':self.d1[4,3],\
                        'D144':self.d1[4,4],\
                        'D145':self.d1[4,5],\
                        'D146':self.d1[4,6],\
                        'D147':self.d1[4,7],\
                        'D148':self.d1[4,8],\
                        'D152':self.d1[5,2],\
                        'D153':self.d1[5,3],\
                        'D154':self.d1[5,4],\
                        'D155':self.d1[5,5],\
                        'D156':self.d1[5,6],\
                        'D157':self.d1[5,7],\
                        'D158':self.d1[5,8],\
                        'D162':self.d1[6,2],\
                        'D163':self.d1[6,3],\
                        'D164':self.d1[6,4],\
                        'D165':self.d1[6,5],\
                        'D166':self.d1[6,6],\
                        'D167':self.d1[6,7],\
                        'D168':self.d1[6,8],\
                        'D172':self.d1[7,2],\
                        'D173':self.d1[7,3],\
                        'D174':self.d1[7,4],\
                        'D175':self.d1[7,5],\
                        'D176':self.d1[7,6],\
                        'D177':self.d1[7,7],\
                        'D178':self.d1[7,8],\
                        'D182':self.d1[8,2],\
                        'D183':self.d1[8,3],\
                        'D184':self.d1[8,4],\
                        'D185':self.d1[8,5],\
                        'D186':self.d1[8,6],\
                        'D187':self.d1[8,7],\
                        'D188':self.d1[8,8],\

                        'D222':self.d2[2,2],\
                        'D223':self.d2[2,3],\
                        'D224':self.d2[2,4],\
                        'D225':self.d2[2,5],\
                        'D226':self.d2[2,6],\
                        'D227':self.d2[2,7],\
                        'D228':self.d2[2,8],\
                        'D232':self.d2[3,2],\
                        'D233':self.d2[3,3],\
                        'D234':self.d2[3,4],\
                        'D235':self.d2[3,5],\
                        'D236':self.d2[3,6],\
                        'D237':self.d2[3,7],\
                        'D238':self.d2[3,8],\
                        'D242':self.d2[4,2],\
                        'D243':self.d2[4,3],\
                        'D244':self.d2[4,4],\
                        'D245':self.d2[4,5],\
                        'D246':self.d2[4,6],\
                        'D247':self.d2[4,7],\
                        'D248':self.d2[4,8],\
                        'D252':self.d2[5,2],\
                        'D253':self.d2[5,3],\
                        'D254':self.d2[5,4],\
                        'D255':self.d2[5,5],\
                        'D256':self.d2[5,6],\
                        'D257':self.d2[5,7],\
                        'D258':self.d2[5,8],\
                        'D262':self.d2[6,2],\
                        'D263':self.d2[6,3],\
                        'D264':self.d2[6,4],\
                        'D265':self.d2[6,5],\
                        'D266':self.d2[6,6],\
                        'D267':self.d2[6,7],\
                        'D268':self.d2[6,8],\
                        'D272':self.d2[7,2],\
                        'D273':self.d2[7,3],\
                        'D274':self.d2[7,4],\
                        'D275':self.d2[7,5],\
                        'D276':self.d2[7,6],\
                        'D277':self.d2[7,7],\
                        'D278':self.d2[7,8],\
                        'D282':self.d2[8,2],\
                        'D283':self.d2[8,3],\
                        'D284':self.d2[8,4],\
                        'D285':self.d2[8,5],\
                        'D286':self.d2[8,6],\
                        'D287':self.d2[8,7],\
                        'D288':self.d2[8,8],\

                        'lc_gpe22':self.lc_gpe[2,2],\
                        'lc_gpe23':self.lc_gpe[2,3],\
                        'lc_gpe24':self.lc_gpe[2,4],\
                        'lc_gpe25':self.lc_gpe[2,5],\
                        'lc_gpe26':self.lc_gpe[2,6],\
                        'lc_gpe27':self.lc_gpe[2,7],\
                        'lc_gpe28':self.lc_gpe[2,8],\
                        'lc_gpe32':self.lc_gpe[3,2],\
                        'lc_gpe33':self.lc_gpe[3,3],\
                        'lc_gpe34':self.lc_gpe[3,4],\
                        'lc_gpe35':self.lc_gpe[3,5],\
                        'lc_gpe36':self.lc_gpe[3,6],\
                        'lc_gpe37':self.lc_gpe[3,7],\
                        'lc_gpe38':self.lc_gpe[3,8],\
                        'lc_gpe42':self.lc_gpe[4,2],\
                        'lc_gpe43':self.lc_gpe[4,3],\
                        'lc_gpe44':self.lc_gpe[4,4],\
                        'lc_gpe45':self.lc_gpe[4,5],\
                        'lc_gpe46':self.lc_gpe[4,6],\
                        'lc_gpe47':self.lc_gpe[4,7],\
                        'lc_gpe48':self.lc_gpe[4,8],\
                        'lc_gpe52':self.lc_gpe[5,2],\
                        'lc_gpe53':self.lc_gpe[5,3],\
                        'lc_gpe54':self.lc_gpe[5,4],\
                        'lc_gpe55':self.lc_gpe[5,5],\
                        'lc_gpe56':self.lc_gpe[5,6],\
                        'lc_gpe57':self.lc_gpe[5,7],\
                        'lc_gpe58':self.lc_gpe[5,8],\
                        'lc_gpe62':self.lc_gpe[6,2],\
                        'lc_gpe63':self.lc_gpe[6,3],\
                        'lc_gpe64':self.lc_gpe[6,4],\
                        'lc_gpe65':self.lc_gpe[6,5],\
                        'lc_gpe66':self.lc_gpe[6,6],\
                        'lc_gpe67':self.lc_gpe[6,7],\
                        'lc_gpe68':self.lc_gpe[6,8],\
                        'lc_gpe72':self.lc_gpe[7,2],\
                        'lc_gpe73':self.lc_gpe[7,3],\
                        'lc_gpe74':self.lc_gpe[7,4],\
                        'lc_gpe75':self.lc_gpe[7,5],\
                        'lc_gpe76':self.lc_gpe[7,6],\
                        'lc_gpe77':self.lc_gpe[7,7],\
                        'lc_gpe78':self.lc_gpe[7,8],\
                        'lc_gpe82':self.lc_gpe[8,2],\
                        'lc_gpe83':self.lc_gpe[8,3],\
                        'lc_gpe84':self.lc_gpe[8,4],\
                        'lc_gpe85':self.lc_gpe[8,5],\
                        'lc_gpe86':self.lc_gpe[8,6],\
                        'lc_gpe87':self.lc_gpe[8,7],\
                        'lc_gpe88':self.lc_gpe[8,8],\

                        'gl_gpe22':self.gl_gpe[2,2],\
                        'gl_gpe23':self.gl_gpe[2,3],\
                        'gl_gpe24':self.gl_gpe[2,4],\
                        'gl_gpe25':self.gl_gpe[2,5],\
                        'gl_gpe26':self.gl_gpe[2,6],\
                        'gl_gpe27':self.gl_gpe[2,7],\
                        'gl_gpe28':self.gl_gpe[2,8],\
                        'gl_gpe32':self.gl_gpe[3,2],\
                        'gl_gpe33':self.gl_gpe[3,3],\
                        'gl_gpe34':self.gl_gpe[3,4],\
                        'gl_gpe35':self.gl_gpe[3,5],\
                        'gl_gpe36':self.gl_gpe[3,6],\
                        'gl_gpe37':self.gl_gpe[3,7],\
                        'gl_gpe38':self.gl_gpe[3,8],\
                        'gl_gpe42':self.gl_gpe[4,2],\
                        'gl_gpe43':self.gl_gpe[4,3],\
                        'gl_gpe44':self.gl_gpe[4,4],\
                        'gl_gpe45':self.gl_gpe[4,5],\
                        'gl_gpe46':self.gl_gpe[4,6],\
                        'gl_gpe47':self.gl_gpe[4,7],\
                        'gl_gpe48':self.gl_gpe[4,8],\
                        'gl_gpe52':self.gl_gpe[5,2],\
                        'gl_gpe53':self.gl_gpe[5,3],\
                        'gl_gpe54':self.gl_gpe[5,4],\
                        'gl_gpe55':self.gl_gpe[5,5],\
                        'gl_gpe56':self.gl_gpe[5,6],\
                        'gl_gpe57':self.gl_gpe[5,7],\
                        'gl_gpe58':self.gl_gpe[5,8],\
                        'gl_gpe62':self.gl_gpe[6,2],\
                        'gl_gpe63':self.gl_gpe[6,3],\
                        'gl_gpe64':self.gl_gpe[6,4],\
                        'gl_gpe65':self.gl_gpe[6,5],\
                        'gl_gpe66':self.gl_gpe[6,6],\
                        'gl_gpe67':self.gl_gpe[6,7],\
                        'gl_gpe68':self.gl_gpe[6,8],\
                        'gl_gpe72':self.gl_gpe[7,2],\
                        'gl_gpe73':self.gl_gpe[7,3],\
                        'gl_gpe74':self.gl_gpe[7,4],\
                        'gl_gpe75':self.gl_gpe[7,5],\
                        'gl_gpe76':self.gl_gpe[7,6],\
                        'gl_gpe77':self.gl_gpe[7,7],\
                        'gl_gpe78':self.gl_gpe[7,8],\
                        'gl_gpe82':self.gl_gpe[8,2],\
                        'gl_gpe83':self.gl_gpe[8,3],\
                        'gl_gpe84':self.gl_gpe[8,4],\
                        'gl_gpe85':self.gl_gpe[8,5],\
                        'gl_gpe86':self.gl_gpe[8,6],\
                        'gl_gpe87':self.gl_gpe[8,7],\
                        'gl_gpe88':self.gl_gpe[8,8],\

                        'lc_hyper22':self.lc_hyper[2,2],\
                        'lc_hyper23':self.lc_hyper[2,3],\
                        'lc_hyper24':self.lc_hyper[2,4],\
                        'lc_hyper25':self.lc_hyper[2,5],\
                        'lc_hyper26':self.lc_hyper[2,6],\
                        'lc_hyper27':self.lc_hyper[2,7],\
                        'lc_hyper28':self.lc_hyper[2,8],\
                        'lc_hyper32':self.lc_hyper[3,2],\
                        'lc_hyper33':self.lc_hyper[3,3],\
                        'lc_hyper34':self.lc_hyper[3,4],\
                        'lc_hyper35':self.lc_hyper[3,5],\
                        'lc_hyper36':self.lc_hyper[3,6],\
                        'lc_hyper37':self.lc_hyper[3,7],\
                        'lc_hyper38':self.lc_hyper[3,8],\
                        'lc_hyper42':self.lc_hyper[4,2],\
                        'lc_hyper43':self.lc_hyper[4,3],\
                        'lc_hyper44':self.lc_hyper[4,4],\
                        'lc_hyper45':self.lc_hyper[4,5],\
                        'lc_hyper46':self.lc_hyper[4,6],\
                        'lc_hyper47':self.lc_hyper[4,7],\
                        'lc_hyper48':self.lc_hyper[4,8],\
                        'lc_hyper52':self.lc_hyper[5,2],\
                        'lc_hyper53':self.lc_hyper[5,3],\
                        'lc_hyper54':self.lc_hyper[5,4],\
                        'lc_hyper55':self.lc_hyper[5,5],\
                        'lc_hyper56':self.lc_hyper[5,6],\
                        'lc_hyper57':self.lc_hyper[5,7],\
                        'lc_hyper58':self.lc_hyper[5,8],\
                        'lc_hyper62':self.lc_hyper[6,2],\
                        'lc_hyper63':self.lc_hyper[6,3],\
                        'lc_hyper64':self.lc_hyper[6,4],\
                        'lc_hyper65':self.lc_hyper[6,5],\
                        'lc_hyper66':self.lc_hyper[6,6],\
                        'lc_hyper67':self.lc_hyper[6,7],\
                        'lc_hyper68':self.lc_hyper[6,8],\
                        'lc_hyper72':self.lc_hyper[7,2],\
                        'lc_hyper73':self.lc_hyper[7,3],\
                        'lc_hyper74':self.lc_hyper[7,4],\
                        'lc_hyper75':self.lc_hyper[7,5],\
                        'lc_hyper76':self.lc_hyper[7,6],\
                        'lc_hyper77':self.lc_hyper[7,7],\
                        'lc_hyper78':self.lc_hyper[7,8],\
                        'lc_hyper82':self.lc_hyper[8,2],\
                        'lc_hyper83':self.lc_hyper[8,3],\
                        'lc_hyper84':self.lc_hyper[8,4],\
                        'lc_hyper85':self.lc_hyper[8,5],\
                        'lc_hyper86':self.lc_hyper[8,6],\
                        'lc_hyper87':self.lc_hyper[8,7],\
                        'lc_hyper88':self.lc_hyper[8,8],\

                        'gl_hyper22':self.gl_hyper[2,2],\
                        'gl_hyper23':self.gl_hyper[2,3],\
                        'gl_hyper24':self.gl_hyper[2,4],\
                        'gl_hyper25':self.gl_hyper[2,5],\
                        'gl_hyper26':self.gl_hyper[2,6],\
                        'gl_hyper27':self.gl_hyper[2,7],\
                        'gl_hyper28':self.gl_hyper[2,8],\
                        'gl_hyper32':self.gl_hyper[3,2],\
                        'gl_hyper33':self.gl_hyper[3,3],\
                        'gl_hyper34':self.gl_hyper[3,4],\
                        'gl_hyper35':self.gl_hyper[3,5],\
                        'gl_hyper36':self.gl_hyper[3,6],\
                        'gl_hyper37':self.gl_hyper[3,7],\
                        'gl_hyper38':self.gl_hyper[3,8],\
                        'gl_hyper42':self.gl_hyper[4,2],\
                        'gl_hyper43':self.gl_hyper[4,3],\
                        'gl_hyper44':self.gl_hyper[4,4],\
                        'gl_hyper45':self.gl_hyper[4,5],\
                        'gl_hyper46':self.gl_hyper[4,6],\
                        'gl_hyper47':self.gl_hyper[4,7],\
                        'gl_hyper48':self.gl_hyper[4,8],\
                        'gl_hyper52':self.gl_hyper[5,2],\
                        'gl_hyper53':self.gl_hyper[5,3],\
                        'gl_hyper54':self.gl_hyper[5,4],\
                        'gl_hyper55':self.gl_hyper[5,5],\
                        'gl_hyper56':self.gl_hyper[5,6],\
                        'gl_hyper57':self.gl_hyper[5,7],\
                        'gl_hyper58':self.gl_hyper[5,8],\
                        'gl_hyper62':self.gl_hyper[6,2],\
                        'gl_hyper63':self.gl_hyper[6,3],\
                        'gl_hyper64':self.gl_hyper[6,4],\
                        'gl_hyper65':self.gl_hyper[6,5],\
                        'gl_hyper66':self.gl_hyper[6,6],\
                        'gl_hyper67':self.gl_hyper[6,7],\
                        'gl_hyper68':self.gl_hyper[6,8],\
                        'gl_hyper72':self.gl_hyper[7,2],\
                        'gl_hyper73':self.gl_hyper[7,3],\
                        'gl_hyper74':self.gl_hyper[7,4],\
                        'gl_hyper75':self.gl_hyper[7,5],\
                        'gl_hyper76':self.gl_hyper[7,6],\
                        'gl_hyper77':self.gl_hyper[7,7],\
                        'gl_hyper78':self.gl_hyper[7,8],\
                        'gl_hyper82':self.gl_hyper[8,2],\
                        'gl_hyper83':self.gl_hyper[8,3],\
                        'gl_hyper84':self.gl_hyper[8,4],\
                        'gl_hyper85':self.gl_hyper[8,5],\
                        'gl_hyper86':self.gl_hyper[8,6],\
                        'gl_hyper87':self.gl_hyper[8,7],\
                        'gl_hyper88':self.gl_hyper[8,8],\

                        'SNr22U':self.snr[2,2,0],\
                        'SNr23U':self.snr[2,3,0],\
                        'SNr24U':self.snr[2,4,0],\
                        'SNr25U':self.snr[2,5,0],\
                        'SNr26U':self.snr[2,6,0],\
                        'SNr27U':self.snr[2,7,0],\
                        'SNr28U':self.snr[2,8,0],\
                        'SNr32U':self.snr[3,2,0],\
                        'SNr33U':self.snr[3,3,0],\
                        'SNr34U':self.snr[3,4,0],\
                        'SNr35U':self.snr[3,5,0],\
                        'SNr36U':self.snr[3,6,0],\
                        'SNr37U':self.snr[3,7,0],\
                        'SNr38U':self.snr[3,8,0],\
                        'SNr42U':self.snr[4,2,0],\
                        'SNr43U':self.snr[4,3,0],\
                        'SNr44U':self.snr[4,4,0],\
                        'SNr45U':self.snr[4,5,0],\
                        'SNr46U':self.snr[4,6,0],\
                        'SNr47U':self.snr[4,7,0],\
                        'SNr48U':self.snr[4,8,0],\
                        'SNr52U':self.snr[5,2,0],\
                        'SNr53U':self.snr[5,3,0],\
                        'SNr54U':self.snr[5,4,0],\
                        'SNr55U':self.snr[5,5,0],\
                        'SNr56U':self.snr[5,6,0],\
                        'SNr57U':self.snr[5,7,0],\
                        'SNr58U':self.snr[5,8,0],\
                        'SNr62U':self.snr[6,2,0],\
                        'SNr63U':self.snr[6,3,0],\
                        'SNr64U':self.snr[6,4,0],\
                        'SNr65U':self.snr[6,5,0],\
                        'SNr66U':self.snr[6,6,0],\
                        'SNr67U':self.snr[6,7,0],\
                        'SNr68U':self.snr[6,8,0],\
                        'SNr72U':self.snr[7,2,0],\
                        'SNr73U':self.snr[7,3,0],\
                        'SNr74U':self.snr[7,4,0],\
                        'SNr75U':self.snr[7,5,0],\
                        'SNr76U':self.snr[7,6,0],\
                        'SNr77U':self.snr[7,7,0],\
                        'SNr78U':self.snr[7,8,0],\
                        'SNr82U':self.snr[8,2,0],\
                        'SNr83U':self.snr[8,3,0],\
                        'SNr84U':self.snr[8,4,0],\
                        'SNr85U':self.snr[8,5,0],\
                        'SNr86U':self.snr[8,6,0],\
                        'SNr87U':self.snr[8,7,0],\
                        'SNr88U':self.snr[8,8,0],\

                        'SNr22R':self.snr[2,2,1],\
                        'SNr23R':self.snr[2,3,1],\
                        'SNr24R':self.snr[2,4,1],\
                        'SNr25R':self.snr[2,5,1],\
                        'SNr26R':self.snr[2,6,1],\
                        'SNr27R':self.snr[2,7,1],\
                        'SNr28R':self.snr[2,8,1],\
                        'SNr32R':self.snr[3,2,1],\
                        'SNr33R':self.snr[3,3,1],\
                        'SNr34R':self.snr[3,4,1],\
                        'SNr35R':self.snr[3,5,1],\
                        'SNr36R':self.snr[3,6,1],\
                        'SNr37R':self.snr[3,7,1],\
                        'SNr38R':self.snr[3,8,1],\
                        'SNr42R':self.snr[4,2,1],\
                        'SNr43R':self.snr[4,3,1],\
                        'SNr44R':self.snr[4,4,1],\
                        'SNr45R':self.snr[4,5,1],\
                        'SNr46R':self.snr[4,6,1],\
                        'SNr47R':self.snr[4,7,1],\
                        'SNr48R':self.snr[4,8,1],\
                        'SNr52R':self.snr[5,2,1],\
                        'SNr53R':self.snr[5,3,1],\
                        'SNr54R':self.snr[5,4,1],\
                        'SNr55R':self.snr[5,5,1],\
                        'SNr56R':self.snr[5,6,1],\
                        'SNr57R':self.snr[5,7,1],\
                        'SNr58R':self.snr[5,8,1],\
                        'SNr62R':self.snr[6,2,1],\
                        'SNr63R':self.snr[6,3,1],\
                        'SNr64R':self.snr[6,4,1],\
                        'SNr65R':self.snr[6,5,1],\
                        'SNr66R':self.snr[6,6,1],\
                        'SNr67R':self.snr[6,7,1],\
                        'SNr68R':self.snr[6,8,1],\
                        'SNr72R':self.snr[7,2,1],\
                        'SNr73R':self.snr[7,3,1],\
                        'SNr74R':self.snr[7,4,1],\
                        'SNr75R':self.snr[7,5,1],\
                        'SNr76R':self.snr[7,6,1],\
                        'SNr77R':self.snr[7,7,1],\
                        'SNr78R':self.snr[7,8,1],\
                        'SNr82R':self.snr[8,2,1],\
                        'SNr83R':self.snr[8,3,1],\
                        'SNr84R':self.snr[8,4,1],\
                        'SNr85R':self.snr[8,5,1],\
                        'SNr86R':self.snr[8,6,1],\
                        'SNr87R':self.snr[8,7,1],\
                        'SNr88R':self.snr[8,8,1],\


                        'SNr22L':self.snr[2,2,2],\
                        'SNr23L':self.snr[2,3,2],\
                        'SNr24L':self.snr[2,4,2],\
                        'SNr25L':self.snr[2,5,2],\
                        'SNr26L':self.snr[2,6,2],\
                        'SNr27L':self.snr[2,7,2],\
                        'SNr28L':self.snr[2,8,2],\
                        'SNr32L':self.snr[3,2,2],\
                        'SNr33L':self.snr[3,3,2],\
                        'SNr34L':self.snr[3,4,2],\
                        'SNr35L':self.snr[3,5,2],\
                        'SNr36L':self.snr[3,6,2],\
                        'SNr37L':self.snr[3,7,2],\
                        'SNr38L':self.snr[3,8,2],\
                        'SNr42L':self.snr[4,2,2],\
                        'SNr43L':self.snr[4,3,2],\
                        'SNr44L':self.snr[4,4,2],\
                        'SNr45L':self.snr[4,5,2],\
                        'SNr46L':self.snr[4,6,2],\
                        'SNr47L':self.snr[4,7,2],\
                        'SNr48L':self.snr[4,8,2],\
                        'SNr52L':self.snr[5,2,2],\
                        'SNr53L':self.snr[5,3,2],\
                        'SNr54L':self.snr[5,4,2],\
                        'SNr55L':self.snr[5,5,2],\
                        'SNr56L':self.snr[5,6,2],\
                        'SNr57L':self.snr[5,7,2],\
                        'SNr58L':self.snr[5,8,2],\
                        'SNr62L':self.snr[6,2,2],\
                        'SNr63L':self.snr[6,3,2],\
                        'SNr64L':self.snr[6,4,2],\
                        'SNr65L':self.snr[6,5,2],\
                        'SNr66L':self.snr[6,6,2],\
                        'SNr67L':self.snr[6,7,2],\
                        'SNr68L':self.snr[6,8,2],\
                        'SNr72L':self.snr[7,2,2],\
                        'SNr73L':self.snr[7,3,2],\
                        'SNr74L':self.snr[7,4,2],\
                        'SNr75L':self.snr[7,5,2],\
                        'SNr76L':self.snr[7,6,2],\
                        'SNr77L':self.snr[7,7,2],\
                        'SNr78L':self.snr[7,8,2],\
                        'SNr82L':self.snr[8,2,2],\
                        'SNr83L':self.snr[8,3,2],\
                        'SNr84L':self.snr[8,4,2],\
                        'SNr85L':self.snr[8,5,2],\
                        'SNr86L':self.snr[8,6,2],\
                        'SNr87L':self.snr[8,7,2],\
                        'SNr88L':self.snr[8,8,2],\

                        'SNr22D':self.snr[2,2,3],\
                        'SNr23D':self.snr[2,3,3],\
                        'SNr24D':self.snr[2,4,3],\
                        'SNr25D':self.snr[2,5,3],\
                        'SNr26D':self.snr[2,6,3],\
                        'SNr27D':self.snr[2,7,3],\
                        'SNr28D':self.snr[2,8,3],\
                        'SNr32D':self.snr[3,2,3],\
                        'SNr33D':self.snr[3,3,3],\
                        'SNr34D':self.snr[3,4,3],\
                        'SNr35D':self.snr[3,5,3],\
                        'SNr36D':self.snr[3,6,3],\
                        'SNr37D':self.snr[3,7,3],\
                        'SNr38D':self.snr[3,8,3],\
                        'SNr42D':self.snr[4,2,3],\
                        'SNr43D':self.snr[4,3,3],\
                        'SNr44D':self.snr[4,4,3],\
                        'SNr45D':self.snr[4,5,3],\
                        'SNr46D':self.snr[4,6,3],\
                        'SNr47D':self.snr[4,7,3],\
                        'SNr48D':self.snr[4,8,3],\
                        'SNr52D':self.snr[5,2,3],\
                        'SNr53D':self.snr[5,3,3],\
                        'SNr54D':self.snr[5,4,3],\
                        'SNr55D':self.snr[5,5,3],\
                        'SNr56D':self.snr[5,6,3],\
                        'SNr57D':self.snr[5,7,3],\
                        'SNr58D':self.snr[5,8,3],\
                        'SNr62D':self.snr[6,2,3],\
                        'SNr63D':self.snr[6,3,3],\
                        'SNr64D':self.snr[6,4,3],\
                        'SNr65D':self.snr[6,5,3],\
                        'SNr66D':self.snr[6,6,3],\
                        'SNr67D':self.snr[6,7,3],\
                        'SNr68D':self.snr[6,8,3],\
                        'SNr72D':self.snr[7,2,3],\
                        'SNr73D':self.snr[7,3,3],\
                        'SNr74D':self.snr[7,4,3],\
                        'SNr75D':self.snr[7,5,3],\
                        'SNr76D':self.snr[7,6,3],\
                        'SNr77D':self.snr[7,7,3],\
                        'SNr78D':self.snr[7,8,3],\
                        'SNr82D':self.snr[8,2,3],\
                        'SNr83D':self.snr[8,3,3],\
                        'SNr84D':self.snr[8,4,3],\
                        'SNr85D':self.snr[8,5,3],\
                        'SNr86D':self.snr[8,6,3],\
                        'SNr87D':self.snr[8,7,3],\
                        'SNr88D':self.snr[8,8,3],\

                        'SNr22S':self.snr[2,2,4],\
                        'SNr23S':self.snr[2,3,4],\
                        'SNr24S':self.snr[2,4,4],\
                        'SNr25S':self.snr[2,5,4],\
                        'SNr26S':self.snr[2,6,4],\
                        'SNr27S':self.snr[2,7,4],\
                        'SNr28S':self.snr[2,8,4],\
                        'SNr32S':self.snr[3,2,4],\
                        'SNr33S':self.snr[3,3,4],\
                        'SNr34S':self.snr[3,4,4],\
                        'SNr35S':self.snr[3,5,4],\
                        'SNr36S':self.snr[3,6,4],\
                        'SNr37S':self.snr[3,7,4],\
                        'SNr38S':self.snr[3,8,4],\
                        'SNr42S':self.snr[4,2,4],\
                        'SNr43S':self.snr[4,3,4],\
                        'SNr44S':self.snr[4,4,4],\
                        'SNr45S':self.snr[4,5,4],\
                        'SNr46S':self.snr[4,6,4],\
                        'SNr47S':self.snr[4,7,4],\
                        'SNr48S':self.snr[4,8,4],\
                        'SNr52S':self.snr[5,2,4],\
                        'SNr53S':self.snr[5,3,4],\
                        'SNr54S':self.snr[5,4,4],\
                        'SNr55S':self.snr[5,5,4],\
                        'SNr56S':self.snr[5,6,4],\
                        'SNr57S':self.snr[5,7,4],\
                        'SNr58S':self.snr[5,8,4],\
                        'SNr62S':self.snr[6,2,4],\
                        'SNr63S':self.snr[6,3,4],\
                        'SNr64S':self.snr[6,4,4],\
                        'SNr65S':self.snr[6,5,4],\
                        'SNr66S':self.snr[6,6,4],\
                        'SNr67S':self.snr[6,7,4],\
                        'SNr68S':self.snr[6,8,4],\
                        'SNr72S':self.snr[7,2,4],\
                        'SNr73S':self.snr[7,3,4],\
                        'SNr74S':self.snr[7,4,4],\
                        'SNr75S':self.snr[7,5,4],\
                        'SNr76S':self.snr[7,6,4],\
                        'SNr77S':self.snr[7,7,4],\
                        'SNr78S':self.snr[7,8,4],\
                        'SNr82S':self.snr[8,2,4],\
                        'SNr83S':self.snr[8,3,4],\
                        'SNr84S':self.snr[8,4,4],\
                        'SNr85S':self.snr[8,5,4],\
                        'SNr86S':self.snr[8,6,4],\
                        'SNr87S':self.snr[8,7,4],\
                        'SNr88S':self.snr[8,8,4],\

                        'strio22':self.str[2,2],\
                        'strio23':self.str[2,3],\
                        'strio24':self.str[2,4],\
                        'strio25':self.str[2,5],\
                        'strio26':self.str[2,6],\
                        'strio27':self.str[2,7],\
                        'strio28':self.str[2,8],\
                        'strio32':self.str[3,2],\
                        'strio33':self.str[3,3],\
                        'strio34':self.str[3,4],\
                        'strio35':self.str[3,5],\
                        'strio36':self.str[3,6],\
                        'strio37':self.str[3,7],\
                        'strio38':self.str[3,8],\
                        'strio42':self.str[4,2],\
                        'strio43':self.str[4,3],\
                        'strio44':self.str[4,4],\
                        'strio45':self.str[4,5],\
                        'strio46':self.str[4,6],\
                        'strio47':self.str[4,7],\
                        'strio48':self.str[4,8],\
                        'strio52':self.str[5,2],\
                        'strio53':self.str[5,3],\
                        'strio54':self.str[5,4],\
                        'strio55':self.str[5,5],\
                        'strio56':self.str[5,6],\
                        'strio57':self.str[5,7],\
                        'strio58':self.str[5,8],\
                        'strio62':self.str[6,2],\
                        'strio63':self.str[6,3],\
                        'strio64':self.str[6,4],\
                        'strio65':self.str[6,5],\
                        'strio66':self.str[6,6],\
                        'strio67':self.str[6,7],\
                        'strio68':self.str[6,8],\
                        'strio72':self.str[7,2],\
                        'strio73':self.str[7,3],\
                        'strio74':self.str[7,4],\
                        'strio75':self.str[7,5],\
                        'strio76':self.str[7,6],\
                        'strio77':self.str[7,7],\
                        'strio78':self.str[7,8],\
                        'strio82':self.str[8,2],\
                        'strio83':self.str[8,3],\
                        'strio84':self.str[8,4],\
                        'strio85':self.str[8,5],\
                        'strio86':self.str[8,6],\
                        'strio87':self.str[8,7],\
                        'strio88':self.str[8,8],\


                        'DA22':self.da[2,2],\
                        'DA23':self.da[2,3],\
                        'DA24':self.da[2,4],\
                        'DA25':self.da[2,5],\
                        'DA26':self.da[2,6],\
                        'DA27':self.da[2,7],\
                        'DA28':self.da[2,8],\
                        'DA32':self.da[3,2],\
                        'DA33':self.da[3,3],\
                        'DA34':self.da[3,4],\
                        'DA35':self.da[3,5],\
                        'DA36':self.da[3,6],\
                        'DA37':self.da[3,7],\
                        'DA38':self.da[3,8],\
                        'DA42':self.da[4,2],\
                        'DA43':self.da[4,3],\
                        'DA44':self.da[4,4],\
                        'DA45':self.da[4,5],\
                        'DA46':self.da[4,6],\
                        'DA47':self.da[4,7],\
                        'DA48':self.da[4,8],\
                        'DA52':self.da[5,2],\
                        'DA53':self.da[5,3],\
                        'DA54':self.da[5,4],\
                        'DA55':self.da[5,5],\
                        'DA56':self.da[5,6],\
                        'DA57':self.da[5,7],\
                        'DA58':self.da[5,8],\
                        'DA62':self.da[6,2],\
                        'DA63':self.da[6,3],\
                        'DA64':self.da[6,4],\
                        'DA65':self.da[6,5],\
                        'DA66':self.da[6,6],\
                        'DA67':self.da[6,7],\
                        'DA68':self.da[6,8],\
                        'DA72':self.da[7,2],\
                        'DA73':self.da[7,3],\
                        'DA74':self.da[7,4],\
                        'DA75':self.da[7,5],\
                        'DA76':self.da[7,6],\
                        'DA77':self.da[7,7],\
                        'DA78':self.da[7,8],\
                        'DA82':self.da[8,2],\
                        'DA83':self.da[8,3],\
                        'DA84':self.da[8,4],\
                        'DA85':self.da[8,5],\
                        'DA86':self.da[8,6],\
                        'DA87':self.da[8,7],\
                        'DA88':self.da[8,8],\
                           })


    def writer_csv(self,n):
        with open("sampleA-{}.csv".format((n)), "w", newline="") as f:
            fieldnames = ['episode', 'a_cnt', 'action', 's_row', 's_col', 'reward', 'DAsum']+\
                ['x'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['D1'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['D2'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['lc_gpe'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['gl_gpe'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['lc_hyper'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['gl_hyper'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['SNr'+str(i)+str(j)+'U' for i in range(2,9) for j in range(2,9)]+\
                ['SNr'+str(i)+str(j)+'R' for i in range(2,9) for j in range(2,9)]+\
                ['SNr'+str(i)+str(j)+'L' for i in range(2,9) for j in range(2,9)]+\
                ['SNr'+str(i)+str(j)+'D' for i in range(2,9) for j in range(2,9)]+\
                ['SNr'+str(i)+str(j)+'S' for i in range(2,9) for j in range(2,9)]+\
                ['strio'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['DA'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]

            dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(self.memory)
