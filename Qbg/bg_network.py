import numpy as np
import random
import csv

class BG_Network():

    def __init__(self, env):
        """環境に応じたネットワーク構築"""
        self.action_num = 4 #行動の選択肢
        self.shape = env.shape + tuple([self.action_num]) #ネットワークの形状

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
        self.wstrio = np.ones(self.shape)*5 #大脳皮質 - ストリオソーム

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

        """細胞活動のリセット"""
        self.x = np.zeros(self.shape) #大脳皮質（入力）
        self.stn = np.zeros(self.shape) #視床下核（D5レセプター）
        self.d1 = np.zeros(self.shape) #D1レセプター
        self.d2 = np.zeros(self.shape) #D2レセプター
        self.gpe = np.zeros(self.shape) #淡蒼球外節
        self.snr = np.zeros(self.shape + tuple([self.action_num + 1])) #淡蒼球内節
        self.xs = np.zeros(self.shape + tuple([self.action_num])) #大脳皮質（再入力）
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

        """
        ハイパー直接路:大脳皮質→視床下核→淡蒼球内節
        """
        self.stn = self.x * self.wstn *self.stn_ratio
        self.lc_hyper = self.stn * self.lc_stn_ratio #局所的ハイパー直接路
        self.gl_hyper = self.stn * self.gl_stn_ratio #広域的ハイパー直接路
        for p in range(0, 4):
            self.snr[s+tuple([p, 4])] = self.snr[s+tuple([p, 4])] + self.lc_hyper[s+tuple([p])]
            self.snr[s+tuple([0, p])] = self.snr[s+tuple([0, p])] + self.lc_hyper[s+tuple([0])] + self.gl_hyper[u+tuple([p])]
            self.snr[s+tuple([1, p])] = self.snr[s+tuple([1, p])] + self.lc_hyper[s+tuple([1])] + self.gl_hyper[r+tuple([p])]
            self.snr[s+tuple([2, p])] = self.snr[s+tuple([2, p])] + self.lc_hyper[s+tuple([2])] + self.gl_hyper[l+tuple([p])]
            self.snr[s+tuple([3, p])] = self.snr[s+tuple([3, p])] + self.lc_hyper[s+tuple([3])] + self.gl_hyper[d+tuple([p])]

        """
        直接路:大脳皮質→D1→[抑制性]淡蒼球内節
        """
        self.d1 = self.x * self.wd1 * self.d1_ratio
        for p in range(0, 4):
            for q in range(0, 5):
                self.snr[s+tuple([p, q])] = self.snr[s+tuple([p, q])] - self.d1[s+tuple([p])]

        """
        関節路:大脳皮質→D2→[抑制性]淡蒼球外節→[抑制性]淡蒼球内節
        """
        self.d2 = self.x * self.wd2 * self.d2_ratio
        self.gpe = -self.d2
        self.lc_gpe = self.gpe * self.lc_gpe_ratio #局所的関節路
        self.gl_gpe = self.gpe * self.gl_gpe_ratio #広域的関節路
        for p in range(0, 4):
            self.snr[s+tuple([p, 4])] = self.snr[s+tuple([p, 4])] - self.lc_gpe[s+tuple([p])]
            self.snr[s+tuple([0, p])] = self.snr[s+tuple([0, p])] - self.lc_gpe[s+tuple([0])] - self.gl_gpe[u+tuple([p])]
            self.snr[s+tuple([1, p])] = self.snr[s+tuple([1, p])] - self.lc_gpe[s+tuple([1])] - self.gl_gpe[r+tuple([p])]
            self.snr[s+tuple([2, p])] = self.snr[s+tuple([2, p])] - self.lc_gpe[s+tuple([2])] - self.gl_gpe[l+tuple([p])]
            self.snr[s+tuple([3, p])] = self.snr[s+tuple([3, p])] - self.lc_gpe[s+tuple([3])] - self.gl_gpe[d+tuple([p])]

        """
        淡蒼球内接→[抑制性]大脳皮質
        """
        self.xs = -(self.snr - (self.lc_stn_ratio + self.gl_stn_ratio + self.gl_gpe_ratio))


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
            xx = np.array([np.max(self.xs[s+tuple([0])]), np.max(self.xs[s+tuple([1])]), np.max(self.xs[s+tuple([2])]), np.max(self.xs[s+tuple([3])])])
            
#             '''softmax_ver'''
#             xx = 400 * xx
#             c = np.max(xx)
#             exp_x = np.exp(xx - c)
#             sum_exp_x = np.sum(exp_x)
#             p = exp_x / sum_exp_x
#             max_index = np.random.choice(np.arange(len(xx)), p=p)
#             return action[max_index]
            
            
            max_x = [i for i, x in enumerate(xx) if x == max(xx)]
            max_x_index = random.choices(max_x)
            return action[max_x_index[0]]


    def bg_loop_sh(self, ep, a_cnt, pre_state, state, action, reward):
        """位置情報と行動情報の取得"""
        if action[0]==-1:
            a=0
        elif action[1]==1:
            a=1
        elif action[1]==-1:
            a=2
        elif action[0]==1:
            a=3
        s = tuple([pre_state[0], pre_state[1], a]) #更新する大脳基底核

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

        # self.wstn[s] += 0.004 * self.da[s] * self.x[s] * self.stn[s]
        # self.wd1[s] += 0.004 * self.da[s] * self.x[s] * self.d1[s]
        # self.wd2[s] += 0.004 * self.da[s] * self.x[s] * self.d2[s]
        # self.wstrio[s] += 0.004 * self.da[s] * self.x[s] * self.strio[s+tuple([4])]
        
        time=1
        self.memorize(ep,a_cnt,time,action,pre_state,reward)


    def episode_fin(self, e):
        """episode毎の処理"""
        pass


    def memorize(self, ep, a_cnt, time, action, state, reward):
        self.memory.append({'episode':ep, 'a_cnt':a_cnt, 'time':time, 'action':action, 's_row':state[0], 's_col':state[1],\
                        'reward':reward, 'DAsum':self.da.sum(),\

                        'x22U':self.x[2,2,0],'x22R':self.x[2,2,1],'x22L':self.x[2,2,2],'x22D':self.x[2,2,3],\
                        'x23U':self.x[2,3,0],'x23R':self.x[2,3,1],'x23L':self.x[2,3,2],'x23D':self.x[2,3,3],\
                        'x24U':self.x[2,4,0],'x24R':self.x[2,4,1],'x24L':self.x[2,4,2],'x24D':self.x[2,4,3],\
                        'x25U':self.x[2,5,0],'x25R':self.x[2,5,1],'x25L':self.x[2,5,2],'x25D':self.x[2,5,3],\
                        'x26U':self.x[2,6,0],'x26R':self.x[2,6,1],'x26L':self.x[2,6,2],'x26D':self.x[2,6,3],\
                        'x32U':self.x[3,2,0],'x32R':self.x[3,2,1],'x32L':self.x[3,2,2],'x32D':self.x[3,2,3],\
                        'x33U':self.x[3,3,0],'x33R':self.x[3,3,1],'x33L':self.x[3,3,2],'x33D':self.x[3,3,3],\
                        'x34U':self.x[3,4,0],'x34R':self.x[3,4,1],'x34L':self.x[3,4,2],'x34D':self.x[3,4,3],\
                        'x35U':self.x[3,5,0],'x35R':self.x[3,5,1],'x35L':self.x[3,5,2],'x35D':self.x[3,5,3],\
                        'x36U':self.x[3,6,0],'x36R':self.x[3,6,1],'x36L':self.x[3,6,2],'x36D':self.x[3,6,3],\
                        'x42U':self.x[4,2,0],'x42R':self.x[4,2,1],'x42L':self.x[4,2,2],'x42D':self.x[4,2,3],\
                        'x43U':self.x[4,3,0],'x43R':self.x[4,3,1],'x43L':self.x[4,3,2],'x43D':self.x[4,3,3],\
                        'x44U':self.x[4,4,0],'x44R':self.x[4,4,1],'x44L':self.x[4,4,2],'x44D':self.x[4,4,3],\
                        'x45U':self.x[4,5,0],'x45R':self.x[4,5,1],'x45L':self.x[4,5,2],'x45D':self.x[4,5,3],\
                        'x46U':self.x[4,6,0],'x46R':self.x[4,6,1],'x46L':self.x[4,6,2],'x46D':self.x[4,6,3],\
                        'x52U':self.x[5,2,0],'x52R':self.x[5,2,1],'x52L':self.x[5,2,2],'x52D':self.x[5,2,3],\
                        'x53U':self.x[5,3,0],'x53R':self.x[5,3,1],'x53L':self.x[5,3,2],'x53D':self.x[5,3,3],\
                        'x54U':self.x[5,4,0],'x54R':self.x[5,4,1],'x54L':self.x[5,4,2],'x54D':self.x[5,4,3],\
                        'x55U':self.x[5,5,0],'x55R':self.x[5,5,1],'x55L':self.x[5,5,2],'x55D':self.x[5,5,3],\
                        'x56U':self.x[5,6,0],'x56R':self.x[5,6,1],'x56L':self.x[5,6,2],'x56D':self.x[5,6,3],\
                        'x62U':self.x[6,2,0],'x62R':self.x[6,2,1],'x62L':self.x[6,2,2],'x62D':self.x[6,2,3],\
                        'x63U':self.x[6,3,0],'x63R':self.x[6,3,1],'x63L':self.x[6,3,2],'x63D':self.x[6,3,3],\
                        'x64U':self.x[6,4,0],'x64R':self.x[6,4,1],'x64L':self.x[6,4,2],'x64D':self.x[6,4,3],\
                        'x65U':self.x[6,5,0],'x65R':self.x[6,5,1],'x65L':self.x[6,5,2],'x65D':self.x[6,5,3],\
                        'x66U':self.x[6,6,0],'x66R':self.x[6,6,1],'x66L':self.x[6,6,2],'x66D':self.x[6,6,3],\

                        'D122U':self.d1[2,2,0],'D122R':self.d1[2,2,1],'D122L':self.d1[2,2,2],'D122D':self.d1[2,2,3],\
                        'D123U':self.d1[2,3,0],'D123R':self.d1[2,3,1],'D123L':self.d1[2,3,2],'D123D':self.d1[2,3,3],\
                        'D124U':self.d1[2,4,0],'D124R':self.d1[2,4,1],'D124L':self.d1[2,4,2],'D124D':self.d1[2,4,3],\
                        'D125U':self.d1[2,5,0],'D125R':self.d1[2,5,1],'D125L':self.d1[2,5,2],'D125D':self.d1[2,5,3],\
                        'D126U':self.d1[2,6,0],'D126R':self.d1[2,6,1],'D126L':self.d1[2,6,2],'D126D':self.d1[2,6,3],\
                        'D132U':self.d1[3,2,0],'D132R':self.d1[3,2,1],'D132L':self.d1[3,2,2],'D132D':self.d1[3,2,3],\
                        'D133U':self.d1[3,3,0],'D133R':self.d1[3,3,1],'D133L':self.d1[3,3,2],'D133D':self.d1[3,3,3],\
                        'D134U':self.d1[3,4,0],'D134R':self.d1[3,4,1],'D134L':self.d1[3,4,2],'D134D':self.d1[3,4,3],\
                        'D135U':self.d1[3,5,0],'D135R':self.d1[3,5,1],'D135L':self.d1[3,5,2],'D135D':self.d1[3,5,3],\
                        'D136U':self.d1[3,6,0],'D136R':self.d1[3,6,1],'D136L':self.d1[3,6,2],'D136D':self.d1[3,6,3],\
                        'D142U':self.d1[4,2,0],'D142R':self.d1[4,2,1],'D142L':self.d1[4,2,2],'D142D':self.d1[4,2,3],\
                        'D143U':self.d1[4,3,0],'D143R':self.d1[4,3,1],'D143L':self.d1[4,3,2],'D143D':self.d1[4,3,3],\
                        'D144U':self.d1[4,4,0],'D144R':self.d1[4,4,1],'D144L':self.d1[4,4,2],'D144D':self.d1[4,4,3],\
                        'D145U':self.d1[4,5,0],'D145R':self.d1[4,5,1],'D145L':self.d1[4,5,2],'D145D':self.d1[4,5,3],\
                        'D146U':self.d1[4,6,0],'D146R':self.d1[4,6,1],'D146L':self.d1[4,6,2],'D146D':self.d1[4,6,3],\
                        'D152U':self.d1[5,2,0],'D152R':self.d1[5,2,1],'D152L':self.d1[5,2,2],'D152D':self.d1[5,2,3],\
                        'D153U':self.d1[5,3,0],'D153R':self.d1[5,3,1],'D153L':self.d1[5,3,2],'D153D':self.d1[5,3,3],\
                        'D154U':self.d1[5,4,0],'D154R':self.d1[5,4,1],'D154L':self.d1[5,4,2],'D154D':self.d1[5,4,3],\
                        'D155U':self.d1[5,5,0],'D155R':self.d1[5,5,1],'D155L':self.d1[5,5,2],'D155D':self.d1[5,5,3],\
                        'D156U':self.d1[5,6,0],'D156R':self.d1[5,6,1],'D156L':self.d1[5,6,2],'D156D':self.d1[5,6,3],\
                        'D162U':self.d1[6,2,0],'D162R':self.d1[6,2,1],'D162L':self.d1[6,2,2],'D162D':self.d1[6,2,3],\
                        'D163U':self.d1[6,3,0],'D163R':self.d1[6,3,1],'D163L':self.d1[6,3,2],'D163D':self.d1[6,3,3],\
                        'D164U':self.d1[6,4,0],'D164R':self.d1[6,4,1],'D164L':self.d1[6,4,2],'D164D':self.d1[6,4,3],\
                        'D165U':self.d1[6,5,0],'D165R':self.d1[6,5,1],'D165L':self.d1[6,5,2],'D165D':self.d1[6,5,3],\
                        'D166U':self.d1[6,6,0],'D166R':self.d1[6,6,1],'D166L':self.d1[6,6,2],'D166D':self.d1[6,6,3],\

                        'D222U':self.d2[2,2,0],'D222R':self.d2[2,2,1],'D222L':self.d2[2,2,2],'D222D':self.d2[2,2,3],\
                        'D223U':self.d2[2,3,0],'D223R':self.d2[2,3,1],'D223L':self.d2[2,3,2],'D223D':self.d2[2,3,3],\
                        'D224U':self.d2[2,4,0],'D224R':self.d2[2,4,1],'D224L':self.d2[2,4,2],'D224D':self.d2[2,4,3],\
                        'D225U':self.d2[2,5,0],'D225R':self.d2[2,5,1],'D225L':self.d2[2,5,2],'D225D':self.d2[2,5,3],\
                        'D226U':self.d2[2,6,0],'D226R':self.d2[2,6,1],'D226L':self.d2[2,6,2],'D226D':self.d2[2,6,3],\
                        'D232U':self.d2[3,2,0],'D232R':self.d2[3,2,1],'D232L':self.d2[3,2,2],'D232D':self.d2[3,2,3],\
                        'D233U':self.d2[3,3,0],'D233R':self.d2[3,3,1],'D233L':self.d2[3,3,2],'D233D':self.d2[3,3,3],\
                        'D234U':self.d2[3,4,0],'D234R':self.d2[3,4,1],'D234L':self.d2[3,4,2],'D234D':self.d2[3,4,3],\
                        'D235U':self.d2[3,5,0],'D235R':self.d2[3,5,1],'D235L':self.d2[3,5,2],'D235D':self.d2[3,5,3],\
                        'D236U':self.d2[3,6,0],'D236R':self.d2[3,6,1],'D236L':self.d2[3,6,2],'D236D':self.d2[3,6,3],\
                        'D242U':self.d2[4,2,0],'D242R':self.d2[4,2,1],'D242L':self.d2[4,2,2],'D242D':self.d2[4,2,3],\
                        'D243U':self.d2[4,3,0],'D243R':self.d2[4,3,1],'D243L':self.d2[4,3,2],'D243D':self.d2[4,3,3],\
                        'D244U':self.d2[4,4,0],'D244R':self.d2[4,4,1],'D244L':self.d2[4,4,2],'D244D':self.d2[4,4,3],\
                        'D245U':self.d2[4,5,0],'D245R':self.d2[4,5,1],'D245L':self.d2[4,5,2],'D245D':self.d2[4,5,3],\
                        'D246U':self.d2[4,6,0],'D246R':self.d2[4,6,1],'D246L':self.d2[4,6,2],'D246D':self.d2[4,6,3],\
                        'D252U':self.d2[5,2,0],'D252R':self.d2[5,2,1],'D252L':self.d2[5,2,2],'D252D':self.d2[5,2,3],\
                        'D253U':self.d2[5,3,0],'D253R':self.d2[5,3,1],'D253L':self.d2[5,3,2],'D253D':self.d2[5,3,3],\
                        'D254U':self.d2[5,4,0],'D254R':self.d2[5,4,1],'D254L':self.d2[5,4,2],'D254D':self.d2[5,4,3],\
                        'D255U':self.d2[5,5,0],'D255R':self.d2[5,5,1],'D255L':self.d2[5,5,2],'D255D':self.d2[5,5,3],\
                        'D256U':self.d2[5,6,0],'D256R':self.d2[5,6,1],'D256L':self.d2[5,6,2],'D256D':self.d2[5,6,3],\
                        'D262U':self.d2[6,2,0],'D262R':self.d2[6,2,1],'D262L':self.d2[6,2,2],'D262D':self.d2[6,2,3],\
                        'D263U':self.d2[6,3,0],'D263R':self.d2[6,3,1],'D263L':self.d2[6,3,2],'D263D':self.d2[6,3,3],\
                        'D264U':self.d2[6,4,0],'D264R':self.d2[6,4,1],'D264L':self.d2[6,4,2],'D264D':self.d2[6,4,3],\
                        'D265U':self.d2[6,5,0],'D265R':self.d2[6,5,1],'D265L':self.d2[6,5,2],'D265D':self.d2[6,5,3],\
                        'D266U':self.d2[6,6,0],'D266R':self.d2[6,6,1],'D266L':self.d2[6,6,2],'D266D':self.d2[6,6,3],\

                        'lc_gpe22U':self.lc_gpe[2,2,0],'lc_gpe22R':self.lc_gpe[2,2,1],'lc_gpe22L':self.lc_gpe[2,2,2],'lc_gpe22D':self.lc_gpe[2,2,3],\
                        'lc_gpe23U':self.lc_gpe[2,3,0],'lc_gpe23R':self.lc_gpe[2,3,1],'lc_gpe23L':self.lc_gpe[2,3,2],'lc_gpe23D':self.lc_gpe[2,3,3],\
                        'lc_gpe24U':self.lc_gpe[2,4,0],'lc_gpe24R':self.lc_gpe[2,4,1],'lc_gpe24L':self.lc_gpe[2,4,2],'lc_gpe24D':self.lc_gpe[2,4,3],\
                        'lc_gpe25U':self.lc_gpe[2,5,0],'lc_gpe25R':self.lc_gpe[2,5,1],'lc_gpe25L':self.lc_gpe[2,5,2],'lc_gpe25D':self.lc_gpe[2,5,3],\
                        'lc_gpe26U':self.lc_gpe[2,6,0],'lc_gpe26R':self.lc_gpe[2,6,1],'lc_gpe26L':self.lc_gpe[2,6,2],'lc_gpe26D':self.lc_gpe[2,6,3],\
                        'lc_gpe32U':self.lc_gpe[3,2,0],'lc_gpe32R':self.lc_gpe[3,2,1],'lc_gpe32L':self.lc_gpe[3,2,2],'lc_gpe32D':self.lc_gpe[3,2,3],\
                        'lc_gpe33U':self.lc_gpe[3,3,0],'lc_gpe33R':self.lc_gpe[3,3,1],'lc_gpe33L':self.lc_gpe[3,3,2],'lc_gpe33D':self.lc_gpe[3,3,3],\
                        'lc_gpe34U':self.lc_gpe[3,4,0],'lc_gpe34R':self.lc_gpe[3,4,1],'lc_gpe34L':self.lc_gpe[3,4,2],'lc_gpe34D':self.lc_gpe[3,4,3],\
                        'lc_gpe35U':self.lc_gpe[3,5,0],'lc_gpe35R':self.lc_gpe[3,5,1],'lc_gpe35L':self.lc_gpe[3,5,2],'lc_gpe35D':self.lc_gpe[3,5,3],\
                        'lc_gpe36U':self.lc_gpe[3,6,0],'lc_gpe36R':self.lc_gpe[3,6,1],'lc_gpe36L':self.lc_gpe[3,6,2],'lc_gpe36D':self.lc_gpe[3,6,3],\
                        'lc_gpe42U':self.lc_gpe[4,2,0],'lc_gpe42R':self.lc_gpe[4,2,1],'lc_gpe42L':self.lc_gpe[4,2,2],'lc_gpe42D':self.lc_gpe[4,2,3],\
                        'lc_gpe43U':self.lc_gpe[4,3,0],'lc_gpe43R':self.lc_gpe[4,3,1],'lc_gpe43L':self.lc_gpe[4,3,2],'lc_gpe43D':self.lc_gpe[4,3,3],\
                        'lc_gpe44U':self.lc_gpe[4,4,0],'lc_gpe44R':self.lc_gpe[4,4,1],'lc_gpe44L':self.lc_gpe[4,4,2],'lc_gpe44D':self.lc_gpe[4,4,3],\
                        'lc_gpe45U':self.lc_gpe[4,5,0],'lc_gpe45R':self.lc_gpe[4,5,1],'lc_gpe45L':self.lc_gpe[4,5,2],'lc_gpe45D':self.lc_gpe[4,5,3],\
                        'lc_gpe46U':self.lc_gpe[4,6,0],'lc_gpe46R':self.lc_gpe[4,6,1],'lc_gpe46L':self.lc_gpe[4,6,2],'lc_gpe46D':self.lc_gpe[4,6,3],\
                        'lc_gpe52U':self.lc_gpe[5,2,0],'lc_gpe52R':self.lc_gpe[5,2,1],'lc_gpe52L':self.lc_gpe[5,2,2],'lc_gpe52D':self.lc_gpe[5,2,3],\
                        'lc_gpe53U':self.lc_gpe[5,3,0],'lc_gpe53R':self.lc_gpe[5,3,1],'lc_gpe53L':self.lc_gpe[5,3,2],'lc_gpe53D':self.lc_gpe[5,3,3],\
                        'lc_gpe54U':self.lc_gpe[5,4,0],'lc_gpe54R':self.lc_gpe[5,4,1],'lc_gpe54L':self.lc_gpe[5,4,2],'lc_gpe54D':self.lc_gpe[5,4,3],\
                        'lc_gpe55U':self.lc_gpe[5,5,0],'lc_gpe55R':self.lc_gpe[5,5,1],'lc_gpe55L':self.lc_gpe[5,5,2],'lc_gpe55D':self.lc_gpe[5,5,3],\
                        'lc_gpe56U':self.lc_gpe[5,6,0],'lc_gpe56R':self.lc_gpe[5,6,1],'lc_gpe56L':self.lc_gpe[5,6,2],'lc_gpe56D':self.lc_gpe[5,6,3],\
                        'lc_gpe62U':self.lc_gpe[6,2,0],'lc_gpe62R':self.lc_gpe[6,2,1],'lc_gpe62L':self.lc_gpe[6,2,2],'lc_gpe62D':self.lc_gpe[6,2,3],\
                        'lc_gpe63U':self.lc_gpe[6,3,0],'lc_gpe63R':self.lc_gpe[6,3,1],'lc_gpe63L':self.lc_gpe[6,3,2],'lc_gpe63D':self.lc_gpe[6,3,3],\
                        'lc_gpe64U':self.lc_gpe[6,4,0],'lc_gpe64R':self.lc_gpe[6,4,1],'lc_gpe64L':self.lc_gpe[6,4,2],'lc_gpe64D':self.lc_gpe[6,4,3],\
                        'lc_gpe65U':self.lc_gpe[6,5,0],'lc_gpe65R':self.lc_gpe[6,5,1],'lc_gpe65L':self.lc_gpe[6,5,2],'lc_gpe65D':self.lc_gpe[6,5,3],\
                        'lc_gpe66U':self.lc_gpe[6,6,0],'lc_gpe66R':self.lc_gpe[6,6,1],'lc_gpe66L':self.lc_gpe[6,6,2],'lc_gpe66D':self.lc_gpe[6,6,3],\

                        'gl_gpe22U':self.gl_gpe[2,2,0],'gl_gpe22R':self.gl_gpe[2,2,1],'gl_gpe22L':self.gl_gpe[2,2,2],'gl_gpe22D':self.gl_gpe[2,2,3],\
                        'gl_gpe23U':self.gl_gpe[2,3,0],'gl_gpe23R':self.gl_gpe[2,3,1],'gl_gpe23L':self.gl_gpe[2,3,2],'gl_gpe23D':self.gl_gpe[2,3,3],\
                        'gl_gpe24U':self.gl_gpe[2,4,0],'gl_gpe24R':self.gl_gpe[2,4,1],'gl_gpe24L':self.gl_gpe[2,4,2],'gl_gpe24D':self.gl_gpe[2,4,3],\
                        'gl_gpe25U':self.gl_gpe[2,5,0],'gl_gpe25R':self.gl_gpe[2,5,1],'gl_gpe25L':self.gl_gpe[2,5,2],'gl_gpe25D':self.gl_gpe[2,5,3],\
                        'gl_gpe26U':self.gl_gpe[2,6,0],'gl_gpe26R':self.gl_gpe[2,6,1],'gl_gpe26L':self.gl_gpe[2,6,2],'gl_gpe26D':self.gl_gpe[2,6,3],\
                        'gl_gpe32U':self.gl_gpe[3,2,0],'gl_gpe32R':self.gl_gpe[3,2,1],'gl_gpe32L':self.gl_gpe[3,2,2],'gl_gpe32D':self.gl_gpe[3,2,3],\
                        'gl_gpe33U':self.gl_gpe[3,3,0],'gl_gpe33R':self.gl_gpe[3,3,1],'gl_gpe33L':self.gl_gpe[3,3,2],'gl_gpe33D':self.gl_gpe[3,3,3],\
                        'gl_gpe34U':self.gl_gpe[3,4,0],'gl_gpe34R':self.gl_gpe[3,4,1],'gl_gpe34L':self.gl_gpe[3,4,2],'gl_gpe34D':self.gl_gpe[3,4,3],\
                        'gl_gpe35U':self.gl_gpe[3,5,0],'gl_gpe35R':self.gl_gpe[3,5,1],'gl_gpe35L':self.gl_gpe[3,5,2],'gl_gpe35D':self.gl_gpe[3,5,3],\
                        'gl_gpe36U':self.gl_gpe[3,6,0],'gl_gpe36R':self.gl_gpe[3,6,1],'gl_gpe36L':self.gl_gpe[3,6,2],'gl_gpe36D':self.gl_gpe[3,6,3],\
                        'gl_gpe42U':self.gl_gpe[4,2,0],'gl_gpe42R':self.gl_gpe[4,2,1],'gl_gpe42L':self.gl_gpe[4,2,2],'gl_gpe42D':self.gl_gpe[4,2,3],\
                        'gl_gpe43U':self.gl_gpe[4,3,0],'gl_gpe43R':self.gl_gpe[4,3,1],'gl_gpe43L':self.gl_gpe[4,3,2],'gl_gpe43D':self.gl_gpe[4,3,3],\
                        'gl_gpe44U':self.gl_gpe[4,4,0],'gl_gpe44R':self.gl_gpe[4,4,1],'gl_gpe44L':self.gl_gpe[4,4,2],'gl_gpe44D':self.gl_gpe[4,4,3],\
                        'gl_gpe45U':self.gl_gpe[4,5,0],'gl_gpe45R':self.gl_gpe[4,5,1],'gl_gpe45L':self.gl_gpe[4,5,2],'gl_gpe45D':self.gl_gpe[4,5,3],\
                        'gl_gpe46U':self.gl_gpe[4,6,0],'gl_gpe46R':self.gl_gpe[4,6,1],'gl_gpe46L':self.gl_gpe[4,6,2],'gl_gpe46D':self.gl_gpe[4,6,3],\
                        'gl_gpe52U':self.gl_gpe[5,2,0],'gl_gpe52R':self.gl_gpe[5,2,1],'gl_gpe52L':self.gl_gpe[5,2,2],'gl_gpe52D':self.gl_gpe[5,2,3],\
                        'gl_gpe53U':self.gl_gpe[5,3,0],'gl_gpe53R':self.gl_gpe[5,3,1],'gl_gpe53L':self.gl_gpe[5,3,2],'gl_gpe53D':self.gl_gpe[5,3,3],\
                        'gl_gpe54U':self.gl_gpe[5,4,0],'gl_gpe54R':self.gl_gpe[5,4,1],'gl_gpe54L':self.gl_gpe[5,4,2],'gl_gpe54D':self.gl_gpe[5,4,3],\
                        'gl_gpe55U':self.gl_gpe[5,5,0],'gl_gpe55R':self.gl_gpe[5,5,1],'gl_gpe55L':self.gl_gpe[5,5,2],'gl_gpe55D':self.gl_gpe[5,5,3],\
                        'gl_gpe56U':self.gl_gpe[5,6,0],'gl_gpe56R':self.gl_gpe[5,6,1],'gl_gpe56L':self.gl_gpe[5,6,2],'gl_gpe56D':self.gl_gpe[5,6,3],\
                        'gl_gpe62U':self.gl_gpe[6,2,0],'gl_gpe62R':self.gl_gpe[6,2,1],'gl_gpe62L':self.gl_gpe[6,2,2],'gl_gpe62D':self.gl_gpe[6,2,3],\
                        'gl_gpe63U':self.gl_gpe[6,3,0],'gl_gpe63R':self.gl_gpe[6,3,1],'gl_gpe63L':self.gl_gpe[6,3,2],'gl_gpe63D':self.gl_gpe[6,3,3],\
                        'gl_gpe64U':self.gl_gpe[6,4,0],'gl_gpe64R':self.gl_gpe[6,4,1],'gl_gpe64L':self.gl_gpe[6,4,2],'gl_gpe64D':self.gl_gpe[6,4,3],\
                        'gl_gpe65U':self.gl_gpe[6,5,0],'gl_gpe65R':self.gl_gpe[6,5,1],'gl_gpe65L':self.gl_gpe[6,5,2],'gl_gpe65D':self.gl_gpe[6,5,3],\
                        'gl_gpe66U':self.gl_gpe[6,6,0],'gl_gpe66R':self.gl_gpe[6,6,1],'gl_gpe66L':self.gl_gpe[6,6,2],'gl_gpe66D':self.gl_gpe[6,6,3],\

                        'lc_hyper22U':self.lc_hyper[2,2,0],'lc_hyper22R':self.lc_hyper[2,2,1],'lc_hyper22L':self.lc_hyper[2,2,2],'lc_hyper22D':self.lc_hyper[2,2,3],\
                        'lc_hyper23U':self.lc_hyper[2,3,0],'lc_hyper23R':self.lc_hyper[2,3,1],'lc_hyper23L':self.lc_hyper[2,3,2],'lc_hyper23D':self.lc_hyper[2,3,3],\
                        'lc_hyper24U':self.lc_hyper[2,4,0],'lc_hyper24R':self.lc_hyper[2,4,1],'lc_hyper24L':self.lc_hyper[2,4,2],'lc_hyper24D':self.lc_hyper[2,4,3],\
                        'lc_hyper25U':self.lc_hyper[2,5,0],'lc_hyper25R':self.lc_hyper[2,5,1],'lc_hyper25L':self.lc_hyper[2,5,2],'lc_hyper25D':self.lc_hyper[2,5,3],\
                        'lc_hyper26U':self.lc_hyper[2,6,0],'lc_hyper26R':self.lc_hyper[2,6,1],'lc_hyper26L':self.lc_hyper[2,6,2],'lc_hyper26D':self.lc_hyper[2,6,3],\
                        'lc_hyper32U':self.lc_hyper[3,2,0],'lc_hyper32R':self.lc_hyper[3,2,1],'lc_hyper32L':self.lc_hyper[3,2,2],'lc_hyper32D':self.lc_hyper[3,2,3],\
                        'lc_hyper33U':self.lc_hyper[3,3,0],'lc_hyper33R':self.lc_hyper[3,3,1],'lc_hyper33L':self.lc_hyper[3,3,2],'lc_hyper33D':self.lc_hyper[3,3,3],\
                        'lc_hyper34U':self.lc_hyper[3,4,0],'lc_hyper34R':self.lc_hyper[3,4,1],'lc_hyper34L':self.lc_hyper[3,4,2],'lc_hyper34D':self.lc_hyper[3,4,3],\
                        'lc_hyper35U':self.lc_hyper[3,5,0],'lc_hyper35R':self.lc_hyper[3,5,1],'lc_hyper35L':self.lc_hyper[3,5,2],'lc_hyper35D':self.lc_hyper[3,5,3],\
                        'lc_hyper36U':self.lc_hyper[3,6,0],'lc_hyper36R':self.lc_hyper[3,6,1],'lc_hyper36L':self.lc_hyper[3,6,2],'lc_hyper36D':self.lc_hyper[3,6,3],\
                        'lc_hyper42U':self.lc_hyper[4,2,0],'lc_hyper42R':self.lc_hyper[4,2,1],'lc_hyper42L':self.lc_hyper[4,2,2],'lc_hyper42D':self.lc_hyper[4,2,3],\
                        'lc_hyper43U':self.lc_hyper[4,3,0],'lc_hyper43R':self.lc_hyper[4,3,1],'lc_hyper43L':self.lc_hyper[4,3,2],'lc_hyper43D':self.lc_hyper[4,3,3],\
                        'lc_hyper44U':self.lc_hyper[4,4,0],'lc_hyper44R':self.lc_hyper[4,4,1],'lc_hyper44L':self.lc_hyper[4,4,2],'lc_hyper44D':self.lc_hyper[4,4,3],\
                        'lc_hyper45U':self.lc_hyper[4,5,0],'lc_hyper45R':self.lc_hyper[4,5,1],'lc_hyper45L':self.lc_hyper[4,5,2],'lc_hyper45D':self.lc_hyper[4,5,3],\
                        'lc_hyper46U':self.lc_hyper[4,6,0],'lc_hyper46R':self.lc_hyper[4,6,1],'lc_hyper46L':self.lc_hyper[4,6,2],'lc_hyper46D':self.lc_hyper[4,6,3],\
                        'lc_hyper52U':self.lc_hyper[5,2,0],'lc_hyper52R':self.lc_hyper[5,2,1],'lc_hyper52L':self.lc_hyper[5,2,2],'lc_hyper52D':self.lc_hyper[5,2,3],\
                        'lc_hyper53U':self.lc_hyper[5,3,0],'lc_hyper53R':self.lc_hyper[5,3,1],'lc_hyper53L':self.lc_hyper[5,3,2],'lc_hyper53D':self.lc_hyper[5,3,3],\
                        'lc_hyper54U':self.lc_hyper[5,4,0],'lc_hyper54R':self.lc_hyper[5,4,1],'lc_hyper54L':self.lc_hyper[5,4,2],'lc_hyper54D':self.lc_hyper[5,4,3],\
                        'lc_hyper55U':self.lc_hyper[5,5,0],'lc_hyper55R':self.lc_hyper[5,5,1],'lc_hyper55L':self.lc_hyper[5,5,2],'lc_hyper55D':self.lc_hyper[5,5,3],\
                        'lc_hyper56U':self.lc_hyper[5,6,0],'lc_hyper56R':self.lc_hyper[5,6,1],'lc_hyper56L':self.lc_hyper[5,6,2],'lc_hyper56D':self.lc_hyper[5,6,3],\
                        'lc_hyper62U':self.lc_hyper[6,2,0],'lc_hyper62R':self.lc_hyper[6,2,1],'lc_hyper62L':self.lc_hyper[6,2,2],'lc_hyper62D':self.lc_hyper[6,2,3],\
                        'lc_hyper63U':self.lc_hyper[6,3,0],'lc_hyper63R':self.lc_hyper[6,3,1],'lc_hyper63L':self.lc_hyper[6,3,2],'lc_hyper63D':self.lc_hyper[6,3,3],\
                        'lc_hyper64U':self.lc_hyper[6,4,0],'lc_hyper64R':self.lc_hyper[6,4,1],'lc_hyper64L':self.lc_hyper[6,4,2],'lc_hyper64D':self.lc_hyper[6,4,3],\
                        'lc_hyper65U':self.lc_hyper[6,5,0],'lc_hyper65R':self.lc_hyper[6,5,1],'lc_hyper65L':self.lc_hyper[6,5,2],'lc_hyper65D':self.lc_hyper[6,5,3],\
                        'lc_hyper66U':self.lc_hyper[6,6,0],'lc_hyper66R':self.lc_hyper[6,6,1],'lc_hyper66L':self.lc_hyper[6,6,2],'lc_hyper66D':self.lc_hyper[6,6,3],\

                        'gl_hyper22U':self.gl_hyper[2,2,0],'gl_hyper22R':self.gl_hyper[2,2,1],'gl_hyper22L':self.gl_hyper[2,2,2],'gl_hyper22D':self.gl_hyper[2,2,3],\
                        'gl_hyper23U':self.gl_hyper[2,3,0],'gl_hyper23R':self.gl_hyper[2,3,1],'gl_hyper23L':self.gl_hyper[2,3,2],'gl_hyper23D':self.gl_hyper[2,3,3],\
                        'gl_hyper24U':self.gl_hyper[2,4,0],'gl_hyper24R':self.gl_hyper[2,4,1],'gl_hyper24L':self.gl_hyper[2,4,2],'gl_hyper24D':self.gl_hyper[2,4,3],\
                        'gl_hyper25U':self.gl_hyper[2,5,0],'gl_hyper25R':self.gl_hyper[2,5,1],'gl_hyper25L':self.gl_hyper[2,5,2],'gl_hyper25D':self.gl_hyper[2,5,3],\
                        'gl_hyper26U':self.gl_hyper[2,6,0],'gl_hyper26R':self.gl_hyper[2,6,1],'gl_hyper26L':self.gl_hyper[2,6,2],'gl_hyper26D':self.gl_hyper[2,6,3],\
                        'gl_hyper32U':self.gl_hyper[3,2,0],'gl_hyper32R':self.gl_hyper[3,2,1],'gl_hyper32L':self.gl_hyper[3,2,2],'gl_hyper32D':self.gl_hyper[3,2,3],\
                        'gl_hyper33U':self.gl_hyper[3,3,0],'gl_hyper33R':self.gl_hyper[3,3,1],'gl_hyper33L':self.gl_hyper[3,3,2],'gl_hyper33D':self.gl_hyper[3,3,3],\
                        'gl_hyper34U':self.gl_hyper[3,4,0],'gl_hyper34R':self.gl_hyper[3,4,1],'gl_hyper34L':self.gl_hyper[3,4,2],'gl_hyper34D':self.gl_hyper[3,4,3],\
                        'gl_hyper35U':self.gl_hyper[3,5,0],'gl_hyper35R':self.gl_hyper[3,5,1],'gl_hyper35L':self.gl_hyper[3,5,2],'gl_hyper35D':self.gl_hyper[3,5,3],\
                        'gl_hyper36U':self.gl_hyper[3,6,0],'gl_hyper36R':self.gl_hyper[3,6,1],'gl_hyper36L':self.gl_hyper[3,6,2],'gl_hyper36D':self.gl_hyper[3,6,3],\
                        'gl_hyper42U':self.gl_hyper[4,2,0],'gl_hyper42R':self.gl_hyper[4,2,1],'gl_hyper42L':self.gl_hyper[4,2,2],'gl_hyper42D':self.gl_hyper[4,2,3],\
                        'gl_hyper43U':self.gl_hyper[4,3,0],'gl_hyper43R':self.gl_hyper[4,3,1],'gl_hyper43L':self.gl_hyper[4,3,2],'gl_hyper43D':self.gl_hyper[4,3,3],\
                        'gl_hyper44U':self.gl_hyper[4,4,0],'gl_hyper44R':self.gl_hyper[4,4,1],'gl_hyper44L':self.gl_hyper[4,4,2],'gl_hyper44D':self.gl_hyper[4,4,3],\
                        'gl_hyper45U':self.gl_hyper[4,5,0],'gl_hyper45R':self.gl_hyper[4,5,1],'gl_hyper45L':self.gl_hyper[4,5,2],'gl_hyper45D':self.gl_hyper[4,5,3],\
                        'gl_hyper46U':self.gl_hyper[4,6,0],'gl_hyper46R':self.gl_hyper[4,6,1],'gl_hyper46L':self.gl_hyper[4,6,2],'gl_hyper46D':self.gl_hyper[4,6,3],\
                        'gl_hyper52U':self.gl_hyper[5,2,0],'gl_hyper52R':self.gl_hyper[5,2,1],'gl_hyper52L':self.gl_hyper[5,2,2],'gl_hyper52D':self.gl_hyper[5,2,3],\
                        'gl_hyper53U':self.gl_hyper[5,3,0],'gl_hyper53R':self.gl_hyper[5,3,1],'gl_hyper53L':self.gl_hyper[5,3,2],'gl_hyper53D':self.gl_hyper[5,3,3],\
                        'gl_hyper54U':self.gl_hyper[5,4,0],'gl_hyper54R':self.gl_hyper[5,4,1],'gl_hyper54L':self.gl_hyper[5,4,2],'gl_hyper54D':self.gl_hyper[5,4,3],\
                        'gl_hyper55U':self.gl_hyper[5,5,0],'gl_hyper55R':self.gl_hyper[5,5,1],'gl_hyper55L':self.gl_hyper[5,5,2],'gl_hyper55D':self.gl_hyper[5,5,3],\
                        'gl_hyper56U':self.gl_hyper[5,6,0],'gl_hyper56R':self.gl_hyper[5,6,1],'gl_hyper56L':self.gl_hyper[5,6,2],'gl_hyper56D':self.gl_hyper[5,6,3],\
                        'gl_hyper62U':self.gl_hyper[6,2,0],'gl_hyper62R':self.gl_hyper[6,2,1],'gl_hyper62L':self.gl_hyper[6,2,2],'gl_hyper62D':self.gl_hyper[6,2,3],\
                        'gl_hyper63U':self.gl_hyper[6,3,0],'gl_hyper63R':self.gl_hyper[6,3,1],'gl_hyper63L':self.gl_hyper[6,3,2],'gl_hyper63D':self.gl_hyper[6,3,3],\
                        'gl_hyper64U':self.gl_hyper[6,4,0],'gl_hyper64R':self.gl_hyper[6,4,1],'gl_hyper64L':self.gl_hyper[6,4,2],'gl_hyper64D':self.gl_hyper[6,4,3],\
                        'gl_hyper65U':self.gl_hyper[6,5,0],'gl_hyper65R':self.gl_hyper[6,5,1],'gl_hyper65L':self.gl_hyper[6,5,2],'gl_hyper65D':self.gl_hyper[6,5,3],\
                        'gl_hyper66U':self.gl_hyper[6,6,0],'gl_hyper66R':self.gl_hyper[6,6,1],'gl_hyper66L':self.gl_hyper[6,6,2],'gl_hyper66D':self.gl_hyper[6,6,3],\

                        'SNr22UU':self.snr[2,2,0,0],'SNr22UR':self.snr[2,2,1,0],'SNr22UL':self.snr[2,2,2,0],'SNr22UD':self.snr[2,2,3,0],\
                        'SNr23UU':self.snr[2,3,0,0],'SNr23UR':self.snr[2,3,1,0],'SNr23UL':self.snr[2,3,2,0],'SNr23UD':self.snr[2,3,3,0],\
                        'SNr24UU':self.snr[2,4,0,0],'SNr24UR':self.snr[2,4,1,0],'SNr24UL':self.snr[2,4,2,0],'SNr24UD':self.snr[2,4,3,0],\
                        'SNr25UU':self.snr[2,5,0,0],'SNr25UR':self.snr[2,5,1,0],'SNr25UL':self.snr[2,5,2,0],'SNr25UD':self.snr[2,5,3,0],\
                        'SNr26UU':self.snr[2,6,0,0],'SNr26UR':self.snr[2,6,1,0],'SNr26UL':self.snr[2,6,2,0],'SNr26UD':self.snr[2,6,3,0],\
                        'SNr32UU':self.snr[3,2,0,0],'SNr32UR':self.snr[3,2,1,0],'SNr32UL':self.snr[3,2,2,0],'SNr32UD':self.snr[3,2,3,0],\
                        'SNr33UU':self.snr[3,3,0,0],'SNr33UR':self.snr[3,3,1,0],'SNr33UL':self.snr[3,3,2,0],'SNr33UD':self.snr[3,3,3,0],\
                        'SNr34UU':self.snr[3,4,0,0],'SNr34UR':self.snr[3,4,1,0],'SNr34UL':self.snr[3,4,2,0],'SNr34UD':self.snr[3,4,3,0],\
                        'SNr35UU':self.snr[3,5,0,0],'SNr35UR':self.snr[3,5,1,0],'SNr35UL':self.snr[3,5,2,0],'SNr35UD':self.snr[3,5,3,0],\
                        'SNr36UU':self.snr[3,6,0,0],'SNr36UR':self.snr[3,6,1,0],'SNr36UL':self.snr[3,6,2,0],'SNr36UD':self.snr[3,6,3,0],\
                        'SNr42UU':self.snr[4,2,0,0],'SNr42UR':self.snr[4,2,1,0],'SNr42UL':self.snr[4,2,2,0],'SNr42UD':self.snr[4,2,3,0],\
                        'SNr43UU':self.snr[4,3,0,0],'SNr43UR':self.snr[4,3,1,0],'SNr43UL':self.snr[4,3,2,0],'SNr43UD':self.snr[4,3,3,0],\
                        'SNr44UU':self.snr[4,4,0,0],'SNr44UR':self.snr[4,4,1,0],'SNr44UL':self.snr[4,4,2,0],'SNr44UD':self.snr[4,4,3,0],\
                        'SNr45UU':self.snr[4,5,0,0],'SNr45UR':self.snr[4,5,1,0],'SNr45UL':self.snr[4,5,2,0],'SNr45UD':self.snr[4,5,3,0],\
                        'SNr46UU':self.snr[4,6,0,0],'SNr46UR':self.snr[4,6,1,0],'SNr46UL':self.snr[4,6,2,0],'SNr46UD':self.snr[4,6,3,0],\
                        'SNr52UU':self.snr[5,2,0,0],'SNr52UR':self.snr[5,2,1,0],'SNr52UL':self.snr[5,2,2,0],'SNr52UD':self.snr[5,2,3,0],\
                        'SNr53UU':self.snr[5,3,0,0],'SNr53UR':self.snr[5,3,1,0],'SNr53UL':self.snr[5,3,2,0],'SNr53UD':self.snr[5,3,3,0],\
                        'SNr54UU':self.snr[5,4,0,0],'SNr54UR':self.snr[5,4,1,0],'SNr54UL':self.snr[5,4,2,0],'SNr54UD':self.snr[5,4,3,0],\
                        'SNr55UU':self.snr[5,5,0,0],'SNr55UR':self.snr[5,5,1,0],'SNr55UL':self.snr[5,5,2,0],'SNr55UD':self.snr[5,5,3,0],\
                        'SNr56UU':self.snr[5,6,0,0],'SNr56UR':self.snr[5,6,1,0],'SNr56UL':self.snr[5,6,2,0],'SNr56UD':self.snr[5,6,3,0],\
                        'SNr62UU':self.snr[6,2,0,0],'SNr62UR':self.snr[6,2,1,0],'SNr62UL':self.snr[6,2,2,0],'SNr62UD':self.snr[6,2,3,0],\
                        'SNr63UU':self.snr[6,3,0,0],'SNr63UR':self.snr[6,3,1,0],'SNr63UL':self.snr[6,3,2,0],'SNr63UD':self.snr[6,3,3,0],\
                        'SNr64UU':self.snr[6,4,0,0],'SNr64UR':self.snr[6,4,1,0],'SNr64UL':self.snr[6,4,2,0],'SNr64UD':self.snr[6,4,3,0],\
                        'SNr65UU':self.snr[6,5,0,0],'SNr65UR':self.snr[6,5,1,0],'SNr65UL':self.snr[6,5,2,0],'SNr65UD':self.snr[6,5,3,0],\
                        'SNr66UU':self.snr[6,6,0,0],'SNr66UR':self.snr[6,6,1,0],'SNr66UL':self.snr[6,6,2,0],'SNr66UD':self.snr[6,6,3,0],\

                        'SNr22RU':self.snr[2,2,0,1],'SNr22RR':self.snr[2,2,1,1],'SNr22RL':self.snr[2,2,2,1],'SNr22RD':self.snr[2,2,3,1],\
                        'SNr23RU':self.snr[2,3,0,1],'SNr23RR':self.snr[2,3,1,1],'SNr23RL':self.snr[2,3,2,1],'SNr23RD':self.snr[2,3,3,1],\
                        'SNr24RU':self.snr[2,4,0,1],'SNr24RR':self.snr[2,4,1,1],'SNr24RL':self.snr[2,4,2,1],'SNr24RD':self.snr[2,4,3,1],\
                        'SNr25RU':self.snr[2,5,0,1],'SNr25RR':self.snr[2,5,1,1],'SNr25RL':self.snr[2,5,2,1],'SNr25RD':self.snr[2,5,3,1],\
                        'SNr26RU':self.snr[2,6,0,1],'SNr26RR':self.snr[2,6,1,1],'SNr26RL':self.snr[2,6,2,1],'SNr26RD':self.snr[2,6,3,1],\
                        'SNr32RU':self.snr[3,2,0,1],'SNr32RR':self.snr[3,2,1,1],'SNr32RL':self.snr[3,2,2,1],'SNr32RD':self.snr[3,2,3,1],\
                        'SNr33RU':self.snr[3,3,0,1],'SNr33RR':self.snr[3,3,1,1],'SNr33RL':self.snr[3,3,2,1],'SNr33RD':self.snr[3,3,3,1],\
                        'SNr34RU':self.snr[3,4,0,1],'SNr34RR':self.snr[3,4,1,1],'SNr34RL':self.snr[3,4,2,1],'SNr34RD':self.snr[3,4,3,1],\
                        'SNr35RU':self.snr[3,5,0,1],'SNr35RR':self.snr[3,5,1,1],'SNr35RL':self.snr[3,5,2,1],'SNr35RD':self.snr[3,5,3,1],\
                        'SNr36RU':self.snr[3,6,0,1],'SNr36RR':self.snr[3,6,1,1],'SNr36RL':self.snr[3,6,2,1],'SNr36RD':self.snr[3,6,3,1],\
                        'SNr42RU':self.snr[4,2,0,1],'SNr42RR':self.snr[4,2,1,1],'SNr42RL':self.snr[4,2,2,1],'SNr42RD':self.snr[4,2,3,1],\
                        'SNr43RU':self.snr[4,3,0,1],'SNr43RR':self.snr[4,3,1,1],'SNr43RL':self.snr[4,3,2,1],'SNr43RD':self.snr[4,3,3,1],\
                        'SNr44RU':self.snr[4,4,0,1],'SNr44RR':self.snr[4,4,1,1],'SNr44RL':self.snr[4,4,2,1],'SNr44RD':self.snr[4,4,3,1],\
                        'SNr45RU':self.snr[4,5,0,1],'SNr45RR':self.snr[4,5,1,1],'SNr45RL':self.snr[4,5,2,1],'SNr45RD':self.snr[4,5,3,1],\
                        'SNr46RU':self.snr[4,6,0,1],'SNr46RR':self.snr[4,6,1,1],'SNr46RL':self.snr[4,6,2,1],'SNr46RD':self.snr[4,6,3,1],\
                        'SNr52RU':self.snr[5,2,0,1],'SNr52RR':self.snr[5,2,1,1],'SNr52RL':self.snr[5,2,2,1],'SNr52RD':self.snr[5,2,3,1],\
                        'SNr53RU':self.snr[5,3,0,1],'SNr53RR':self.snr[5,3,1,1],'SNr53RL':self.snr[5,3,2,1],'SNr53RD':self.snr[5,3,3,1],\
                        'SNr54RU':self.snr[5,4,0,1],'SNr54RR':self.snr[5,4,1,1],'SNr54RL':self.snr[5,4,2,1],'SNr54RD':self.snr[5,4,3,1],\
                        'SNr55RU':self.snr[5,5,0,1],'SNr55RR':self.snr[5,5,1,1],'SNr55RL':self.snr[5,5,2,1],'SNr55RD':self.snr[5,5,3,1],\
                        'SNr56RU':self.snr[5,6,0,1],'SNr56RR':self.snr[5,6,1,1],'SNr56RL':self.snr[5,6,2,1],'SNr56RD':self.snr[5,6,3,1],\
                        'SNr62RU':self.snr[6,2,0,1],'SNr62RR':self.snr[6,2,1,1],'SNr62RL':self.snr[6,2,2,1],'SNr62RD':self.snr[6,2,3,1],\
                        'SNr63RU':self.snr[6,3,0,1],'SNr63RR':self.snr[6,3,1,1],'SNr63RL':self.snr[6,3,2,1],'SNr63RD':self.snr[6,3,3,1],\
                        'SNr64RU':self.snr[6,4,0,1],'SNr64RR':self.snr[6,4,1,1],'SNr64RL':self.snr[6,4,2,1],'SNr64RD':self.snr[6,4,3,1],\
                        'SNr65RU':self.snr[6,5,0,1],'SNr65RR':self.snr[6,5,1,1],'SNr65RL':self.snr[6,5,2,1],'SNr65RD':self.snr[6,5,3,1],\
                        'SNr66RU':self.snr[6,6,0,1],'SNr66RR':self.snr[6,6,1,1],'SNr66RL':self.snr[6,6,2,1],'SNr66RD':self.snr[6,6,3,1],\

                        'SNr22LU':self.snr[2,2,0,2],'SNr22LR':self.snr[2,2,1,2],'SNr22LL':self.snr[2,2,2,2],'SNr22LD':self.snr[2,2,3,2],\
                        'SNr23LU':self.snr[2,3,0,2],'SNr23LR':self.snr[2,3,1,2],'SNr23LL':self.snr[2,3,2,2],'SNr23LD':self.snr[2,3,3,2],\
                        'SNr24LU':self.snr[2,4,0,2],'SNr24LR':self.snr[2,4,1,2],'SNr24LL':self.snr[2,4,2,2],'SNr24LD':self.snr[2,4,3,2],\
                        'SNr25LU':self.snr[2,5,0,2],'SNr25LR':self.snr[2,5,1,2],'SNr25LL':self.snr[2,5,2,2],'SNr25LD':self.snr[2,5,3,2],\
                        'SNr26LU':self.snr[2,6,0,2],'SNr26LR':self.snr[2,6,1,2],'SNr26LL':self.snr[2,6,2,2],'SNr26LD':self.snr[2,6,3,2],\
                        'SNr32LU':self.snr[3,2,0,2],'SNr32LR':self.snr[3,2,1,2],'SNr32LL':self.snr[3,2,2,2],'SNr32LD':self.snr[3,2,3,2],\
                        'SNr33LU':self.snr[3,3,0,2],'SNr33LR':self.snr[3,3,1,2],'SNr33LL':self.snr[3,3,2,2],'SNr33LD':self.snr[3,3,3,2],\
                        'SNr34LU':self.snr[3,4,0,2],'SNr34LR':self.snr[3,4,1,2],'SNr34LL':self.snr[3,4,2,2],'SNr34LD':self.snr[3,4,3,2],\
                        'SNr35LU':self.snr[3,5,0,2],'SNr35LR':self.snr[3,5,1,2],'SNr35LL':self.snr[3,5,2,2],'SNr35LD':self.snr[3,5,3,2],\
                        'SNr36LU':self.snr[3,6,0,2],'SNr36LR':self.snr[3,6,1,2],'SNr36LL':self.snr[3,6,2,2],'SNr36LD':self.snr[3,6,3,2],\
                        'SNr42LU':self.snr[4,2,0,2],'SNr42LR':self.snr[4,2,1,2],'SNr42LL':self.snr[4,2,2,2],'SNr42LD':self.snr[4,2,3,2],\
                        'SNr43LU':self.snr[4,3,0,2],'SNr43LR':self.snr[4,3,1,2],'SNr43LL':self.snr[4,3,2,2],'SNr43LD':self.snr[4,3,3,2],\
                        'SNr44LU':self.snr[4,4,0,2],'SNr44LR':self.snr[4,4,1,2],'SNr44LL':self.snr[4,4,2,2],'SNr44LD':self.snr[4,4,3,2],\
                        'SNr45LU':self.snr[4,5,0,2],'SNr45LR':self.snr[4,5,1,2],'SNr45LL':self.snr[4,5,2,2],'SNr45LD':self.snr[4,5,3,2],\
                        'SNr46LU':self.snr[4,6,0,2],'SNr46LR':self.snr[4,6,1,2],'SNr46LL':self.snr[4,6,2,2],'SNr46LD':self.snr[4,6,3,2],\
                        'SNr52LU':self.snr[5,2,0,2],'SNr52LR':self.snr[5,2,1,2],'SNr52LL':self.snr[5,2,2,2],'SNr52LD':self.snr[5,2,3,2],\
                        'SNr53LU':self.snr[5,3,0,2],'SNr53LR':self.snr[5,3,1,2],'SNr53LL':self.snr[5,3,2,2],'SNr53LD':self.snr[5,3,3,2],\
                        'SNr54LU':self.snr[5,4,0,2],'SNr54LR':self.snr[5,4,1,2],'SNr54LL':self.snr[5,4,2,2],'SNr54LD':self.snr[5,4,3,2],\
                        'SNr55LU':self.snr[5,5,0,2],'SNr55LR':self.snr[5,5,1,2],'SNr55LL':self.snr[5,5,2,2],'SNr55LD':self.snr[5,5,3,2],\
                        'SNr56LU':self.snr[5,6,0,2],'SNr56LR':self.snr[5,6,1,2],'SNr56LL':self.snr[5,6,2,2],'SNr56LD':self.snr[5,6,3,2],\
                        'SNr62LU':self.snr[6,2,0,2],'SNr62LR':self.snr[6,2,1,2],'SNr62LL':self.snr[6,2,2,2],'SNr62LD':self.snr[6,2,3,2],\
                        'SNr63LU':self.snr[6,3,0,2],'SNr63LR':self.snr[6,3,1,2],'SNr63LL':self.snr[6,3,2,2],'SNr63LD':self.snr[6,3,3,2],\
                        'SNr64LU':self.snr[6,4,0,2],'SNr64LR':self.snr[6,4,1,2],'SNr64LL':self.snr[6,4,2,2],'SNr64LD':self.snr[6,4,3,2],\
                        'SNr65LU':self.snr[6,5,0,2],'SNr65LR':self.snr[6,5,1,2],'SNr65LL':self.snr[6,5,2,2],'SNr65LD':self.snr[6,5,3,2],\
                        'SNr66LU':self.snr[6,6,0,2],'SNr66LR':self.snr[6,6,1,2],'SNr66LL':self.snr[6,6,2,2],'SNr66LD':self.snr[6,6,3,2],\

                        'SNr22DU':self.snr[2,2,0,3],'SNr22DR':self.snr[2,2,1,3],'SNr22DL':self.snr[2,2,2,3],'SNr22DD':self.snr[2,2,3,3],\
                        'SNr23DU':self.snr[2,3,0,3],'SNr23DR':self.snr[2,3,1,3],'SNr23DL':self.snr[2,3,2,3],'SNr23DD':self.snr[2,3,3,3],\
                        'SNr24DU':self.snr[2,4,0,3],'SNr24DR':self.snr[2,4,1,3],'SNr24DL':self.snr[2,4,2,3],'SNr24DD':self.snr[2,4,3,3],\
                        'SNr25DU':self.snr[2,5,0,3],'SNr25DR':self.snr[2,5,1,3],'SNr25DL':self.snr[2,5,2,3],'SNr25DD':self.snr[2,5,3,3],\
                        'SNr26DU':self.snr[2,6,0,3],'SNr26DR':self.snr[2,6,1,3],'SNr26DL':self.snr[2,6,2,3],'SNr26DD':self.snr[2,6,3,3],\
                        'SNr32DU':self.snr[3,2,0,3],'SNr32DR':self.snr[3,2,1,3],'SNr32DL':self.snr[3,2,2,3],'SNr32DD':self.snr[3,2,3,3],\
                        'SNr33DU':self.snr[3,3,0,3],'SNr33DR':self.snr[3,3,1,3],'SNr33DL':self.snr[3,3,2,3],'SNr33DD':self.snr[3,3,3,3],\
                        'SNr34DU':self.snr[3,4,0,3],'SNr34DR':self.snr[3,4,1,3],'SNr34DL':self.snr[3,4,2,3],'SNr34DD':self.snr[3,4,3,3],\
                        'SNr35DU':self.snr[3,5,0,3],'SNr35DR':self.snr[3,5,1,3],'SNr35DL':self.snr[3,5,2,3],'SNr35DD':self.snr[3,5,3,3],\
                        'SNr36DU':self.snr[3,6,0,3],'SNr36DR':self.snr[3,6,1,3],'SNr36DL':self.snr[3,6,2,3],'SNr36DD':self.snr[3,6,3,3],\
                        'SNr42DU':self.snr[4,2,0,3],'SNr42DR':self.snr[4,2,1,3],'SNr42DL':self.snr[4,2,2,3],'SNr42DD':self.snr[4,2,3,3],\
                        'SNr43DU':self.snr[4,3,0,3],'SNr43DR':self.snr[4,3,1,3],'SNr43DL':self.snr[4,3,2,3],'SNr43DD':self.snr[4,3,3,3],\
                        'SNr44DU':self.snr[4,4,0,3],'SNr44DR':self.snr[4,4,1,3],'SNr44DL':self.snr[4,4,2,3],'SNr44DD':self.snr[4,4,3,3],\
                        'SNr45DU':self.snr[4,5,0,3],'SNr45DR':self.snr[4,5,1,3],'SNr45DL':self.snr[4,5,2,3],'SNr45DD':self.snr[4,5,3,3],\
                        'SNr46DU':self.snr[4,6,0,3],'SNr46DR':self.snr[4,6,1,3],'SNr46DL':self.snr[4,6,2,3],'SNr46DD':self.snr[4,6,3,3],\
                        'SNr52DU':self.snr[5,2,0,3],'SNr52DR':self.snr[5,2,1,3],'SNr52DL':self.snr[5,2,2,3],'SNr52DD':self.snr[5,2,3,3],\
                        'SNr53DU':self.snr[5,3,0,3],'SNr53DR':self.snr[5,3,1,3],'SNr53DL':self.snr[5,3,2,3],'SNr53DD':self.snr[5,3,3,3],\
                        'SNr54DU':self.snr[5,4,0,3],'SNr54DR':self.snr[5,4,1,3],'SNr54DL':self.snr[5,4,2,3],'SNr54DD':self.snr[5,4,3,3],\
                        'SNr55DU':self.snr[5,5,0,3],'SNr55DR':self.snr[5,5,1,3],'SNr55DL':self.snr[5,5,2,3],'SNr55DD':self.snr[5,5,3,3],\
                        'SNr56DU':self.snr[5,6,0,3],'SNr56DR':self.snr[5,6,1,3],'SNr56DL':self.snr[5,6,2,3],'SNr56DD':self.snr[5,6,3,3],\
                        'SNr62DU':self.snr[6,2,0,3],'SNr62DR':self.snr[6,2,1,3],'SNr62DL':self.snr[6,2,2,3],'SNr62DD':self.snr[6,2,3,3],\
                        'SNr63DU':self.snr[6,3,0,3],'SNr63DR':self.snr[6,3,1,3],'SNr63DL':self.snr[6,3,2,3],'SNr63DD':self.snr[6,3,3,3],\
                        'SNr64DU':self.snr[6,4,0,3],'SNr64DR':self.snr[6,4,1,3],'SNr64DL':self.snr[6,4,2,3],'SNr64DD':self.snr[6,4,3,3],\
                        'SNr65DU':self.snr[6,5,0,3],'SNr65DR':self.snr[6,5,1,3],'SNr65DL':self.snr[6,5,2,3],'SNr65DD':self.snr[6,5,3,3],\
                        'SNr66DU':self.snr[6,6,0,3],'SNr66DR':self.snr[6,6,1,3],'SNr66DL':self.snr[6,6,2,3],'SNr66DD':self.snr[6,6,3,3],\

                        'SNr22SU':self.snr[2,2,0,4],'SNr22SR':self.snr[2,2,1,4],'SNr22SL':self.snr[2,2,2,4],'SNr22SD':self.snr[2,2,3,4],\
                        'SNr23SU':self.snr[2,3,0,4],'SNr23SR':self.snr[2,3,1,4],'SNr23SL':self.snr[2,3,2,4],'SNr23SD':self.snr[2,3,3,4],\
                        'SNr24SU':self.snr[2,4,0,4],'SNr24SR':self.snr[2,4,1,4],'SNr24SL':self.snr[2,4,2,4],'SNr24SD':self.snr[2,4,3,4],\
                        'SNr25SU':self.snr[2,5,0,4],'SNr25SR':self.snr[2,5,1,4],'SNr25SL':self.snr[2,5,2,4],'SNr25SD':self.snr[2,5,3,4],\
                        'SNr26SU':self.snr[2,6,0,4],'SNr26SR':self.snr[2,6,1,4],'SNr26SL':self.snr[2,6,2,4],'SNr26SD':self.snr[2,6,3,4],\
                        'SNr32SU':self.snr[3,2,0,4],'SNr32SR':self.snr[3,2,1,4],'SNr32SL':self.snr[3,2,2,4],'SNr32SD':self.snr[3,2,3,4],\
                        'SNr33SU':self.snr[3,3,0,4],'SNr33SR':self.snr[3,3,1,4],'SNr33SL':self.snr[3,3,2,4],'SNr33SD':self.snr[3,3,3,4],\
                        'SNr34SU':self.snr[3,4,0,4],'SNr34SR':self.snr[3,4,1,4],'SNr34SL':self.snr[3,4,2,4],'SNr34SD':self.snr[3,4,3,4],\
                        'SNr35SU':self.snr[3,5,0,4],'SNr35SR':self.snr[3,5,1,4],'SNr35SL':self.snr[3,5,2,4],'SNr35SD':self.snr[3,5,3,4],\
                        'SNr36SU':self.snr[3,6,0,4],'SNr36SR':self.snr[3,6,1,4],'SNr36SL':self.snr[3,6,2,4],'SNr36SD':self.snr[3,6,3,4],\
                        'SNr42SU':self.snr[4,2,0,4],'SNr42SR':self.snr[4,2,1,4],'SNr42SL':self.snr[4,2,2,4],'SNr42SD':self.snr[4,2,3,4],\
                        'SNr43SU':self.snr[4,3,0,4],'SNr43SR':self.snr[4,3,1,4],'SNr43SL':self.snr[4,3,2,4],'SNr43SD':self.snr[4,3,3,4],\
                        'SNr44SU':self.snr[4,4,0,4],'SNr44SR':self.snr[4,4,1,4],'SNr44SL':self.snr[4,4,2,4],'SNr44SD':self.snr[4,4,3,4],\
                        'SNr45SU':self.snr[4,5,0,4],'SNr45SR':self.snr[4,5,1,4],'SNr45SL':self.snr[4,5,2,4],'SNr45SD':self.snr[4,5,3,4],\
                        'SNr46SU':self.snr[4,6,0,4],'SNr46SR':self.snr[4,6,1,4],'SNr46SL':self.snr[4,6,2,4],'SNr46SD':self.snr[4,6,3,4],\
                        'SNr52SU':self.snr[5,2,0,4],'SNr52SR':self.snr[5,2,1,4],'SNr52SL':self.snr[5,2,2,4],'SNr52SD':self.snr[5,2,3,4],\
                        'SNr53SU':self.snr[5,3,0,4],'SNr53SR':self.snr[5,3,1,4],'SNr53SL':self.snr[5,3,2,4],'SNr53SD':self.snr[5,3,3,4],\
                        'SNr54SU':self.snr[5,4,0,4],'SNr54SR':self.snr[5,4,1,4],'SNr54SL':self.snr[5,4,2,4],'SNr54SD':self.snr[5,4,3,4],\
                        'SNr55SU':self.snr[5,5,0,4],'SNr55SR':self.snr[5,5,1,4],'SNr55SL':self.snr[5,5,2,4],'SNr55SD':self.snr[5,5,3,4],\
                        'SNr56SU':self.snr[5,6,0,4],'SNr56SR':self.snr[5,6,1,4],'SNr56SL':self.snr[5,6,2,4],'SNr56SD':self.snr[5,6,3,4],\
                        'SNr62SU':self.snr[6,2,0,4],'SNr62SR':self.snr[6,2,1,4],'SNr62SL':self.snr[6,2,2,4],'SNr62SD':self.snr[6,2,3,4],\
                        'SNr63SU':self.snr[6,3,0,4],'SNr63SR':self.snr[6,3,1,4],'SNr63SL':self.snr[6,3,2,4],'SNr63SD':self.snr[6,3,3,4],\
                        'SNr64SU':self.snr[6,4,0,4],'SNr64SR':self.snr[6,4,1,4],'SNr64SL':self.snr[6,4,2,4],'SNr64SD':self.snr[6,4,3,4],\
                        'SNr65SU':self.snr[6,5,0,4],'SNr65SR':self.snr[6,5,1,4],'SNr65SL':self.snr[6,5,2,4],'SNr65SD':self.snr[6,5,3,4],\
                        'SNr66SU':self.snr[6,6,0,4],'SNr66SR':self.snr[6,6,1,4],'SNr66SL':self.snr[6,6,2,4],'SNr66SD':self.snr[6,6,3,4],\

                        'strio22U':self.str[2,2,0],'strio22R':self.str[2,2,1],'strio22L':self.str[2,2,2],'strio22D':self.str[2,2,3],\
                        'strio23U':self.str[2,3,0],'strio23R':self.str[2,3,1],'strio23L':self.str[2,3,2],'strio23D':self.str[2,3,3],\
                        'strio24U':self.str[2,4,0],'strio24R':self.str[2,4,1],'strio24L':self.str[2,4,2],'strio24D':self.str[2,4,3],\
                        'strio25U':self.str[2,5,0],'strio25R':self.str[2,5,1],'strio25L':self.str[2,5,2],'strio25D':self.str[2,5,3],\
                        'strio26U':self.str[2,6,0],'strio26R':self.str[2,6,1],'strio26L':self.str[2,6,2],'strio26D':self.str[2,6,3],\
                        'strio32U':self.str[3,2,0],'strio32R':self.str[3,2,1],'strio32L':self.str[3,2,2],'strio32D':self.str[3,2,3],\
                        'strio33U':self.str[3,3,0],'strio33R':self.str[3,3,1],'strio33L':self.str[3,3,2],'strio33D':self.str[3,3,3],\
                        'strio34U':self.str[3,4,0],'strio34R':self.str[3,4,1],'strio34L':self.str[3,4,2],'strio34D':self.str[3,4,3],\
                        'strio35U':self.str[3,5,0],'strio35R':self.str[3,5,1],'strio35L':self.str[3,5,2],'strio35D':self.str[3,5,3],\
                        'strio36U':self.str[3,6,0],'strio36R':self.str[3,6,1],'strio36L':self.str[3,6,2],'strio36D':self.str[3,6,3],\
                        'strio42U':self.str[4,2,0],'strio42R':self.str[4,2,1],'strio42L':self.str[4,2,2],'strio42D':self.str[4,2,3],\
                        'strio43U':self.str[4,3,0],'strio43R':self.str[4,3,1],'strio43L':self.str[4,3,2],'strio43D':self.str[4,3,3],\
                        'strio44U':self.str[4,4,0],'strio44R':self.str[4,4,1],'strio44L':self.str[4,4,2],'strio44D':self.str[4,4,3],\
                        'strio45U':self.str[4,5,0],'strio45R':self.str[4,5,1],'strio45L':self.str[4,5,2],'strio45D':self.str[4,5,3],\
                        'strio46U':self.str[4,6,0],'strio46R':self.str[4,6,1],'strio46L':self.str[4,6,2],'strio46D':self.str[4,6,3],\
                        'strio52U':self.str[5,2,0],'strio52R':self.str[5,2,1],'strio52L':self.str[5,2,2],'strio52D':self.str[5,2,3],\
                        'strio53U':self.str[5,3,0],'strio53R':self.str[5,3,1],'strio53L':self.str[5,3,2],'strio53D':self.str[5,3,3],\
                        'strio54U':self.str[5,4,0],'strio54R':self.str[5,4,1],'strio54L':self.str[5,4,2],'strio54D':self.str[5,4,3],\
                        'strio55U':self.str[5,5,0],'strio55R':self.str[5,5,1],'strio55L':self.str[5,5,2],'strio55D':self.str[5,5,3],\
                        'strio56U':self.str[5,6,0],'strio56R':self.str[5,6,1],'strio56L':self.str[5,6,2],'strio56D':self.str[5,6,3],\
                        'strio62U':self.str[6,2,0],'strio62R':self.str[6,2,1],'strio62L':self.str[6,2,2],'strio62D':self.str[6,2,3],\
                        'strio63U':self.str[6,3,0],'strio63R':self.str[6,3,1],'strio63L':self.str[6,3,2],'strio63D':self.str[6,3,3],\
                        'strio64U':self.str[6,4,0],'strio64R':self.str[6,4,1],'strio64L':self.str[6,4,2],'strio64D':self.str[6,4,3],\
                        'strio65U':self.str[6,5,0],'strio65R':self.str[6,5,1],'strio65L':self.str[6,5,2],'strio65D':self.str[6,5,3],\
                        'strio66U':self.str[6,6,0],'strio66R':self.str[6,6,1],'strio66L':self.str[6,6,2],'strio66D':self.str[6,6,3],\

                        'DA22U':self.da[2,2,0],'DA22R':self.da[2,2,1],'DA22L':self.da[2,2,2],'DA22D':self.da[2,2,3],\
                        'DA23U':self.da[2,3,0],'DA23R':self.da[2,3,1],'DA23L':self.da[2,3,2],'DA23D':self.da[2,3,3],\
                        'DA24U':self.da[2,4,0],'DA24R':self.da[2,4,1],'DA24L':self.da[2,4,2],'DA24D':self.da[2,4,3],\
                        'DA25U':self.da[2,5,0],'DA25R':self.da[2,5,1],'DA25L':self.da[2,5,2],'DA25D':self.da[2,5,3],\
                        'DA26U':self.da[2,6,0],'DA26R':self.da[2,6,1],'DA26L':self.da[2,6,2],'DA26D':self.da[2,6,3],\
                        'DA32U':self.da[3,2,0],'DA32R':self.da[3,2,1],'DA32L':self.da[3,2,2],'DA32D':self.da[3,2,3],\
                        'DA33U':self.da[3,3,0],'DA33R':self.da[3,3,1],'DA33L':self.da[3,3,2],'DA33D':self.da[3,3,3],\
                        'DA34U':self.da[3,4,0],'DA34R':self.da[3,4,1],'DA34L':self.da[3,4,2],'DA34D':self.da[3,4,3],\
                        'DA35U':self.da[3,5,0],'DA35R':self.da[3,5,1],'DA35L':self.da[3,5,2],'DA35D':self.da[3,5,3],\
                        'DA36U':self.da[3,6,0],'DA36R':self.da[3,6,1],'DA36L':self.da[3,6,2],'DA36D':self.da[3,6,3],\
                        'DA42U':self.da[4,2,0],'DA42R':self.da[4,2,1],'DA42L':self.da[4,2,2],'DA42D':self.da[4,2,3],\
                        'DA43U':self.da[4,3,0],'DA43R':self.da[4,3,1],'DA43L':self.da[4,3,2],'DA43D':self.da[4,3,3],\
                        'DA44U':self.da[4,4,0],'DA44R':self.da[4,4,1],'DA44L':self.da[4,4,2],'DA44D':self.da[4,4,3],\
                        'DA45U':self.da[4,5,0],'DA45R':self.da[4,5,1],'DA45L':self.da[4,5,2],'DA45D':self.da[4,5,3],\
                        'DA46U':self.da[4,6,0],'DA46R':self.da[4,6,1],'DA46L':self.da[4,6,2],'DA46D':self.da[4,6,3],\
                        'DA52U':self.da[5,2,0],'DA52R':self.da[5,2,1],'DA52L':self.da[5,2,2],'DA52D':self.da[5,2,3],\
                        'DA53U':self.da[5,3,0],'DA53R':self.da[5,3,1],'DA53L':self.da[5,3,2],'DA53D':self.da[5,3,3],\
                        'DA54U':self.da[5,4,0],'DA54R':self.da[5,4,1],'DA54L':self.da[5,4,2],'DA54D':self.da[5,4,3],\
                        'DA55U':self.da[5,5,0],'DA55R':self.da[5,5,1],'DA55L':self.da[5,5,2],'DA55D':self.da[5,5,3],\
                        'DA56U':self.da[5,6,0],'DA56R':self.da[5,6,1],'DA56L':self.da[5,6,2],'DA56D':self.da[5,6,3],\
                        'DA62U':self.da[6,2,0],'DA62R':self.da[6,2,1],'DA62L':self.da[6,2,2],'DA62D':self.da[6,2,3],\
                        'DA63U':self.da[6,3,0],'DA63R':self.da[6,3,1],'DA63L':self.da[6,3,2],'DA63D':self.da[6,3,3],\
                        'DA64U':self.da[6,4,0],'DA64R':self.da[6,4,1],'DA64L':self.da[6,4,2],'DA64D':self.da[6,4,3],\
                        'DA65U':self.da[6,5,0],'DA65R':self.da[6,5,1],'DA65L':self.da[6,5,2],'DA65D':self.da[6,5,3],\
                        'DA66U':self.da[6,6,0],'DA66R':self.da[6,6,1],'DA66L':self.da[6,6,2],'DA66D':self.da[6,6,3],\

                        'wstn22U':self.wstn[2,2,0],'wstn22R':self.wstn[2,2,1],'wstn22L':self.wstn[2,2,2],'wstn22D':self.wstn[2,2,3],\
                        'wstn23U':self.wstn[2,3,0],'wstn23R':self.wstn[2,3,1],'wstn23L':self.wstn[2,3,2],'wstn23D':self.wstn[2,3,3],\
                        'wstn24U':self.wstn[2,4,0],'wstn24R':self.wstn[2,4,1],'wstn24L':self.wstn[2,4,2],'wstn24D':self.wstn[2,4,3],\
                        'wstn25U':self.wstn[2,5,0],'wstn25R':self.wstn[2,5,1],'wstn25L':self.wstn[2,5,2],'wstn25D':self.wstn[2,5,3],\
                        'wstn26U':self.wstn[2,6,0],'wstn26R':self.wstn[2,6,1],'wstn26L':self.wstn[2,6,2],'wstn26D':self.wstn[2,6,3],\
                        'wstn32U':self.wstn[3,2,0],'wstn32R':self.wstn[3,2,1],'wstn32L':self.wstn[3,2,2],'wstn32D':self.wstn[3,2,3],\
                        'wstn33U':self.wstn[3,3,0],'wstn33R':self.wstn[3,3,1],'wstn33L':self.wstn[3,3,2],'wstn33D':self.wstn[3,3,3],\
                        'wstn34U':self.wstn[3,4,0],'wstn34R':self.wstn[3,4,1],'wstn34L':self.wstn[3,4,2],'wstn34D':self.wstn[3,4,3],\
                        'wstn35U':self.wstn[3,5,0],'wstn35R':self.wstn[3,5,1],'wstn35L':self.wstn[3,5,2],'wstn35D':self.wstn[3,5,3],\
                        'wstn36U':self.wstn[3,6,0],'wstn36R':self.wstn[3,6,1],'wstn36L':self.wstn[3,6,2],'wstn36D':self.wstn[3,6,3],\
                        'wstn42U':self.wstn[4,2,0],'wstn42R':self.wstn[4,2,1],'wstn42L':self.wstn[4,2,2],'wstn42D':self.wstn[4,2,3],\
                        'wstn43U':self.wstn[4,3,0],'wstn43R':self.wstn[4,3,1],'wstn43L':self.wstn[4,3,2],'wstn43D':self.wstn[4,3,3],\
                        'wstn44U':self.wstn[4,4,0],'wstn44R':self.wstn[4,4,1],'wstn44L':self.wstn[4,4,2],'wstn44D':self.wstn[4,4,3],\
                        'wstn45U':self.wstn[4,5,0],'wstn45R':self.wstn[4,5,1],'wstn45L':self.wstn[4,5,2],'wstn45D':self.wstn[4,5,3],\
                        'wstn46U':self.wstn[4,6,0],'wstn46R':self.wstn[4,6,1],'wstn46L':self.wstn[4,6,2],'wstn46D':self.wstn[4,6,3],\
                        'wstn52U':self.wstn[5,2,0],'wstn52R':self.wstn[5,2,1],'wstn52L':self.wstn[5,2,2],'wstn52D':self.wstn[5,2,3],\
                        'wstn53U':self.wstn[5,3,0],'wstn53R':self.wstn[5,3,1],'wstn53L':self.wstn[5,3,2],'wstn53D':self.wstn[5,3,3],\
                        'wstn54U':self.wstn[5,4,0],'wstn54R':self.wstn[5,4,1],'wstn54L':self.wstn[5,4,2],'wstn54D':self.wstn[5,4,3],\
                        'wstn55U':self.wstn[5,5,0],'wstn55R':self.wstn[5,5,1],'wstn55L':self.wstn[5,5,2],'wstn55D':self.wstn[5,5,3],\
                        'wstn56U':self.wstn[5,6,0],'wstn56R':self.wstn[5,6,1],'wstn56L':self.wstn[5,6,2],'wstn56D':self.wstn[5,6,3],\
                        'wstn62U':self.wstn[6,2,0],'wstn62R':self.wstn[6,2,1],'wstn62L':self.wstn[6,2,2],'wstn62D':self.wstn[6,2,3],\
                        'wstn63U':self.wstn[6,3,0],'wstn63R':self.wstn[6,3,1],'wstn63L':self.wstn[6,3,2],'wstn63D':self.wstn[6,3,3],\
                        'wstn64U':self.wstn[6,4,0],'wstn64R':self.wstn[6,4,1],'wstn64L':self.wstn[6,4,2],'wstn64D':self.wstn[6,4,3],\
                        'wstn65U':self.wstn[6,5,0],'wstn65R':self.wstn[6,5,1],'wstn65L':self.wstn[6,5,2],'wstn65D':self.wstn[6,5,3],\
                        'wstn66U':self.wstn[6,6,0],'wstn66R':self.wstn[6,6,1],'wstn66L':self.wstn[6,6,2],'wstn66D':self.wstn[6,6,3],\

                        'wd122U':self.wd1[2,2,0],'wd122R':self.wd1[2,2,1],'wd122L':self.wd1[2,2,2],'wd122D':self.wd1[2,2,3],\
                        'wd123U':self.wd1[2,3,0],'wd123R':self.wd1[2,3,1],'wd123L':self.wd1[2,3,2],'wd123D':self.wd1[2,3,3],\
                        'wd124U':self.wd1[2,4,0],'wd124R':self.wd1[2,4,1],'wd124L':self.wd1[2,4,2],'wd124D':self.wd1[2,4,3],\
                        'wd125U':self.wd1[2,5,0],'wd125R':self.wd1[2,5,1],'wd125L':self.wd1[2,5,2],'wd125D':self.wd1[2,5,3],\
                        'wd126U':self.wd1[2,6,0],'wd126R':self.wd1[2,6,1],'wd126L':self.wd1[2,6,2],'wd126D':self.wd1[2,6,3],\
                        'wd132U':self.wd1[3,2,0],'wd132R':self.wd1[3,2,1],'wd132L':self.wd1[3,2,2],'wd132D':self.wd1[3,2,3],\
                        'wd133U':self.wd1[3,3,0],'wd133R':self.wd1[3,3,1],'wd133L':self.wd1[3,3,2],'wd133D':self.wd1[3,3,3],\
                        'wd134U':self.wd1[3,4,0],'wd134R':self.wd1[3,4,1],'wd134L':self.wd1[3,4,2],'wd134D':self.wd1[3,4,3],\
                        'wd135U':self.wd1[3,5,0],'wd135R':self.wd1[3,5,1],'wd135L':self.wd1[3,5,2],'wd135D':self.wd1[3,5,3],\
                        'wd136U':self.wd1[3,6,0],'wd136R':self.wd1[3,6,1],'wd136L':self.wd1[3,6,2],'wd136D':self.wd1[3,6,3],\
                        'wd142U':self.wd1[4,2,0],'wd142R':self.wd1[4,2,1],'wd142L':self.wd1[4,2,2],'wd142D':self.wd1[4,2,3],\
                        'wd143U':self.wd1[4,3,0],'wd143R':self.wd1[4,3,1],'wd143L':self.wd1[4,3,2],'wd143D':self.wd1[4,3,3],\
                        'wd144U':self.wd1[4,4,0],'wd144R':self.wd1[4,4,1],'wd144L':self.wd1[4,4,2],'wd144D':self.wd1[4,4,3],\
                        'wd145U':self.wd1[4,5,0],'wd145R':self.wd1[4,5,1],'wd145L':self.wd1[4,5,2],'wd145D':self.wd1[4,5,3],\
                        'wd146U':self.wd1[4,6,0],'wd146R':self.wd1[4,6,1],'wd146L':self.wd1[4,6,2],'wd146D':self.wd1[4,6,3],\
                        'wd152U':self.wd1[5,2,0],'wd152R':self.wd1[5,2,1],'wd152L':self.wd1[5,2,2],'wd152D':self.wd1[5,2,3],\
                        'wd153U':self.wd1[5,3,0],'wd153R':self.wd1[5,3,1],'wd153L':self.wd1[5,3,2],'wd153D':self.wd1[5,3,3],\
                        'wd154U':self.wd1[5,4,0],'wd154R':self.wd1[5,4,1],'wd154L':self.wd1[5,4,2],'wd154D':self.wd1[5,4,3],\
                        'wd155U':self.wd1[5,5,0],'wd155R':self.wd1[5,5,1],'wd155L':self.wd1[5,5,2],'wd155D':self.wd1[5,5,3],\
                        'wd156U':self.wd1[5,6,0],'wd156R':self.wd1[5,6,1],'wd156L':self.wd1[5,6,2],'wd156D':self.wd1[5,6,3],\
                        'wd162U':self.wd1[6,2,0],'wd162R':self.wd1[6,2,1],'wd162L':self.wd1[6,2,2],'wd162D':self.wd1[6,2,3],\
                        'wd163U':self.wd1[6,3,0],'wd163R':self.wd1[6,3,1],'wd163L':self.wd1[6,3,2],'wd163D':self.wd1[6,3,3],\
                        'wd164U':self.wd1[6,4,0],'wd164R':self.wd1[6,4,1],'wd164L':self.wd1[6,4,2],'wd164D':self.wd1[6,4,3],\
                        'wd165U':self.wd1[6,5,0],'wd165R':self.wd1[6,5,1],'wd165L':self.wd1[6,5,2],'wd165D':self.wd1[6,5,3],\
                        'wd166U':self.wd1[6,6,0],'wd166R':self.wd1[6,6,1],'wd166L':self.wd1[6,6,2],'wd166D':self.wd1[6,6,3]
                           })


    def writer_csv(self,n):
        with open("sampleA-{}.csv".format((n, self.gl_stn_ratio, self.gl_gpe_ratio)), "w", newline="") as f:
            fieldnames = ['episode', 'a_cnt', 'time', 'action', 's_row', 's_col', 'reward', 'DAsum']+\
                ['x'+str(i)+str(j)+'U' for i in range(2,7) for j in range(2,7)]+\
                ['x'+str(i)+str(j)+'R' for i in range(2,7) for j in range(2,7)]+\
                ['x'+str(i)+str(j)+'L' for i in range(2,7) for j in range(2,7)]+\
                ['x'+str(i)+str(j)+'D' for i in range(2,7) for j in range(2,7)]+\
                ['D1'+str(i)+str(j)+'U' for i in range(2,7) for j in range(2,7)]+\
                ['D1'+str(i)+str(j)+'R' for i in range(2,7) for j in range(2,7)]+\
                ['D1'+str(i)+str(j)+'L' for i in range(2,7) for j in range(2,7)]+\
                ['D1'+str(i)+str(j)+'D' for i in range(2,7) for j in range(2,7)]+\
                ['D2'+str(i)+str(j)+'U' for i in range(2,7) for j in range(2,7)]+\
                ['D2'+str(i)+str(j)+'R' for i in range(2,7) for j in range(2,7)]+\
                ['D2'+str(i)+str(j)+'L' for i in range(2,7) for j in range(2,7)]+\
                ['D2'+str(i)+str(j)+'D' for i in range(2,7) for j in range(2,7)]+\
                ['lc_gpe'+str(i)+str(j)+'U' for i in range(2,7) for j in range(2,7)]+\
                ['lc_gpe'+str(i)+str(j)+'R' for i in range(2,7) for j in range(2,7)]+\
                ['lc_gpe'+str(i)+str(j)+'L' for i in range(2,7) for j in range(2,7)]+\
                ['lc_gpe'+str(i)+str(j)+'D' for i in range(2,7) for j in range(2,7)]+\
                ['gl_gpe'+str(i)+str(j)+'U' for i in range(2,7) for j in range(2,7)]+\
                ['gl_gpe'+str(i)+str(j)+'R' for i in range(2,7) for j in range(2,7)]+\
                ['gl_gpe'+str(i)+str(j)+'L' for i in range(2,7) for j in range(2,7)]+\
                ['gl_gpe'+str(i)+str(j)+'D' for i in range(2,7) for j in range(2,7)]+\
                ['lc_hyper'+str(i)+str(j)+'U' for i in range(2,7) for j in range(2,7)]+\
                ['lc_hyper'+str(i)+str(j)+'R' for i in range(2,7) for j in range(2,7)]+\
                ['lc_hyper'+str(i)+str(j)+'L' for i in range(2,7) for j in range(2,7)]+\
                ['lc_hyper'+str(i)+str(j)+'D' for i in range(2,7) for j in range(2,7)]+\
                ['gl_hyper'+str(i)+str(j)+'U' for i in range(2,7) for j in range(2,7)]+\
                ['gl_hyper'+str(i)+str(j)+'R' for i in range(2,7) for j in range(2,7)]+\
                ['gl_hyper'+str(i)+str(j)+'L' for i in range(2,7) for j in range(2,7)]+\
                ['gl_hyper'+str(i)+str(j)+'D' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'U'+'U' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'U'+'R' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'U'+'L' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'U'+'D' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'R'+'U' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'R'+'R' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'R'+'L' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'R'+'D' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'L'+'U' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'L'+'R' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'L'+'L' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'L'+'D' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'D'+'U' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'D'+'R' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'D'+'L' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'D'+'D' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'S'+'U' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'S'+'R' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'S'+'L' for i in range(2,7) for j in range(2,7)]+\
                ['SNr'+str(i)+str(j)+'S'+'D' for i in range(2,7) for j in range(2,7)]+\
                ['strio'+str(i)+str(j)+'U' for i in range(2,7) for j in range(2,7)]+\
                ['strio'+str(i)+str(j)+'R' for i in range(2,7) for j in range(2,7)]+\
                ['strio'+str(i)+str(j)+'L' for i in range(2,7) for j in range(2,7)]+\
                ['strio'+str(i)+str(j)+'D' for i in range(2,7) for j in range(2,7)]+\
                ['DA'+str(i)+str(j)+'U' for i in range(2,7) for j in range(2,7)]+\
                ['DA'+str(i)+str(j)+'R' for i in range(2,7) for j in range(2,7)]+\
                ['DA'+str(i)+str(j)+'L' for i in range(2,7) for j in range(2,7)]+\
                ['DA'+str(i)+str(j)+'D' for i in range(2,7) for j in range(2,7)]+\
                ['wstn'+str(i)+str(j)+'U' for i in range(2,7) for j in range(2,7)]+\
                ['wstn'+str(i)+str(j)+'R' for i in range(2,7) for j in range(2,7)]+\
                ['wstn'+str(i)+str(j)+'L' for i in range(2,7) for j in range(2,7)]+\
                ['wstn'+str(i)+str(j)+'D' for i in range(2,7) for j in range(2,7)]+\
                ['wd1'+str(i)+str(j)+'U' for i in range(2,7) for j in range(2,7)]+\
                ['wd1'+str(i)+str(j)+'R' for i in range(2,7) for j in range(2,7)]+\
                ['wd1'+str(i)+str(j)+'L' for i in range(2,7) for j in range(2,7)]+\
                ['wd1'+str(i)+str(j)+'D' for i in range(2,7) for j in range(2,7)]
            
            dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(self.memory)
