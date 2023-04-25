import numpy as np
import math
import random
import csv

"""BGクラス"""

class BG_Network():

    def __init__(self, env):
        """環境に応じたネットワーク構築"""
        self.action_num = 2 #行動の選択肢
        self.shape = (env, self.action_num) #ネットワークの形状

        """細胞の設定"""
        self.x = np.zeros(self.shape) #大脳皮質（入力）
        self.stn = np.zeros(self.shape) #視床下核（D5レセプター）
        self.d1 = np.zeros(self.shape) #D1レセプター
        self.d2 = np.zeros(self.shape) #D2レセプター
        self.gpe = np.zeros(self.shape) #淡蒼球外節
        self.snr_s = np.zeros(self.shape)
        self.snr_g = np.zeros(self.shape)
        self.xs_s = np.zeros(self.shape)
        self.xs_g = np.zeros(self.shape)
        self.strio = np.zeros(self.shape) #ストリオソーム
        self.da = np.zeros(self.shape) #ドーパミン細胞

        """入力"""
        self.input = 1 #入力

        """重みパラメータ"""
        self.wstn = np.ones(self.shape) #大脳皮質 - 視床下核（D5レセプター）
        self.wd1 = np.ones(self.shape) #大脳皮質 - D1レセプター
        self.wd2 = np.ones(self.shape) #大脳皮質 - D2レセプター
        self.wstrio = np.ones(self.shape)*5 #大脳皮質 - ストリオソーム
        self.w = np.ones(self.shape + self.shape) #大脳皮質 - 大脳皮質

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
        self.memory=[] #メモリ


    def bg_loop_fh(self, state, pre_state, action, a_cnt):
        """位置情報の取得"""
        ps = tuple([pre_state])
        s = tuple([state]) #現在の位置
        r = tuple([state,0]) #右
        l = tuple([state,1]) #左

        """細胞活動のリセット"""
        self.x = np.zeros(self.shape) #大脳皮質（入力）
        self.stn = np.zeros(self.shape) #視床下核（D5レセプター）
        self.d1 = np.zeros(self.shape) #D1レセプター
        self.d2 = np.zeros(self.shape) #D2レセプター
        self.gpe = np.zeros(self.shape) #淡蒼球外節
        self.snr_s = np.zeros(self.shape)
        self.snr_g = np.zeros(self.shape)
        self.xs_s = np.zeros(self.shape)
        self.xs_g = np.zeros(self.shape)
        self.strio = np.zeros(self.shape) #ストリオソーム
        self.da = np.zeros(self.shape) #ドーパミン細胞

        """BG_loopツールのリセット"""
        self.lc_hyper = np.zeros(self.shape) #局所的ハイパー直接路
        self.gl_hyper = np.zeros(self.shape) #広域的ハイパー直接路
        self.lc_gpe = np.zeros(self.shape) #局所的淡蒼球外節（局所的関節路）
        self.gl_gpe = np.zeros(self.shape) #広域的淡蒼球外節（広域的関節路）

        """入力"""
        self.x[s] = self.input

        """
        ハイパー直接路:大脳皮質→視床下核→淡蒼球内節
        """
        self.stn = self.x * self.wstn * self.stn_ratio
        self.lc_hyper = self.stn * self.lc_stn_ratio #局所的ハイパー直接路
        self.gl_hyper = self.stn * self.gl_stn_ratio #広域的ハイパー直接路

        self.snr_s[s] = self.snr_s[s] + self.lc_hyper[s]

        """
        直接路:大脳皮質→D1→[抑制性]淡蒼球内節
        """
        self.d1 = self.x * self.wd1 * self.d1_ratio

        self.snr_s[s] = self.snr_s[s] - self.d1[s]

        """
        関節路:大脳皮質→D2→[抑制性]淡蒼球外節→[抑制性]淡蒼球内節
        """
        self.d2 = self.x * self.wd2 * self.d2_ratio
        self.gpe = -self.d2
        self.lc_gpe = self.gpe * self.lc_gpe_ratio #局所的関節路
        self.gl_gpe = self.gpe * self.gl_gpe_ratio #広域的関節路

        self.snr_s[s] = self.snr_s[s] - self.lc_gpe[s]

        """
        淡蒼球内接→[抑制性]大脳皮質
        """
        self.xs_s[s] = -np.round(self.snr_s[s] - (self.lc_stn_ratio), 10)

        # """
        # 大脳皮質間のヘブ則
        # """
        if a_cnt != 0:
            self.w[ps+tuple([action])+s] += abs(np.max(-(self.gl_hyper[s]+self.gl_gpe[s]-self.lc_stn_ratio)) * np.max(self.xs_s[s], axis=0))
            for i in range(0, 2):
                if self.w[ps+tuple([action])+s+tuple([i])] > 1:
                    self.w[ps+tuple([action])+s+tuple([i])] = 1


    def policy(self, ep, state):
        """位置情報と行動情報の取得"""
        r=tuple([state,0]) #右
        l=tuple([state,1]) #左

        """ε-greedyに基づく方策"""
        self.epsilon = 0.5 * (1 / (ep + 1e-8)) # 0.5:defo  3.0がよさげ 
        if np.random.random() < self.epsilon:
            a = random.choices([0, 1])
            action = a[0]
            return action
        else:
            xx = np.array([self.xs_s[r], self.xs_s[l]]) #cr,clのstate細胞の活動を比較
            max_x = [i for i, x in enumerate(xx) if x == max(xx)]
            action = random.choices(max_x)
            return action[0]


    def bg_loop_sh(self, pre_state, state, a_cnt, action, reward, ep, cart_pos, cart_v, pole_angle, pole_v):
        """位置情報と行動情報の取得"""
        ps = tuple([pre_state, action]) #更新する大脳基底核
        s = tuple([state])

        self.snr_g[ps] = self.snr_s[ps] + np.max((self.w[ps+s]*self.wstn[s]*self.stn_ratio*self.gl_stn_ratio)
            - (-self.w[ps+s]*self.wd2[s]*self.d2_ratio*self.gl_gpe_ratio))
        self.xs_g[ps] = -np.round(self.snr_g[ps] - (self.lc_stn_ratio+self.gl_gpe_ratio+self.gl_stn_ratio), 10)

        """
        大脳皮質→ストリオソーム→[抑制性]ドーパミン細胞
        """
        self.strio[ps] = self.xs_g[ps] * self.wstrio[ps]
        self.da[ps] = reward - self.strio[ps]

        """
        ドーパミンによるシナプス可塑性:ドーパミン細胞→ハイパー直接路, 直接路, 間接路
        """
        learning_rate = 0.01
        self.wstn[ps] += learning_rate * self.da[ps]
        self.wd1[ps] += learning_rate * self.da[ps]
        self.wd2[ps] += -learning_rate * self.da[ps]
        self.wstrio[ps] += 0.1*leraning_rate * self.da[ps]
#         print(self.wstrio[ps])
        if ep < 100 or ep % 50 == 0:
            self.memorize(ep, a_cnt, pre_state, action, reward, self.xs_s[ps], cart_pos, cart_v, pole_angle, pole_v)


    def episode_fin(self, e):
        """episode毎の処理"""


    def memorize(self, ep, a_cnt, pre_state, action, reward, q, cart_pos, cart_v, pole_angle, pole_v):
        #以下, 長いのでloadtxtにいずれ変更
        self.memory.append({'episode':ep,'a_cnt':a_cnt,'state':pre_state,'action':action,'reward':reward,\
                            'DAsum':self.da.sum(), 'SNr':q,\
                            'cart_pos':cart_pos,'cart_v':cart_v,'pole_angle':pole_angle,'pole_v':pole_v
                    })


    def writer_csv(self,n):
        with open("sampleA-{}.csv".format(n), "w", newline="") as f:
            fieldnames = ['episode','a_cnt','state','action','reward','DAsum','SNr','cart_pos','cart_v','pole_angle','pole_v']
            # ['snr'+str(i)+'R' for i in range(0,1296)]+\
            # ['snr'+str(i)+'L' for i in range(0,1296)]

            dict_writer = csv.DictWriter(f, fieldnames = fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(self.memory)

"""
                        'snr0R':self.snr_s[0, 0], 'snr0L':self.snr_s[0, 1], 'snr1R':self.snr_s[1, 0], 'snr1L':self.snr_s[1, 1], 
                        'snr2R':self.snr_s[2, 0], 'snr2L':self.snr_s[2, 1], 'snr3R':self.snr_s[3, 0], 'snr3L':self.snr_s[3, 1], 
                        'snr4R':self.snr_s[4, 0], 'snr4L':self.snr_s[4, 1], 'snr5R':self.snr_s[5, 0], 'snr5L':self.snr_s[5, 1], 
                        'snr6R':self.snr_s[6, 0], 'snr6L':self.snr_s[6, 1], 'snr7R':self.snr_s[7, 0], 'snr7L':self.snr_s[7, 1], 
                        'snr8R':self.snr_s[8, 0], 'snr8L':self.snr_s[8, 1], 'snr9R':self.snr_s[9, 0], 'snr9L':self.snr_s[9, 1],
                        'snr10R':self.snr_s[10, 0], 'snr10L':self.snr_s[10, 1], 'snr11R':self.snr_s[11, 0], 'snr11L':self.snr_s[11, 1], 
                        'snr12R':self.snr_s[12, 0], 'snr12L':self.snr_s[12, 1], 'snr13R':self.snr_s[13, 0], 'snr13L':self.snr_s[13, 1], 
                        'snr14R':self.snr_s[14, 0], 'snr14L':self.snr_s[14, 1], 'snr15R':self.snr_s[15, 0], 'snr15L':self.snr_s[15, 1], 
                        'snr16R':self.snr_s[16, 0], 'snr16L':self.snr_s[16, 1], 'snr17R':self.snr_s[17, 0], 'snr17L':self.snr_s[17, 1], 
                        'snr18R':self.snr_s[18, 0], 'snr18L':self.snr_s[18, 1], 'snr19R':self.snr_s[19, 0], 'snr19L':self.snr_s[19, 1], 
                        'snr20R':self.snr_s[20, 0], 'snr20L':self.snr_s[20, 1], 'snr21R':self.snr_s[21, 0], 'snr21L':self.snr_s[21, 1], 
                        'snr22R':self.snr_s[22, 0], 'snr22L':self.snr_s[22, 1], 'snr23R':self.snr_s[23, 0], 'snr23L':self.snr_s[23, 1], 
                        'snr24R':self.snr_s[24, 0], 'snr24L':self.snr_s[24, 1], 'snr25R':self.snr_s[25, 0], 'snr25L':self.snr_s[25, 1], 
                        'snr26R':self.snr_s[26, 0], 'snr26L':self.snr_s[26, 1], 'snr27R':self.snr_s[27, 0], 'snr27L':self.snr_s[27, 1], 
                        'snr28R':self.snr_s[28, 0], 'snr28L':self.snr_s[28, 1], 'snr29R':self.snr_s[29, 0], 'snr29L':self.snr_s[29, 1], 
                        'snr30R':self.snr_s[30, 0], 'snr30L':self.snr_s[30, 1], 'snr31R':self.snr_s[31, 0], 'snr31L':self.snr_s[31, 1], 
                        'snr32R':self.snr_s[32, 0], 'snr32L':self.snr_s[32, 1], 'snr33R':self.snr_s[33, 0], 'snr33L':self.snr_s[33, 1], 
                        'snr34R':self.snr_s[34, 0], 'snr34L':self.snr_s[34, 1], 'snr35R':self.snr_s[35, 0], 'snr35L':self.snr_s[35, 1], 
                        'snr36R':self.snr_s[36, 0], 'snr36L':self.snr_s[36, 1], 'snr37R':self.snr_s[37, 0], 'snr37L':self.snr_s[37, 1], 
                        'snr38R':self.snr_s[38, 0], 'snr38L':self.snr_s[38, 1], 'snr39R':self.snr_s[39, 0], 'snr39L':self.snr_s[39, 1], 
                        'snr40R':self.snr_s[40, 0], 'snr40L':self.snr_s[40, 1], 'snr41R':self.snr_s[41, 0], 'snr41L':self.snr_s[41, 1], 
                        'snr42R':self.snr_s[42, 0], 'snr42L':self.snr_s[42, 1], 'snr43R':self.snr_s[43, 0], 'snr43L':self.snr_s[43, 1], 
                        'snr44R':self.snr_s[44, 0], 'snr44L':self.snr_s[44, 1], 'snr45R':self.snr_s[45, 0], 'snr45L':self.snr_s[45, 1], 
                        'snr46R':self.snr_s[46, 0], 'snr46L':self.snr_s[46, 1], 'snr47R':self.snr_s[47, 0], 'snr47L':self.snr_s[47, 1], 
                        'snr48R':self.snr_s[48, 0], 'snr48L':self.snr_s[48, 1], 'snr49R':self.snr_s[49, 0], 'snr49L':self.snr_s[49, 1], 
                        'snr50R':self.snr_s[50, 0], 'snr50L':self.snr_s[50, 1], 'snr51R':self.snr_s[51, 0], 'snr51L':self.snr_s[51, 1], 
                        'snr52R':self.snr_s[52, 0], 'snr52L':self.snr_s[52, 1], 'snr53R':self.snr_s[53, 0], 'snr53L':self.snr_s[53, 1], 
                        'snr54R':self.snr_s[54, 0], 'snr54L':self.snr_s[54, 1], 'snr55R':self.snr_s[55, 0], 'snr55L':self.snr_s[55, 1], 
                        'snr56R':self.snr_s[56, 0], 'snr56L':self.snr_s[56, 1], 'snr57R':self.snr_s[57, 0], 'snr57L':self.snr_s[57, 1], 
                        'snr58R':self.snr_s[58, 0], 'snr58L':self.snr_s[58, 1], 'snr59R':self.snr_s[59, 0], 'snr59L':self.snr_s[59, 1], 
                        'snr60R':self.snr_s[60, 0], 'snr60L':self.snr_s[60, 1], 'snr61R':self.snr_s[61, 0], 'snr61L':self.snr_s[61, 1], 
                        'snr62R':self.snr_s[62, 0], 'snr62L':self.snr_s[62, 1], 'snr63R':self.snr_s[63, 0], 'snr63L':self.snr_s[63, 1], 
                        'snr64R':self.snr_s[64, 0], 'snr64L':self.snr_s[64, 1], 'snr65R':self.snr_s[65, 0], 'snr65L':self.snr_s[65, 1], 
                        'snr66R':self.snr_s[66, 0], 'snr66L':self.snr_s[66, 1], 'snr67R':self.snr_s[67, 0], 'snr67L':self.snr_s[67, 1], 
                        'snr68R':self.snr_s[68, 0], 'snr68L':self.snr_s[68, 1], 'snr69R':self.snr_s[69, 0], 'snr69L':self.snr_s[69, 1], 
                        'snr70R':self.snr_s[70, 0], 'snr70L':self.snr_s[70, 1], 'snr71R':self.snr_s[71, 0], 'snr71L':self.snr_s[71, 1], 
                        'snr72R':self.snr_s[72, 0], 'snr72L':self.snr_s[72, 1], 'snr73R':self.snr_s[73, 0], 'snr73L':self.snr_s[73, 1], 
                        'snr74R':self.snr_s[74, 0], 'snr74L':self.snr_s[74, 1], 'snr75R':self.snr_s[75, 0], 'snr75L':self.snr_s[75, 1], 
                        'snr76R':self.snr_s[76, 0], 'snr76L':self.snr_s[76, 1], 'snr77R':self.snr_s[77, 0], 'snr77L':self.snr_s[77, 1], 
                        'snr78R':self.snr_s[78, 0], 'snr78L':self.snr_s[78, 1], 'snr79R':self.snr_s[79, 0], 'snr79L':self.snr_s[79, 1], 
                        'snr80R':self.snr_s[80, 0], 'snr80L':self.snr_s[80, 1], 'snr81R':self.snr_s[81, 0], 'snr81L':self.snr_s[81, 1], 
                        'snr82R':self.snr_s[82, 0], 'snr82L':self.snr_s[82, 1], 'snr83R':self.snr_s[83, 0], 'snr83L':self.snr_s[83, 1], 
                        'snr84R':self.snr_s[84, 0], 'snr84L':self.snr_s[84, 1], 'snr85R':self.snr_s[85, 0], 'snr85L':self.snr_s[85, 1], 
                        'snr86R':self.snr_s[86, 0], 'snr86L':self.snr_s[86, 1], 'snr87R':self.snr_s[87, 0], 'snr87L':self.snr_s[87, 1], 
                        'snr88R':self.snr_s[88, 0], 'snr88L':self.snr_s[88, 1], 'snr89R':self.snr_s[89, 0], 'snr89L':self.snr_s[89, 1], 
                        'snr90R':self.snr_s[90, 0], 'snr90L':self.snr_s[90, 1], 'snr91R':self.snr_s[91, 0], 'snr91L':self.snr_s[91, 1], 
                        'snr92R':self.snr_s[92, 0], 'snr92L':self.snr_s[92, 1], 'snr93R':self.snr_s[93, 0], 'snr93L':self.snr_s[93, 1], 
                        'snr94R':self.snr_s[94, 0], 'snr94L':self.snr_s[94, 1], 'snr95R':self.snr_s[95, 0], 'snr95L':self.snr_s[95, 1], 
                        'snr96R':self.snr_s[96, 0], 'snr96L':self.snr_s[96, 1], 'snr97R':self.snr_s[97, 0], 'snr97L':self.snr_s[97, 1], 
                        'snr98R':self.snr_s[98, 0], 'snr98L':self.snr_s[98, 1], 'snr99R':self.snr_s[99, 0], 'snr99L':self.snr_s[99, 1], 
                        'snr100R':self.snr_s[100, 0], 'snr100L':self.snr_s[100, 1], 'snr101R':self.snr_s[101, 0], 'snr101L':self.snr_s[101, 1], 
                        'snr102R':self.snr_s[102, 0], 'snr102L':self.snr_s[102, 1], 'snr103R':self.snr_s[103, 0], 'snr103L':self.snr_s[103, 1], 
                        'snr104R':self.snr_s[104, 0], 'snr104L':self.snr_s[104, 1], 'snr105R':self.snr_s[105, 0], 'snr105L':self.snr_s[105, 1], 
                        'snr106R':self.snr_s[106, 0], 'snr106L':self.snr_s[106, 1], 'snr107R':self.snr_s[107, 0], 'snr107L':self.snr_s[107, 1], 
                        'snr108R':self.snr_s[108, 0], 'snr108L':self.snr_s[108, 1], 'snr109R':self.snr_s[109, 0], 'snr109L':self.snr_s[109, 1], 
                        'snr110R':self.snr_s[110, 0], 'snr110L':self.snr_s[110, 1], 'snr111R':self.snr_s[111, 0], 'snr111L':self.snr_s[111, 1], 
                        'snr112R':self.snr_s[112, 0], 'snr112L':self.snr_s[112, 1], 'snr113R':self.snr_s[113, 0], 'snr113L':self.snr_s[113, 1], 
                        'snr114R':self.snr_s[114, 0], 'snr114L':self.snr_s[114, 1], 'snr115R':self.snr_s[115, 0], 'snr115L':self.snr_s[115, 1], 
                        'snr116R':self.snr_s[116, 0], 'snr116L':self.snr_s[116, 1], 'snr117R':self.snr_s[117, 0], 'snr117L':self.snr_s[117, 1], 
                        'snr118R':self.snr_s[118, 0], 'snr118L':self.snr_s[118, 1], 'snr119R':self.snr_s[119, 0], 'snr119L':self.snr_s[119, 1], 
                        'snr120R':self.snr_s[120, 0], 'snr120L':self.snr_s[120, 1], 'snr121R':self.snr_s[121, 0], 'snr121L':self.snr_s[121, 1], 
                        'snr122R':self.snr_s[122, 0], 'snr122L':self.snr_s[122, 1], 'snr123R':self.snr_s[123, 0], 'snr123L':self.snr_s[123, 1], 
                        'snr124R':self.snr_s[124, 0], 'snr124L':self.snr_s[124, 1], 'snr125R':self.snr_s[125, 0], 'snr125L':self.snr_s[125, 1], 
                        'snr126R':self.snr_s[126, 0], 'snr126L':self.snr_s[126, 1], 'snr127R':self.snr_s[127, 0], 'snr127L':self.snr_s[127, 1], 
                        'snr128R':self.snr_s[128, 0], 'snr128L':self.snr_s[128, 1], 'snr129R':self.snr_s[129, 0], 'snr129L':self.snr_s[129, 1], 
                        'snr130R':self.snr_s[130, 0], 'snr130L':self.snr_s[130, 1], 'snr131R':self.snr_s[131, 0], 'snr131L':self.snr_s[131, 1], 
                        'snr132R':self.snr_s[132, 0], 'snr132L':self.snr_s[132, 1], 'snr133R':self.snr_s[133, 0], 'snr133L':self.snr_s[133, 1], 
                        'snr134R':self.snr_s[134, 0], 'snr134L':self.snr_s[134, 1], 'snr135R':self.snr_s[135, 0], 'snr135L':self.snr_s[135, 1], 
                        'snr136R':self.snr_s[136, 0], 'snr136L':self.snr_s[136, 1], 'snr137R':self.snr_s[137, 0], 'snr137L':self.snr_s[137, 1], 
                        'snr138R':self.snr_s[138, 0], 'snr138L':self.snr_s[138, 1], 'snr139R':self.snr_s[139, 0], 'snr139L':self.snr_s[139, 1], 
                        'snr140R':self.snr_s[140, 0], 'snr140L':self.snr_s[140, 1], 'snr141R':self.snr_s[141, 0], 'snr141L':self.snr_s[141, 1], 
                        'snr142R':self.snr_s[142, 0], 'snr142L':self.snr_s[142, 1], 'snr143R':self.snr_s[143, 0], 'snr143L':self.snr_s[143, 1], 
                        'snr144R':self.snr_s[144, 0], 'snr144L':self.snr_s[144, 1], 'snr145R':self.snr_s[145, 0], 'snr145L':self.snr_s[145, 1], 
                        'snr146R':self.snr_s[146, 0], 'snr146L':self.snr_s[146, 1], 'snr147R':self.snr_s[147, 0], 'snr147L':self.snr_s[147, 1], 
                        'snr148R':self.snr_s[148, 0], 'snr148L':self.snr_s[148, 1], 'snr149R':self.snr_s[149, 0], 'snr149L':self.snr_s[149, 1], 
                        'snr150R':self.snr_s[150, 0], 'snr150L':self.snr_s[150, 1], 'snr151R':self.snr_s[151, 0], 'snr151L':self.snr_s[151, 1], 
                        'snr152R':self.snr_s[152, 0], 'snr152L':self.snr_s[152, 1], 'snr153R':self.snr_s[153, 0], 'snr153L':self.snr_s[153, 1], 
                        'snr154R':self.snr_s[154, 0], 'snr154L':self.snr_s[154, 1], 'snr155R':self.snr_s[155, 0], 'snr155L':self.snr_s[155, 1], 
                        'snr156R':self.snr_s[156, 0], 'snr156L':self.snr_s[156, 1], 'snr157R':self.snr_s[157, 0], 'snr157L':self.snr_s[157, 1], 
                        'snr158R':self.snr_s[158, 0], 'snr158L':self.snr_s[158, 1], 'snr159R':self.snr_s[159, 0], 'snr159L':self.snr_s[159, 1], 
                        'snr160R':self.snr_s[160, 0], 'snr160L':self.snr_s[160, 1], 'snr161R':self.snr_s[161, 0], 'snr161L':self.snr_s[161, 1], 
                        'snr162R':self.snr_s[162, 0], 'snr162L':self.snr_s[162, 1], 'snr163R':self.snr_s[163, 0], 'snr163L':self.snr_s[163, 1], 
                        'snr164R':self.snr_s[164, 0], 'snr164L':self.snr_s[164, 1], 'snr165R':self.snr_s[165, 0], 'snr165L':self.snr_s[165, 1], 
                        'snr166R':self.snr_s[166, 0], 'snr166L':self.snr_s[166, 1], 'snr167R':self.snr_s[167, 0], 'snr167L':self.snr_s[167, 1], 
                        'snr168R':self.snr_s[168, 0], 'snr168L':self.snr_s[168, 1], 'snr169R':self.snr_s[169, 0], 'snr169L':self.snr_s[169, 1], 
                        'snr170R':self.snr_s[170, 0], 'snr170L':self.snr_s[170, 1], 'snr171R':self.snr_s[171, 0], 'snr171L':self.snr_s[171, 1], 
                        'snr172R':self.snr_s[172, 0], 'snr172L':self.snr_s[172, 1], 'snr173R':self.snr_s[173, 0], 'snr173L':self.snr_s[173, 1], 
                        'snr174R':self.snr_s[174, 0], 'snr174L':self.snr_s[174, 1], 'snr175R':self.snr_s[175, 0], 'snr175L':self.snr_s[175, 1], 
                        'snr176R':self.snr_s[176, 0], 'snr176L':self.snr_s[176, 1], 'snr177R':self.snr_s[177, 0], 'snr177L':self.snr_s[177, 1], 
                        'snr178R':self.snr_s[178, 0], 'snr178L':self.snr_s[178, 1], 'snr179R':self.snr_s[179, 0], 'snr179L':self.snr_s[179, 1], 
                        'snr180R':self.snr_s[180, 0], 'snr180L':self.snr_s[180, 1], 'snr181R':self.snr_s[181, 0], 'snr181L':self.snr_s[181, 1], 
                        'snr182R':self.snr_s[182, 0], 'snr182L':self.snr_s[182, 1], 'snr183R':self.snr_s[183, 0], 'snr183L':self.snr_s[183, 1], 
                        'snr184R':self.snr_s[184, 0], 'snr184L':self.snr_s[184, 1], 'snr185R':self.snr_s[185, 0], 'snr185L':self.snr_s[185, 1], 
                        'snr186R':self.snr_s[186, 0], 'snr186L':self.snr_s[186, 1], 'snr187R':self.snr_s[187, 0], 'snr187L':self.snr_s[187, 1], 
                        'snr188R':self.snr_s[188, 0], 'snr188L':self.snr_s[188, 1], 'snr189R':self.snr_s[189, 0], 'snr189L':self.snr_s[189, 1], 
                        'snr190R':self.snr_s[190, 0], 'snr190L':self.snr_s[190, 1], 'snr191R':self.snr_s[191, 0], 'snr191L':self.snr_s[191, 1], 
                        'snr192R':self.snr_s[192, 0], 'snr192L':self.snr_s[192, 1], 'snr193R':self.snr_s[193, 0], 'snr193L':self.snr_s[193, 1], 
                        'snr194R':self.snr_s[194, 0], 'snr194L':self.snr_s[194, 1], 'snr195R':self.snr_s[195, 0], 'snr195L':self.snr_s[195, 1], 
                        'snr196R':self.snr_s[196, 0], 'snr196L':self.snr_s[196, 1], 'snr197R':self.snr_s[197, 0], 'snr197L':self.snr_s[197, 1], 
                        'snr198R':self.snr_s[198, 0], 'snr198L':self.snr_s[198, 1], 'snr199R':self.snr_s[199, 0], 'snr199L':self.snr_s[199, 1], 
                        'snr200R':self.snr_s[200, 0], 'snr200L':self.snr_s[200, 1], 'snr201R':self.snr_s[201, 0], 'snr201L':self.snr_s[201, 1], 
                        'snr202R':self.snr_s[202, 0], 'snr202L':self.snr_s[202, 1], 'snr203R':self.snr_s[203, 0], 'snr203L':self.snr_s[203, 1], 
                        'snr204R':self.snr_s[204, 0], 'snr204L':self.snr_s[204, 1], 'snr205R':self.snr_s[205, 0], 'snr205L':self.snr_s[205, 1], 
                        'snr206R':self.snr_s[206, 0], 'snr206L':self.snr_s[206, 1], 'snr207R':self.snr_s[207, 0], 'snr207L':self.snr_s[207, 1], 
                        'snr208R':self.snr_s[208, 0], 'snr208L':self.snr_s[208, 1], 'snr209R':self.snr_s[209, 0], 'snr209L':self.snr_s[209, 1], 
                        'snr210R':self.snr_s[210, 0], 'snr210L':self.snr_s[210, 1], 'snr211R':self.snr_s[211, 0], 'snr211L':self.snr_s[211, 1], 
                        'snr212R':self.snr_s[212, 0], 'snr212L':self.snr_s[212, 1], 'snr213R':self.snr_s[213, 0], 'snr213L':self.snr_s[213, 1], 
                        'snr214R':self.snr_s[214, 0], 'snr214L':self.snr_s[214, 1], 'snr215R':self.snr_s[215, 0], 'snr215L':self.snr_s[215, 1], 
                        'snr216R':self.snr_s[216, 0], 'snr216L':self.snr_s[216, 1], 'snr217R':self.snr_s[217, 0], 'snr217L':self.snr_s[217, 1], 
                        'snr218R':self.snr_s[218, 0], 'snr218L':self.snr_s[218, 1], 'snr219R':self.snr_s[219, 0], 'snr219L':self.snr_s[219, 1], 
                        'snr220R':self.snr_s[220, 0], 'snr220L':self.snr_s[220, 1], 'snr221R':self.snr_s[221, 0], 'snr221L':self.snr_s[221, 1], 
                        'snr222R':self.snr_s[222, 0], 'snr222L':self.snr_s[222, 1], 'snr223R':self.snr_s[223, 0], 'snr223L':self.snr_s[223, 1], 
                        'snr224R':self.snr_s[224, 0], 'snr224L':self.snr_s[224, 1], 'snr225R':self.snr_s[225, 0], 'snr225L':self.snr_s[225, 1], 
                        'snr226R':self.snr_s[226, 0], 'snr226L':self.snr_s[226, 1], 'snr227R':self.snr_s[227, 0], 'snr227L':self.snr_s[227, 1], 
                        'snr228R':self.snr_s[228, 0], 'snr228L':self.snr_s[228, 1], 'snr229R':self.snr_s[229, 0], 'snr229L':self.snr_s[229, 1], 
                        'snr230R':self.snr_s[230, 0], 'snr230L':self.snr_s[230, 1], 'snr231R':self.snr_s[231, 0], 'snr231L':self.snr_s[231, 1], 
                        'snr232R':self.snr_s[232, 0], 'snr232L':self.snr_s[232, 1], 'snr233R':self.snr_s[233, 0], 'snr233L':self.snr_s[233, 1], 
                        'snr234R':self.snr_s[234, 0], 'snr234L':self.snr_s[234, 1], 'snr235R':self.snr_s[235, 0], 'snr235L':self.snr_s[235, 1], 
                        'snr236R':self.snr_s[236, 0], 'snr236L':self.snr_s[236, 1], 'snr237R':self.snr_s[237, 0], 'snr237L':self.snr_s[237, 1], 
                        'snr238R':self.snr_s[238, 0], 'snr238L':self.snr_s[238, 1], 'snr239R':self.snr_s[239, 0], 'snr239L':self.snr_s[239, 1], 
                        'snr240R':self.snr_s[240, 0], 'snr240L':self.snr_s[240, 1], 'snr241R':self.snr_s[241, 0], 'snr241L':self.snr_s[241, 1], 
                        'snr242R':self.snr_s[242, 0], 'snr242L':self.snr_s[242, 1], 'snr243R':self.snr_s[243, 0], 'snr243L':self.snr_s[243, 1], 
                        'snr244R':self.snr_s[244, 0], 'snr244L':self.snr_s[244, 1], 'snr245R':self.snr_s[245, 0], 'snr245L':self.snr_s[245, 1], 
                        'snr246R':self.snr_s[246, 0], 'snr246L':self.snr_s[246, 1], 'snr247R':self.snr_s[247, 0], 'snr247L':self.snr_s[247, 1], 
                        'snr248R':self.snr_s[248, 0], 'snr248L':self.snr_s[248, 1], 'snr249R':self.snr_s[249, 0], 'snr249L':self.snr_s[249, 1], 
                        'snr250R':self.snr_s[250, 0], 'snr250L':self.snr_s[250, 1], 'snr251R':self.snr_s[251, 0], 'snr251L':self.snr_s[251, 1], 
                        'snr252R':self.snr_s[252, 0], 'snr252L':self.snr_s[252, 1], 'snr253R':self.snr_s[253, 0], 'snr253L':self.snr_s[253, 1], 
                        'snr254R':self.snr_s[254, 0], 'snr254L':self.snr_s[254, 1], 'snr255R':self.snr_s[255, 0], 'snr255L':self.snr_s[255, 1], 
                        'snr256R':self.snr_s[256, 0], 'snr256L':self.snr_s[256, 1], 'snr257R':self.snr_s[257, 0], 'snr257L':self.snr_s[257, 1], 
                        'snr258R':self.snr_s[258, 0], 'snr258L':self.snr_s[258, 1], 'snr259R':self.snr_s[259, 0], 'snr259L':self.snr_s[259, 1], 
                        'snr260R':self.snr_s[260, 0], 'snr260L':self.snr_s[260, 1], 'snr261R':self.snr_s[261, 0], 'snr261L':self.snr_s[261, 1], 
                        'snr262R':self.snr_s[262, 0], 'snr262L':self.snr_s[262, 1], 'snr263R':self.snr_s[263, 0], 'snr263L':self.snr_s[263, 1], 
                        'snr264R':self.snr_s[264, 0], 'snr264L':self.snr_s[264, 1], 'snr265R':self.snr_s[265, 0], 'snr265L':self.snr_s[265, 1], 
                        'snr266R':self.snr_s[266, 0], 'snr266L':self.snr_s[266, 1], 'snr267R':self.snr_s[267, 0], 'snr267L':self.snr_s[267, 1], 
                        'snr268R':self.snr_s[268, 0], 'snr268L':self.snr_s[268, 1], 'snr269R':self.snr_s[269, 0], 'snr269L':self.snr_s[269, 1], 
                        'snr270R':self.snr_s[270, 0], 'snr270L':self.snr_s[270, 1], 'snr271R':self.snr_s[271, 0], 'snr271L':self.snr_s[271, 1], 
                        'snr272R':self.snr_s[272, 0], 'snr272L':self.snr_s[272, 1], 'snr273R':self.snr_s[273, 0], 'snr273L':self.snr_s[273, 1], 
                        'snr274R':self.snr_s[274, 0], 'snr274L':self.snr_s[274, 1], 'snr275R':self.snr_s[275, 0], 'snr275L':self.snr_s[275, 1], 
                        'snr276R':self.snr_s[276, 0], 'snr276L':self.snr_s[276, 1], 'snr277R':self.snr_s[277, 0], 'snr277L':self.snr_s[277, 1], 
                        'snr278R':self.snr_s[278, 0], 'snr278L':self.snr_s[278, 1], 'snr279R':self.snr_s[279, 0], 'snr279L':self.snr_s[279, 1], 
                        'snr280R':self.snr_s[280, 0], 'snr280L':self.snr_s[280, 1], 'snr281R':self.snr_s[281, 0], 'snr281L':self.snr_s[281, 1], 
                        'snr282R':self.snr_s[282, 0], 'snr282L':self.snr_s[282, 1], 'snr283R':self.snr_s[283, 0], 'snr283L':self.snr_s[283, 1], 
                        'snr284R':self.snr_s[284, 0], 'snr284L':self.snr_s[284, 1], 'snr285R':self.snr_s[285, 0], 'snr285L':self.snr_s[285, 1], 
                        'snr286R':self.snr_s[286, 0], 'snr286L':self.snr_s[286, 1], 'snr287R':self.snr_s[287, 0], 'snr287L':self.snr_s[287, 1], 
                        'snr288R':self.snr_s[288, 0], 'snr288L':self.snr_s[288, 1], 'snr289R':self.snr_s[289, 0], 'snr289L':self.snr_s[289, 1], 
                        'snr290R':self.snr_s[290, 0], 'snr290L':self.snr_s[290, 1], 'snr291R':self.snr_s[291, 0], 'snr291L':self.snr_s[291, 1], 
                        'snr292R':self.snr_s[292, 0], 'snr292L':self.snr_s[292, 1], 'snr293R':self.snr_s[293, 0], 'snr293L':self.snr_s[293, 1], 
                        'snr294R':self.snr_s[294, 0], 'snr294L':self.snr_s[294, 1], 'snr295R':self.snr_s[295, 0], 'snr295L':self.snr_s[295, 1], 
                        'snr296R':self.snr_s[296, 0], 'snr296L':self.snr_s[296, 1], 'snr297R':self.snr_s[297, 0], 'snr297L':self.snr_s[297, 1], 
                        'snr298R':self.snr_s[298, 0], 'snr298L':self.snr_s[298, 1], 'snr299R':self.snr_s[299, 0], 'snr299L':self.snr_s[299, 1], 
                        'snr300R':self.snr_s[300, 0], 'snr300L':self.snr_s[300, 1], 'snr301R':self.snr_s[301, 0], 'snr301L':self.snr_s[301, 1], 
                        'snr302R':self.snr_s[302, 0], 'snr302L':self.snr_s[302, 1], 'snr303R':self.snr_s[303, 0], 'snr303L':self.snr_s[303, 1], 
                        'snr304R':self.snr_s[304, 0], 'snr304L':self.snr_s[304, 1], 'snr305R':self.snr_s[305, 0], 'snr305L':self.snr_s[305, 1], 
                        'snr306R':self.snr_s[306, 0], 'snr306L':self.snr_s[306, 1], 'snr307R':self.snr_s[307, 0], 'snr307L':self.snr_s[307, 1], 
                        'snr308R':self.snr_s[308, 0], 'snr308L':self.snr_s[308, 1], 'snr309R':self.snr_s[309, 0], 'snr309L':self.snr_s[309, 1], 
                        'snr310R':self.snr_s[310, 0], 'snr310L':self.snr_s[310, 1], 'snr311R':self.snr_s[311, 0], 'snr311L':self.snr_s[311, 1], 
                        'snr312R':self.snr_s[312, 0], 'snr312L':self.snr_s[312, 1], 'snr313R':self.snr_s[313, 0], 'snr313L':self.snr_s[313, 1], 
                        'snr314R':self.snr_s[314, 0], 'snr314L':self.snr_s[314, 1], 'snr315R':self.snr_s[315, 0], 'snr315L':self.snr_s[315, 1], 
                        'snr316R':self.snr_s[316, 0], 'snr316L':self.snr_s[316, 1], 'snr317R':self.snr_s[317, 0], 'snr317L':self.snr_s[317, 1], 
                        'snr318R':self.snr_s[318, 0], 'snr318L':self.snr_s[318, 1], 'snr319R':self.snr_s[319, 0], 'snr319L':self.snr_s[319, 1], 
                        'snr320R':self.snr_s[320, 0], 'snr320L':self.snr_s[320, 1], 'snr321R':self.snr_s[321, 0], 'snr321L':self.snr_s[321, 1], 
                        'snr322R':self.snr_s[322, 0], 'snr322L':self.snr_s[322, 1], 'snr323R':self.snr_s[323, 0], 'snr323L':self.snr_s[323, 1], 
                        'snr324R':self.snr_s[324, 0], 'snr324L':self.snr_s[324, 1], 'snr325R':self.snr_s[325, 0], 'snr325L':self.snr_s[325, 1], 
                        'snr326R':self.snr_s[326, 0], 'snr326L':self.snr_s[326, 1], 'snr327R':self.snr_s[327, 0], 'snr327L':self.snr_s[327, 1], 
                        'snr328R':self.snr_s[328, 0], 'snr328L':self.snr_s[328, 1], 'snr329R':self.snr_s[329, 0], 'snr329L':self.snr_s[329, 1], 
                        'snr330R':self.snr_s[330, 0], 'snr330L':self.snr_s[330, 1], 'snr331R':self.snr_s[331, 0], 'snr331L':self.snr_s[331, 1], 
                        'snr332R':self.snr_s[332, 0], 'snr332L':self.snr_s[332, 1], 'snr333R':self.snr_s[333, 0], 'snr333L':self.snr_s[333, 1], 
                        'snr334R':self.snr_s[334, 0], 'snr334L':self.snr_s[334, 1], 'snr335R':self.snr_s[335, 0], 'snr335L':self.snr_s[335, 1], 
                        'snr336R':self.snr_s[336, 0], 'snr336L':self.snr_s[336, 1], 'snr337R':self.snr_s[337, 0], 'snr337L':self.snr_s[337, 1], 
                        'snr338R':self.snr_s[338, 0], 'snr338L':self.snr_s[338, 1], 'snr339R':self.snr_s[339, 0], 'snr339L':self.snr_s[339, 1], 
                        'snr340R':self.snr_s[340, 0], 'snr340L':self.snr_s[340, 1], 'snr341R':self.snr_s[341, 0], 'snr341L':self.snr_s[341, 1], 
                        'snr342R':self.snr_s[342, 0], 'snr342L':self.snr_s[342, 1], 'snr343R':self.snr_s[343, 0], 'snr343L':self.snr_s[343, 1], 
                        'snr344R':self.snr_s[344, 0], 'snr344L':self.snr_s[344, 1], 'snr345R':self.snr_s[345, 0], 'snr345L':self.snr_s[345, 1], 
                        'snr346R':self.snr_s[346, 0], 'snr346L':self.snr_s[346, 1], 'snr347R':self.snr_s[347, 0], 'snr347L':self.snr_s[347, 1], 
                        'snr348R':self.snr_s[348, 0], 'snr348L':self.snr_s[348, 1], 'snr349R':self.snr_s[349, 0], 'snr349L':self.snr_s[349, 1], 
                        'snr350R':self.snr_s[350, 0], 'snr350L':self.snr_s[350, 1], 'snr351R':self.snr_s[351, 0], 'snr351L':self.snr_s[351, 1], 
                        'snr352R':self.snr_s[352, 0], 'snr352L':self.snr_s[352, 1], 'snr353R':self.snr_s[353, 0], 'snr353L':self.snr_s[353, 1], 
                        'snr354R':self.snr_s[354, 0], 'snr354L':self.snr_s[354, 1], 'snr355R':self.snr_s[355, 0], 'snr355L':self.snr_s[355, 1], 
                        'snr356R':self.snr_s[356, 0], 'snr356L':self.snr_s[356, 1], 'snr357R':self.snr_s[357, 0], 'snr357L':self.snr_s[357, 1], 
                        'snr358R':self.snr_s[358, 0], 'snr358L':self.snr_s[358, 1], 'snr359R':self.snr_s[359, 0], 'snr359L':self.snr_s[359, 1], 
                        'snr360R':self.snr_s[360, 0], 'snr360L':self.snr_s[360, 1], 'snr361R':self.snr_s[361, 0], 'snr361L':self.snr_s[361, 1], 
                        'snr362R':self.snr_s[362, 0], 'snr362L':self.snr_s[362, 1], 'snr363R':self.snr_s[363, 0], 'snr363L':self.snr_s[363, 1], 
                        'snr364R':self.snr_s[364, 0], 'snr364L':self.snr_s[364, 1], 'snr365R':self.snr_s[365, 0], 'snr365L':self.snr_s[365, 1], 
                        'snr366R':self.snr_s[366, 0], 'snr366L':self.snr_s[366, 1], 'snr367R':self.snr_s[367, 0], 'snr367L':self.snr_s[367, 1], 
                        'snr368R':self.snr_s[368, 0], 'snr368L':self.snr_s[368, 1], 'snr369R':self.snr_s[369, 0], 'snr369L':self.snr_s[369, 1], 
                        'snr370R':self.snr_s[370, 0], 'snr370L':self.snr_s[370, 1], 'snr371R':self.snr_s[371, 0], 'snr371L':self.snr_s[371, 1], 
                        'snr372R':self.snr_s[372, 0], 'snr372L':self.snr_s[372, 1], 'snr373R':self.snr_s[373, 0], 'snr373L':self.snr_s[373, 1], 
                        'snr374R':self.snr_s[374, 0], 'snr374L':self.snr_s[374, 1], 'snr375R':self.snr_s[375, 0], 'snr375L':self.snr_s[375, 1], 
                        'snr376R':self.snr_s[376, 0], 'snr376L':self.snr_s[376, 1], 'snr377R':self.snr_s[377, 0], 'snr377L':self.snr_s[377, 1], 
                        'snr378R':self.snr_s[378, 0], 'snr378L':self.snr_s[378, 1], 'snr379R':self.snr_s[379, 0], 'snr379L':self.snr_s[379, 1], 
                        'snr380R':self.snr_s[380, 0], 'snr380L':self.snr_s[380, 1], 'snr381R':self.snr_s[381, 0], 'snr381L':self.snr_s[381, 1], 
                        'snr382R':self.snr_s[382, 0], 'snr382L':self.snr_s[382, 1], 'snr383R':self.snr_s[383, 0], 'snr383L':self.snr_s[383, 1], 
                        'snr384R':self.snr_s[384, 0], 'snr384L':self.snr_s[384, 1], 'snr385R':self.snr_s[385, 0], 'snr385L':self.snr_s[385, 1], 
                        'snr386R':self.snr_s[386, 0], 'snr386L':self.snr_s[386, 1], 'snr387R':self.snr_s[387, 0], 'snr387L':self.snr_s[387, 1], 
                        'snr388R':self.snr_s[388, 0], 'snr388L':self.snr_s[388, 1], 'snr389R':self.snr_s[389, 0], 'snr389L':self.snr_s[389, 1], 
                        'snr390R':self.snr_s[390, 0], 'snr390L':self.snr_s[390, 1], 'snr391R':self.snr_s[391, 0], 'snr391L':self.snr_s[391, 1], 
                        'snr392R':self.snr_s[392, 0], 'snr392L':self.snr_s[392, 1], 'snr393R':self.snr_s[393, 0], 'snr393L':self.snr_s[393, 1], 
                        'snr394R':self.snr_s[394, 0], 'snr394L':self.snr_s[394, 1], 'snr395R':self.snr_s[395, 0], 'snr395L':self.snr_s[395, 1], 
                        'snr396R':self.snr_s[396, 0], 'snr396L':self.snr_s[396, 1], 'snr397R':self.snr_s[397, 0], 'snr397L':self.snr_s[397, 1], 
                        'snr398R':self.snr_s[398, 0], 'snr398L':self.snr_s[398, 1], 'snr399R':self.snr_s[399, 0], 'snr399L':self.snr_s[399, 1], 
                        'snr400R':self.snr_s[400, 0], 'snr400L':self.snr_s[400, 1], 'snr401R':self.snr_s[401, 0], 'snr401L':self.snr_s[401, 1], 
                        'snr402R':self.snr_s[402, 0], 'snr402L':self.snr_s[402, 1], 'snr403R':self.snr_s[403, 0], 'snr403L':self.snr_s[403, 1], 
                        'snr404R':self.snr_s[404, 0], 'snr404L':self.snr_s[404, 1], 'snr405R':self.snr_s[405, 0], 'snr405L':self.snr_s[405, 1], 
                        'snr406R':self.snr_s[406, 0], 'snr406L':self.snr_s[406, 1], 'snr407R':self.snr_s[407, 0], 'snr407L':self.snr_s[407, 1], 
                        'snr408R':self.snr_s[408, 0], 'snr408L':self.snr_s[408, 1], 'snr409R':self.snr_s[409, 0], 'snr409L':self.snr_s[409, 1], 
                        'snr410R':self.snr_s[410, 0], 'snr410L':self.snr_s[410, 1], 'snr411R':self.snr_s[411, 0], 'snr411L':self.snr_s[411, 1], 
                        'snr412R':self.snr_s[412, 0], 'snr412L':self.snr_s[412, 1], 'snr413R':self.snr_s[413, 0], 'snr413L':self.snr_s[413, 1], 
                        'snr414R':self.snr_s[414, 0], 'snr414L':self.snr_s[414, 1], 'snr415R':self.snr_s[415, 0], 'snr415L':self.snr_s[415, 1], 
                        'snr416R':self.snr_s[416, 0], 'snr416L':self.snr_s[416, 1], 'snr417R':self.snr_s[417, 0], 'snr417L':self.snr_s[417, 1], 
                        'snr418R':self.snr_s[418, 0], 'snr418L':self.snr_s[418, 1], 'snr419R':self.snr_s[419, 0], 'snr419L':self.snr_s[419, 1], 
                        'snr420R':self.snr_s[420, 0], 'snr420L':self.snr_s[420, 1], 'snr421R':self.snr_s[421, 0], 'snr421L':self.snr_s[421, 1], 
                        'snr422R':self.snr_s[422, 0], 'snr422L':self.snr_s[422, 1], 'snr423R':self.snr_s[423, 0], 'snr423L':self.snr_s[423, 1], 
                        'snr424R':self.snr_s[424, 0], 'snr424L':self.snr_s[424, 1], 'snr425R':self.snr_s[425, 0], 'snr425L':self.snr_s[425, 1], 
                        'snr426R':self.snr_s[426, 0], 'snr426L':self.snr_s[426, 1], 'snr427R':self.snr_s[427, 0], 'snr427L':self.snr_s[427, 1], 
                        'snr428R':self.snr_s[428, 0], 'snr428L':self.snr_s[428, 1], 'snr429R':self.snr_s[429, 0], 'snr429L':self.snr_s[429, 1], 
                        'snr430R':self.snr_s[430, 0], 'snr430L':self.snr_s[430, 1], 'snr431R':self.snr_s[431, 0], 'snr431L':self.snr_s[431, 1], 
                        'snr432R':self.snr_s[432, 0], 'snr432L':self.snr_s[432, 1], 'snr433R':self.snr_s[433, 0], 'snr433L':self.snr_s[433, 1], 
                        'snr434R':self.snr_s[434, 0], 'snr434L':self.snr_s[434, 1], 'snr435R':self.snr_s[435, 0], 'snr435L':self.snr_s[435, 1], 
                        'snr436R':self.snr_s[436, 0], 'snr436L':self.snr_s[436, 1], 'snr437R':self.snr_s[437, 0], 'snr437L':self.snr_s[437, 1], 
                        'snr438R':self.snr_s[438, 0], 'snr438L':self.snr_s[438, 1], 'snr439R':self.snr_s[439, 0], 'snr439L':self.snr_s[439, 1], 
                        'snr440R':self.snr_s[440, 0], 'snr440L':self.snr_s[440, 1], 'snr441R':self.snr_s[441, 0], 'snr441L':self.snr_s[441, 1], 
                        'snr442R':self.snr_s[442, 0], 'snr442L':self.snr_s[442, 1], 'snr443R':self.snr_s[443, 0], 'snr443L':self.snr_s[443, 1], 
                        'snr444R':self.snr_s[444, 0], 'snr444L':self.snr_s[444, 1], 'snr445R':self.snr_s[445, 0], 'snr445L':self.snr_s[445, 1], 
                        'snr446R':self.snr_s[446, 0], 'snr446L':self.snr_s[446, 1], 'snr447R':self.snr_s[447, 0], 'snr447L':self.snr_s[447, 1], 
                        'snr448R':self.snr_s[448, 0], 'snr448L':self.snr_s[448, 1], 'snr449R':self.snr_s[449, 0], 'snr449L':self.snr_s[449, 1], 
                        'snr450R':self.snr_s[450, 0], 'snr450L':self.snr_s[450, 1], 'snr451R':self.snr_s[451, 0], 'snr451L':self.snr_s[451, 1], 
                        'snr452R':self.snr_s[452, 0], 'snr452L':self.snr_s[452, 1], 'snr453R':self.snr_s[453, 0], 'snr453L':self.snr_s[453, 1], 
                        'snr454R':self.snr_s[454, 0], 'snr454L':self.snr_s[454, 1], 'snr455R':self.snr_s[455, 0], 'snr455L':self.snr_s[455, 1], 
                        'snr456R':self.snr_s[456, 0], 'snr456L':self.snr_s[456, 1], 'snr457R':self.snr_s[457, 0], 'snr457L':self.snr_s[457, 1], 
                        'snr458R':self.snr_s[458, 0], 'snr458L':self.snr_s[458, 1], 'snr459R':self.snr_s[459, 0], 'snr459L':self.snr_s[459, 1], 
                        'snr460R':self.snr_s[460, 0], 'snr460L':self.snr_s[460, 1], 'snr461R':self.snr_s[461, 0], 'snr461L':self.snr_s[461, 1], 
                        'snr462R':self.snr_s[462, 0], 'snr462L':self.snr_s[462, 1], 'snr463R':self.snr_s[463, 0], 'snr463L':self.snr_s[463, 1], 
                        'snr464R':self.snr_s[464, 0], 'snr464L':self.snr_s[464, 1], 'snr465R':self.snr_s[465, 0], 'snr465L':self.snr_s[465, 1], 
                        'snr466R':self.snr_s[466, 0], 'snr466L':self.snr_s[466, 1], 'snr467R':self.snr_s[467, 0], 'snr467L':self.snr_s[467, 1], 
                        'snr468R':self.snr_s[468, 0], 'snr468L':self.snr_s[468, 1], 'snr469R':self.snr_s[469, 0], 'snr469L':self.snr_s[469, 1], 
                        'snr470R':self.snr_s[470, 0], 'snr470L':self.snr_s[470, 1], 'snr471R':self.snr_s[471, 0], 'snr471L':self.snr_s[471, 1], 
                        'snr472R':self.snr_s[472, 0], 'snr472L':self.snr_s[472, 1], 'snr473R':self.snr_s[473, 0], 'snr473L':self.snr_s[473, 1], 
                        'snr474R':self.snr_s[474, 0], 'snr474L':self.snr_s[474, 1], 'snr475R':self.snr_s[475, 0], 'snr475L':self.snr_s[475, 1], 
                        'snr476R':self.snr_s[476, 0], 'snr476L':self.snr_s[476, 1], 'snr477R':self.snr_s[477, 0], 'snr477L':self.snr_s[477, 1], 
                        'snr478R':self.snr_s[478, 0], 'snr478L':self.snr_s[478, 1], 'snr479R':self.snr_s[479, 0], 'snr479L':self.snr_s[479, 1], 
                        'snr480R':self.snr_s[480, 0], 'snr480L':self.snr_s[480, 1], 'snr481R':self.snr_s[481, 0], 'snr481L':self.snr_s[481, 1], 
                        'snr482R':self.snr_s[482, 0], 'snr482L':self.snr_s[482, 1], 'snr483R':self.snr_s[483, 0], 'snr483L':self.snr_s[483, 1], 
                        'snr484R':self.snr_s[484, 0], 'snr484L':self.snr_s[484, 1], 'snr485R':self.snr_s[485, 0], 'snr485L':self.snr_s[485, 1], 
                        'snr486R':self.snr_s[486, 0], 'snr486L':self.snr_s[486, 1], 'snr487R':self.snr_s[487, 0], 'snr487L':self.snr_s[487, 1], 
                        'snr488R':self.snr_s[488, 0], 'snr488L':self.snr_s[488, 1], 'snr489R':self.snr_s[489, 0], 'snr489L':self.snr_s[489, 1], 
                        'snr490R':self.snr_s[490, 0], 'snr490L':self.snr_s[490, 1], 'snr491R':self.snr_s[491, 0], 'snr491L':self.snr_s[491, 1], 
                        'snr492R':self.snr_s[492, 0], 'snr492L':self.snr_s[492, 1], 'snr493R':self.snr_s[493, 0], 'snr493L':self.snr_s[493, 1], 
                        'snr494R':self.snr_s[494, 0], 'snr494L':self.snr_s[494, 1], 'snr495R':self.snr_s[495, 0], 'snr495L':self.snr_s[495, 1], 
                        'snr496R':self.snr_s[496, 0], 'snr496L':self.snr_s[496, 1], 'snr497R':self.snr_s[497, 0], 'snr497L':self.snr_s[497, 1], 
                        'snr498R':self.snr_s[498, 0], 'snr498L':self.snr_s[498, 1], 'snr499R':self.snr_s[499, 0], 'snr499L':self.snr_s[499, 1], 
                        'snr500R':self.snr_s[500, 0], 'snr500L':self.snr_s[500, 1], 'snr501R':self.snr_s[501, 0], 'snr501L':self.snr_s[501, 1], 
                        'snr502R':self.snr_s[502, 0], 'snr502L':self.snr_s[502, 1], 'snr503R':self.snr_s[503, 0], 'snr503L':self.snr_s[503, 1], 
                        'snr504R':self.snr_s[504, 0], 'snr504L':self.snr_s[504, 1], 'snr505R':self.snr_s[505, 0], 'snr505L':self.snr_s[505, 1], 
                        'snr506R':self.snr_s[506, 0], 'snr506L':self.snr_s[506, 1], 'snr507R':self.snr_s[507, 0], 'snr507L':self.snr_s[507, 1], 
                        'snr508R':self.snr_s[508, 0], 'snr508L':self.snr_s[508, 1], 'snr509R':self.snr_s[509, 0], 'snr509L':self.snr_s[509, 1], 
                        'snr510R':self.snr_s[510, 0], 'snr510L':self.snr_s[510, 1], 'snr511R':self.snr_s[511, 0], 'snr511L':self.snr_s[511, 1], 
                        'snr512R':self.snr_s[512, 0], 'snr512L':self.snr_s[512, 1], 'snr513R':self.snr_s[513, 0], 'snr513L':self.snr_s[513, 1], 
                        'snr514R':self.snr_s[514, 0], 'snr514L':self.snr_s[514, 1], 'snr515R':self.snr_s[515, 0], 'snr515L':self.snr_s[515, 1], 
                        'snr516R':self.snr_s[516, 0], 'snr516L':self.snr_s[516, 1], 'snr517R':self.snr_s[517, 0], 'snr517L':self.snr_s[517, 1], 
                        'snr518R':self.snr_s[518, 0], 'snr518L':self.snr_s[518, 1], 'snr519R':self.snr_s[519, 0], 'snr519L':self.snr_s[519, 1], 
                        'snr520R':self.snr_s[520, 0], 'snr520L':self.snr_s[520, 1], 'snr521R':self.snr_s[521, 0], 'snr521L':self.snr_s[521, 1], 
                        'snr522R':self.snr_s[522, 0], 'snr522L':self.snr_s[522, 1], 'snr523R':self.snr_s[523, 0], 'snr523L':self.snr_s[523, 1], 
                        'snr524R':self.snr_s[524, 0], 'snr524L':self.snr_s[524, 1], 'snr525R':self.snr_s[525, 0], 'snr525L':self.snr_s[525, 1], 
                        'snr526R':self.snr_s[526, 0], 'snr526L':self.snr_s[526, 1], 'snr527R':self.snr_s[527, 0], 'snr527L':self.snr_s[527, 1], 
                        'snr528R':self.snr_s[528, 0], 'snr528L':self.snr_s[528, 1], 'snr529R':self.snr_s[529, 0], 'snr529L':self.snr_s[529, 1], 
                        'snr530R':self.snr_s[530, 0], 'snr530L':self.snr_s[530, 1], 'snr531R':self.snr_s[531, 0], 'snr531L':self.snr_s[531, 1], 
                        'snr532R':self.snr_s[532, 0], 'snr532L':self.snr_s[532, 1], 'snr533R':self.snr_s[533, 0], 'snr533L':self.snr_s[533, 1], 
                        'snr534R':self.snr_s[534, 0], 'snr534L':self.snr_s[534, 1], 'snr535R':self.snr_s[535, 0], 'snr535L':self.snr_s[535, 1], 
                        'snr536R':self.snr_s[536, 0], 'snr536L':self.snr_s[536, 1], 'snr537R':self.snr_s[537, 0], 'snr537L':self.snr_s[537, 1], 
                        'snr538R':self.snr_s[538, 0], 'snr538L':self.snr_s[538, 1], 'snr539R':self.snr_s[539, 0], 'snr539L':self.snr_s[539, 1], 
                        'snr540R':self.snr_s[540, 0], 'snr540L':self.snr_s[540, 1], 'snr541R':self.snr_s[541, 0], 'snr541L':self.snr_s[541, 1], 
                        'snr542R':self.snr_s[542, 0], 'snr542L':self.snr_s[542, 1], 'snr543R':self.snr_s[543, 0], 'snr543L':self.snr_s[543, 1], 
                        'snr544R':self.snr_s[544, 0], 'snr544L':self.snr_s[544, 1], 'snr545R':self.snr_s[545, 0], 'snr545L':self.snr_s[545, 1], 
                        'snr546R':self.snr_s[546, 0], 'snr546L':self.snr_s[546, 1], 'snr547R':self.snr_s[547, 0], 'snr547L':self.snr_s[547, 1], 
                        'snr548R':self.snr_s[548, 0], 'snr548L':self.snr_s[548, 1], 'snr549R':self.snr_s[549, 0], 'snr549L':self.snr_s[549, 1], 
                        'snr550R':self.snr_s[550, 0], 'snr550L':self.snr_s[550, 1], 'snr551R':self.snr_s[551, 0], 'snr551L':self.snr_s[551, 1], 
                        'snr552R':self.snr_s[552, 0], 'snr552L':self.snr_s[552, 1], 'snr553R':self.snr_s[553, 0], 'snr553L':self.snr_s[553, 1], 
                        'snr554R':self.snr_s[554, 0], 'snr554L':self.snr_s[554, 1], 'snr555R':self.snr_s[555, 0], 'snr555L':self.snr_s[555, 1], 
                        'snr556R':self.snr_s[556, 0], 'snr556L':self.snr_s[556, 1], 'snr557R':self.snr_s[557, 0], 'snr557L':self.snr_s[557, 1], 
                        'snr558R':self.snr_s[558, 0], 'snr558L':self.snr_s[558, 1], 'snr559R':self.snr_s[559, 0], 'snr559L':self.snr_s[559, 1], 
                        'snr560R':self.snr_s[560, 0], 'snr560L':self.snr_s[560, 1], 'snr561R':self.snr_s[561, 0], 'snr561L':self.snr_s[561, 1], 
                        'snr562R':self.snr_s[562, 0], 'snr562L':self.snr_s[562, 1], 'snr563R':self.snr_s[563, 0], 'snr563L':self.snr_s[563, 1], 
                        'snr564R':self.snr_s[564, 0], 'snr564L':self.snr_s[564, 1], 'snr565R':self.snr_s[565, 0], 'snr565L':self.snr_s[565, 1], 
                        'snr566R':self.snr_s[566, 0], 'snr566L':self.snr_s[566, 1], 'snr567R':self.snr_s[567, 0], 'snr567L':self.snr_s[567, 1], 
                        'snr568R':self.snr_s[568, 0], 'snr568L':self.snr_s[568, 1], 'snr569R':self.snr_s[569, 0], 'snr569L':self.snr_s[569, 1], 
                        'snr570R':self.snr_s[570, 0], 'snr570L':self.snr_s[570, 1], 'snr571R':self.snr_s[571, 0], 'snr571L':self.snr_s[571, 1], 
                        'snr572R':self.snr_s[572, 0], 'snr572L':self.snr_s[572, 1], 'snr573R':self.snr_s[573, 0], 'snr573L':self.snr_s[573, 1], 
                        'snr574R':self.snr_s[574, 0], 'snr574L':self.snr_s[574, 1], 'snr575R':self.snr_s[575, 0], 'snr575L':self.snr_s[575, 1], 
                        'snr576R':self.snr_s[576, 0], 'snr576L':self.snr_s[576, 1], 'snr577R':self.snr_s[577, 0], 'snr577L':self.snr_s[577, 1], 
                        'snr578R':self.snr_s[578, 0], 'snr578L':self.snr_s[578, 1], 'snr579R':self.snr_s[579, 0], 'snr579L':self.snr_s[579, 1], 
                        'snr580R':self.snr_s[580, 0], 'snr580L':self.snr_s[580, 1], 'snr581R':self.snr_s[581, 0], 'snr581L':self.snr_s[581, 1], 
                        'snr582R':self.snr_s[582, 0], 'snr582L':self.snr_s[582, 1], 'snr583R':self.snr_s[583, 0], 'snr583L':self.snr_s[583, 1], 
                        'snr584R':self.snr_s[584, 0], 'snr584L':self.snr_s[584, 1], 'snr585R':self.snr_s[585, 0], 'snr585L':self.snr_s[585, 1], 
                        'snr586R':self.snr_s[586, 0], 'snr586L':self.snr_s[586, 1], 'snr587R':self.snr_s[587, 0], 'snr587L':self.snr_s[587, 1], 
                        'snr588R':self.snr_s[588, 0], 'snr588L':self.snr_s[588, 1], 'snr589R':self.snr_s[589, 0], 'snr589L':self.snr_s[589, 1], 
                        'snr590R':self.snr_s[590, 0], 'snr590L':self.snr_s[590, 1], 'snr591R':self.snr_s[591, 0], 'snr591L':self.snr_s[591, 1], 
                        'snr592R':self.snr_s[592, 0], 'snr592L':self.snr_s[592, 1], 'snr593R':self.snr_s[593, 0], 'snr593L':self.snr_s[593, 1], 
                        'snr594R':self.snr_s[594, 0], 'snr594L':self.snr_s[594, 1], 'snr595R':self.snr_s[595, 0], 'snr595L':self.snr_s[595, 1], 
                        'snr596R':self.snr_s[596, 0], 'snr596L':self.snr_s[596, 1], 'snr597R':self.snr_s[597, 0], 'snr597L':self.snr_s[597, 1], 
                        'snr598R':self.snr_s[598, 0], 'snr598L':self.snr_s[598, 1], 'snr599R':self.snr_s[599, 0], 'snr599L':self.snr_s[599, 1], 
                        'snr600R':self.snr_s[600, 0], 'snr600L':self.snr_s[600, 1], 'snr601R':self.snr_s[601, 0], 'snr601L':self.snr_s[601, 1], 
                        'snr602R':self.snr_s[602, 0], 'snr602L':self.snr_s[602, 1], 'snr603R':self.snr_s[603, 0], 'snr603L':self.snr_s[603, 1], 
                        'snr604R':self.snr_s[604, 0], 'snr604L':self.snr_s[604, 1], 'snr605R':self.snr_s[605, 0], 'snr605L':self.snr_s[605, 1], 
                        'snr606R':self.snr_s[606, 0], 'snr606L':self.snr_s[606, 1], 'snr607R':self.snr_s[607, 0], 'snr607L':self.snr_s[607, 1], 
                        'snr608R':self.snr_s[608, 0], 'snr608L':self.snr_s[608, 1], 'snr609R':self.snr_s[609, 0], 'snr609L':self.snr_s[609, 1], 
                        'snr610R':self.snr_s[610, 0], 'snr610L':self.snr_s[610, 1], 'snr611R':self.snr_s[611, 0], 'snr611L':self.snr_s[611, 1], 
                        'snr612R':self.snr_s[612, 0], 'snr612L':self.snr_s[612, 1], 'snr613R':self.snr_s[613, 0], 'snr613L':self.snr_s[613, 1], 
                        'snr614R':self.snr_s[614, 0], 'snr614L':self.snr_s[614, 1], 'snr615R':self.snr_s[615, 0], 'snr615L':self.snr_s[615, 1], 
                        'snr616R':self.snr_s[616, 0], 'snr616L':self.snr_s[616, 1], 'snr617R':self.snr_s[617, 0], 'snr617L':self.snr_s[617, 1], 
                        'snr618R':self.snr_s[618, 0], 'snr618L':self.snr_s[618, 1], 'snr619R':self.snr_s[619, 0], 'snr619L':self.snr_s[619, 1], 
                        'snr620R':self.snr_s[620, 0], 'snr620L':self.snr_s[620, 1], 'snr621R':self.snr_s[621, 0], 'snr621L':self.snr_s[621, 1], 
                        'snr622R':self.snr_s[622, 0], 'snr622L':self.snr_s[622, 1], 'snr623R':self.snr_s[623, 0], 'snr623L':self.snr_s[623, 1], 
                        'snr624R':self.snr_s[624, 0], 'snr624L':self.snr_s[624, 1], 'snr625R':self.snr_s[625, 0], 'snr625L':self.snr_s[625, 1], 
                        'snr626R':self.snr_s[626, 0], 'snr626L':self.snr_s[626, 1], 'snr627R':self.snr_s[627, 0], 'snr627L':self.snr_s[627, 1], 
                        'snr628R':self.snr_s[628, 0], 'snr628L':self.snr_s[628, 1], 'snr629R':self.snr_s[629, 0], 'snr629L':self.snr_s[629, 1], 
                        'snr630R':self.snr_s[630, 0], 'snr630L':self.snr_s[630, 1], 'snr631R':self.snr_s[631, 0], 'snr631L':self.snr_s[631, 1], 
                        'snr632R':self.snr_s[632, 0], 'snr632L':self.snr_s[632, 1], 'snr633R':self.snr_s[633, 0], 'snr633L':self.snr_s[633, 1], 
                        'snr634R':self.snr_s[634, 0], 'snr634L':self.snr_s[634, 1], 'snr635R':self.snr_s[635, 0], 'snr635L':self.snr_s[635, 1], 
                        'snr636R':self.snr_s[636, 0], 'snr636L':self.snr_s[636, 1], 'snr637R':self.snr_s[637, 0], 'snr637L':self.snr_s[637, 1], 
                        'snr638R':self.snr_s[638, 0], 'snr638L':self.snr_s[638, 1], 'snr639R':self.snr_s[639, 0], 'snr639L':self.snr_s[639, 1], 
                        'snr640R':self.snr_s[640, 0], 'snr640L':self.snr_s[640, 1], 'snr641R':self.snr_s[641, 0], 'snr641L':self.snr_s[641, 1], 
                        'snr642R':self.snr_s[642, 0], 'snr642L':self.snr_s[642, 1], 'snr643R':self.snr_s[643, 0], 'snr643L':self.snr_s[643, 1], 
                        'snr644R':self.snr_s[644, 0], 'snr644L':self.snr_s[644, 1], 'snr645R':self.snr_s[645, 0], 'snr645L':self.snr_s[645, 1], 
                        'snr646R':self.snr_s[646, 0], 'snr646L':self.snr_s[646, 1], 'snr647R':self.snr_s[647, 0], 'snr647L':self.snr_s[647, 1], 
                        'snr648R':self.snr_s[648, 0], 'snr648L':self.snr_s[648, 1], 'snr649R':self.snr_s[649, 0], 'snr649L':self.snr_s[649, 1], 
                        'snr650R':self.snr_s[650, 0], 'snr650L':self.snr_s[650, 1], 'snr651R':self.snr_s[651, 0], 'snr651L':self.snr_s[651, 1], 
                        'snr652R':self.snr_s[652, 0], 'snr652L':self.snr_s[652, 1], 'snr653R':self.snr_s[653, 0], 'snr653L':self.snr_s[653, 1], 
                        'snr654R':self.snr_s[654, 0], 'snr654L':self.snr_s[654, 1], 'snr655R':self.snr_s[655, 0], 'snr655L':self.snr_s[655, 1], 
                        'snr656R':self.snr_s[656, 0], 'snr656L':self.snr_s[656, 1], 'snr657R':self.snr_s[657, 0], 'snr657L':self.snr_s[657, 1], 
                        'snr658R':self.snr_s[658, 0], 'snr658L':self.snr_s[658, 1], 'snr659R':self.snr_s[659, 0], 'snr659L':self.snr_s[659, 1], 
                        'snr660R':self.snr_s[660, 0], 'snr660L':self.snr_s[660, 1], 'snr661R':self.snr_s[661, 0], 'snr661L':self.snr_s[661, 1], 
                        'snr662R':self.snr_s[662, 0], 'snr662L':self.snr_s[662, 1], 'snr663R':self.snr_s[663, 0], 'snr663L':self.snr_s[663, 1], 
                        'snr664R':self.snr_s[664, 0], 'snr664L':self.snr_s[664, 1], 'snr665R':self.snr_s[665, 0], 'snr665L':self.snr_s[665, 1], 
                        'snr666R':self.snr_s[666, 0], 'snr666L':self.snr_s[666, 1], 'snr667R':self.snr_s[667, 0], 'snr667L':self.snr_s[667, 1], 
                        'snr668R':self.snr_s[668, 0], 'snr668L':self.snr_s[668, 1], 'snr669R':self.snr_s[669, 0], 'snr669L':self.snr_s[669, 1], 
                        'snr670R':self.snr_s[670, 0], 'snr670L':self.snr_s[670, 1], 'snr671R':self.snr_s[671, 0], 'snr671L':self.snr_s[671, 1], 
                        'snr672R':self.snr_s[672, 0], 'snr672L':self.snr_s[672, 1], 'snr673R':self.snr_s[673, 0], 'snr673L':self.snr_s[673, 1], 
                        'snr674R':self.snr_s[674, 0], 'snr674L':self.snr_s[674, 1], 'snr675R':self.snr_s[675, 0], 'snr675L':self.snr_s[675, 1], 
                        'snr676R':self.snr_s[676, 0], 'snr676L':self.snr_s[676, 1], 'snr677R':self.snr_s[677, 0], 'snr677L':self.snr_s[677, 1], 
                        'snr678R':self.snr_s[678, 0], 'snr678L':self.snr_s[678, 1], 'snr679R':self.snr_s[679, 0], 'snr679L':self.snr_s[679, 1], 
                        'snr680R':self.snr_s[680, 0], 'snr680L':self.snr_s[680, 1], 'snr681R':self.snr_s[681, 0], 'snr681L':self.snr_s[681, 1], 
                        'snr682R':self.snr_s[682, 0], 'snr682L':self.snr_s[682, 1], 'snr683R':self.snr_s[683, 0], 'snr683L':self.snr_s[683, 1], 
                        'snr684R':self.snr_s[684, 0], 'snr684L':self.snr_s[684, 1], 'snr685R':self.snr_s[685, 0], 'snr685L':self.snr_s[685, 1], 
                        'snr686R':self.snr_s[686, 0], 'snr686L':self.snr_s[686, 1], 'snr687R':self.snr_s[687, 0], 'snr687L':self.snr_s[687, 1], 
                        'snr688R':self.snr_s[688, 0], 'snr688L':self.snr_s[688, 1], 'snr689R':self.snr_s[689, 0], 'snr689L':self.snr_s[689, 1], 
                        'snr690R':self.snr_s[690, 0], 'snr690L':self.snr_s[690, 1], 'snr691R':self.snr_s[691, 0], 'snr691L':self.snr_s[691, 1], 
                        'snr692R':self.snr_s[692, 0], 'snr692L':self.snr_s[692, 1], 'snr693R':self.snr_s[693, 0], 'snr693L':self.snr_s[693, 1], 
                        'snr694R':self.snr_s[694, 0], 'snr694L':self.snr_s[694, 1], 'snr695R':self.snr_s[695, 0], 'snr695L':self.snr_s[695, 1], 
                        'snr696R':self.snr_s[696, 0], 'snr696L':self.snr_s[696, 1], 'snr697R':self.snr_s[697, 0], 'snr697L':self.snr_s[697, 1], 
                        'snr698R':self.snr_s[698, 0], 'snr698L':self.snr_s[698, 1], 'snr699R':self.snr_s[699, 0], 'snr699L':self.snr_s[699, 1], 
                        'snr700R':self.snr_s[700, 0], 'snr700L':self.snr_s[700, 1], 'snr701R':self.snr_s[701, 0], 'snr701L':self.snr_s[701, 1], 
                        'snr702R':self.snr_s[702, 0], 'snr702L':self.snr_s[702, 1], 'snr703R':self.snr_s[703, 0], 'snr703L':self.snr_s[703, 1], 
                        'snr704R':self.snr_s[704, 0], 'snr704L':self.snr_s[704, 1], 'snr705R':self.snr_s[705, 0], 'snr705L':self.snr_s[705, 1], 
                        'snr706R':self.snr_s[706, 0], 'snr706L':self.snr_s[706, 1], 'snr707R':self.snr_s[707, 0], 'snr707L':self.snr_s[707, 1], 
                        'snr708R':self.snr_s[708, 0], 'snr708L':self.snr_s[708, 1], 'snr709R':self.snr_s[709, 0], 'snr709L':self.snr_s[709, 1], 
                        'snr710R':self.snr_s[710, 0], 'snr710L':self.snr_s[710, 1], 'snr711R':self.snr_s[711, 0], 'snr711L':self.snr_s[711, 1], 
                        'snr712R':self.snr_s[712, 0], 'snr712L':self.snr_s[712, 1], 'snr713R':self.snr_s[713, 0], 'snr713L':self.snr_s[713, 1], 
                        'snr714R':self.snr_s[714, 0], 'snr714L':self.snr_s[714, 1], 'snr715R':self.snr_s[715, 0], 'snr715L':self.snr_s[715, 1], 
                        'snr716R':self.snr_s[716, 0], 'snr716L':self.snr_s[716, 1], 'snr717R':self.snr_s[717, 0], 'snr717L':self.snr_s[717, 1], 
                        'snr718R':self.snr_s[718, 0], 'snr718L':self.snr_s[718, 1], 'snr719R':self.snr_s[719, 0], 'snr719L':self.snr_s[719, 1], 
                        'snr720R':self.snr_s[720, 0], 'snr720L':self.snr_s[720, 1], 'snr721R':self.snr_s[721, 0], 'snr721L':self.snr_s[721, 1], 
                        'snr722R':self.snr_s[722, 0], 'snr722L':self.snr_s[722, 1], 'snr723R':self.snr_s[723, 0], 'snr723L':self.snr_s[723, 1], 
                        'snr724R':self.snr_s[724, 0], 'snr724L':self.snr_s[724, 1], 'snr725R':self.snr_s[725, 0], 'snr725L':self.snr_s[725, 1], 
                        'snr726R':self.snr_s[726, 0], 'snr726L':self.snr_s[726, 1], 'snr727R':self.snr_s[727, 0], 'snr727L':self.snr_s[727, 1], 
                        'snr728R':self.snr_s[728, 0], 'snr728L':self.snr_s[728, 1], 'snr729R':self.snr_s[729, 0], 'snr729L':self.snr_s[729, 1], 
                        'snr730R':self.snr_s[730, 0], 'snr730L':self.snr_s[730, 1], 'snr731R':self.snr_s[731, 0], 'snr731L':self.snr_s[731, 1], 
                        'snr732R':self.snr_s[732, 0], 'snr732L':self.snr_s[732, 1], 'snr733R':self.snr_s[733, 0], 'snr733L':self.snr_s[733, 1], 
                        'snr734R':self.snr_s[734, 0], 'snr734L':self.snr_s[734, 1], 'snr735R':self.snr_s[735, 0], 'snr735L':self.snr_s[735, 1], 
                        'snr736R':self.snr_s[736, 0], 'snr736L':self.snr_s[736, 1], 'snr737R':self.snr_s[737, 0], 'snr737L':self.snr_s[737, 1], 
                        'snr738R':self.snr_s[738, 0], 'snr738L':self.snr_s[738, 1], 'snr739R':self.snr_s[739, 0], 'snr739L':self.snr_s[739, 1], 
                        'snr740R':self.snr_s[740, 0], 'snr740L':self.snr_s[740, 1], 'snr741R':self.snr_s[741, 0], 'snr741L':self.snr_s[741, 1], 
                        'snr742R':self.snr_s[742, 0], 'snr742L':self.snr_s[742, 1], 'snr743R':self.snr_s[743, 0], 'snr743L':self.snr_s[743, 1], 
                        'snr744R':self.snr_s[744, 0], 'snr744L':self.snr_s[744, 1], 'snr745R':self.snr_s[745, 0], 'snr745L':self.snr_s[745, 1], 
                        'snr746R':self.snr_s[746, 0], 'snr746L':self.snr_s[746, 1], 'snr747R':self.snr_s[747, 0], 'snr747L':self.snr_s[747, 1], 
                        'snr748R':self.snr_s[748, 0], 'snr748L':self.snr_s[748, 1], 'snr749R':self.snr_s[749, 0], 'snr749L':self.snr_s[749, 1], 
                        'snr750R':self.snr_s[750, 0], 'snr750L':self.snr_s[750, 1], 'snr751R':self.snr_s[751, 0], 'snr751L':self.snr_s[751, 1], 
                        'snr752R':self.snr_s[752, 0], 'snr752L':self.snr_s[752, 1], 'snr753R':self.snr_s[753, 0], 'snr753L':self.snr_s[753, 1], 
                        'snr754R':self.snr_s[754, 0], 'snr754L':self.snr_s[754, 1], 'snr755R':self.snr_s[755, 0], 'snr755L':self.snr_s[755, 1], 
                        'snr756R':self.snr_s[756, 0], 'snr756L':self.snr_s[756, 1], 'snr757R':self.snr_s[757, 0], 'snr757L':self.snr_s[757, 1], 
                        'snr758R':self.snr_s[758, 0], 'snr758L':self.snr_s[758, 1], 'snr759R':self.snr_s[759, 0], 'snr759L':self.snr_s[759, 1], 
                        'snr760R':self.snr_s[760, 0], 'snr760L':self.snr_s[760, 1], 'snr761R':self.snr_s[761, 0], 'snr761L':self.snr_s[761, 1], 
                        'snr762R':self.snr_s[762, 0], 'snr762L':self.snr_s[762, 1], 'snr763R':self.snr_s[763, 0], 'snr763L':self.snr_s[763, 1], 
                        'snr764R':self.snr_s[764, 0], 'snr764L':self.snr_s[764, 1], 'snr765R':self.snr_s[765, 0], 'snr765L':self.snr_s[765, 1], 
                        'snr766R':self.snr_s[766, 0], 'snr766L':self.snr_s[766, 1], 'snr767R':self.snr_s[767, 0], 'snr767L':self.snr_s[767, 1], 
                        'snr768R':self.snr_s[768, 0], 'snr768L':self.snr_s[768, 1], 'snr769R':self.snr_s[769, 0], 'snr769L':self.snr_s[769, 1], 
                        'snr770R':self.snr_s[770, 0], 'snr770L':self.snr_s[770, 1], 'snr771R':self.snr_s[771, 0], 'snr771L':self.snr_s[771, 1], 
                        'snr772R':self.snr_s[772, 0], 'snr772L':self.snr_s[772, 1], 'snr773R':self.snr_s[773, 0], 'snr773L':self.snr_s[773, 1], 
                        'snr774R':self.snr_s[774, 0], 'snr774L':self.snr_s[774, 1], 'snr775R':self.snr_s[775, 0], 'snr775L':self.snr_s[775, 1], 
                        'snr776R':self.snr_s[776, 0], 'snr776L':self.snr_s[776, 1], 'snr777R':self.snr_s[777, 0], 'snr777L':self.snr_s[777, 1], 
                        'snr778R':self.snr_s[778, 0], 'snr778L':self.snr_s[778, 1], 'snr779R':self.snr_s[779, 0], 'snr779L':self.snr_s[779, 1], 
                        'snr780R':self.snr_s[780, 0], 'snr780L':self.snr_s[780, 1], 'snr781R':self.snr_s[781, 0], 'snr781L':self.snr_s[781, 1], 
                        'snr782R':self.snr_s[782, 0], 'snr782L':self.snr_s[782, 1], 'snr783R':self.snr_s[783, 0], 'snr783L':self.snr_s[783, 1], 
                        'snr784R':self.snr_s[784, 0], 'snr784L':self.snr_s[784, 1], 'snr785R':self.snr_s[785, 0], 'snr785L':self.snr_s[785, 1], 
                        'snr786R':self.snr_s[786, 0], 'snr786L':self.snr_s[786, 1], 'snr787R':self.snr_s[787, 0], 'snr787L':self.snr_s[787, 1], 
                        'snr788R':self.snr_s[788, 0], 'snr788L':self.snr_s[788, 1], 'snr789R':self.snr_s[789, 0], 'snr789L':self.snr_s[789, 1], 
                        'snr790R':self.snr_s[790, 0], 'snr790L':self.snr_s[790, 1], 'snr791R':self.snr_s[791, 0], 'snr791L':self.snr_s[791, 1], 
                        'snr792R':self.snr_s[792, 0], 'snr792L':self.snr_s[792, 1], 'snr793R':self.snr_s[793, 0], 'snr793L':self.snr_s[793, 1], 
                        'snr794R':self.snr_s[794, 0], 'snr794L':self.snr_s[794, 1], 'snr795R':self.snr_s[795, 0], 'snr795L':self.snr_s[795, 1], 
                        'snr796R':self.snr_s[796, 0], 'snr796L':self.snr_s[796, 1], 'snr797R':self.snr_s[797, 0], 'snr797L':self.snr_s[797, 1], 
                        'snr798R':self.snr_s[798, 0], 'snr798L':self.snr_s[798, 1], 'snr799R':self.snr_s[799, 0], 'snr799L':self.snr_s[799, 1], 
                        'snr800R':self.snr_s[800, 0], 'snr800L':self.snr_s[800, 1], 'snr801R':self.snr_s[801, 0], 'snr801L':self.snr_s[801, 1], 
                        'snr802R':self.snr_s[802, 0], 'snr802L':self.snr_s[802, 1], 'snr803R':self.snr_s[803, 0], 'snr803L':self.snr_s[803, 1], 
                        'snr804R':self.snr_s[804, 0], 'snr804L':self.snr_s[804, 1], 'snr805R':self.snr_s[805, 0], 'snr805L':self.snr_s[805, 1], 
                        'snr806R':self.snr_s[806, 0], 'snr806L':self.snr_s[806, 1], 'snr807R':self.snr_s[807, 0], 'snr807L':self.snr_s[807, 1], 
                        'snr808R':self.snr_s[808, 0], 'snr808L':self.snr_s[808, 1], 'snr809R':self.snr_s[809, 0], 'snr809L':self.snr_s[809, 1], 
                        'snr810R':self.snr_s[810, 0], 'snr810L':self.snr_s[810, 1], 'snr811R':self.snr_s[811, 0], 'snr811L':self.snr_s[811, 1], 
                        'snr812R':self.snr_s[812, 0], 'snr812L':self.snr_s[812, 1], 'snr813R':self.snr_s[813, 0], 'snr813L':self.snr_s[813, 1], 
                        'snr814R':self.snr_s[814, 0], 'snr814L':self.snr_s[814, 1], 'snr815R':self.snr_s[815, 0], 'snr815L':self.snr_s[815, 1], 
                        'snr816R':self.snr_s[816, 0], 'snr816L':self.snr_s[816, 1], 'snr817R':self.snr_s[817, 0], 'snr817L':self.snr_s[817, 1], 
                        'snr818R':self.snr_s[818, 0], 'snr818L':self.snr_s[818, 1], 'snr819R':self.snr_s[819, 0], 'snr819L':self.snr_s[819, 1], 
                        'snr820R':self.snr_s[820, 0], 'snr820L':self.snr_s[820, 1], 'snr821R':self.snr_s[821, 0], 'snr821L':self.snr_s[821, 1], 
                        'snr822R':self.snr_s[822, 0], 'snr822L':self.snr_s[822, 1], 'snr823R':self.snr_s[823, 0], 'snr823L':self.snr_s[823, 1], 
                        'snr824R':self.snr_s[824, 0], 'snr824L':self.snr_s[824, 1], 'snr825R':self.snr_s[825, 0], 'snr825L':self.snr_s[825, 1], 
                        'snr826R':self.snr_s[826, 0], 'snr826L':self.snr_s[826, 1], 'snr827R':self.snr_s[827, 0], 'snr827L':self.snr_s[827, 1], 
                        'snr828R':self.snr_s[828, 0], 'snr828L':self.snr_s[828, 1], 'snr829R':self.snr_s[829, 0], 'snr829L':self.snr_s[829, 1], 
                        'snr830R':self.snr_s[830, 0], 'snr830L':self.snr_s[830, 1], 'snr831R':self.snr_s[831, 0], 'snr831L':self.snr_s[831, 1], 
                        'snr832R':self.snr_s[832, 0], 'snr832L':self.snr_s[832, 1], 'snr833R':self.snr_s[833, 0], 'snr833L':self.snr_s[833, 1], 
                        'snr834R':self.snr_s[834, 0], 'snr834L':self.snr_s[834, 1], 'snr835R':self.snr_s[835, 0], 'snr835L':self.snr_s[835, 1], 
                        'snr836R':self.snr_s[836, 0], 'snr836L':self.snr_s[836, 1], 'snr837R':self.snr_s[837, 0], 'snr837L':self.snr_s[837, 1], 
                        'snr838R':self.snr_s[838, 0], 'snr838L':self.snr_s[838, 1], 'snr839R':self.snr_s[839, 0], 'snr839L':self.snr_s[839, 1], 
                        'snr840R':self.snr_s[840, 0], 'snr840L':self.snr_s[840, 1], 'snr841R':self.snr_s[841, 0], 'snr841L':self.snr_s[841, 1], 
                        'snr842R':self.snr_s[842, 0], 'snr842L':self.snr_s[842, 1], 'snr843R':self.snr_s[843, 0], 'snr843L':self.snr_s[843, 1], 
                        'snr844R':self.snr_s[844, 0], 'snr844L':self.snr_s[844, 1], 'snr845R':self.snr_s[845, 0], 'snr845L':self.snr_s[845, 1], 
                        'snr846R':self.snr_s[846, 0], 'snr846L':self.snr_s[846, 1], 'snr847R':self.snr_s[847, 0], 'snr847L':self.snr_s[847, 1], 
                        'snr848R':self.snr_s[848, 0], 'snr848L':self.snr_s[848, 1], 'snr849R':self.snr_s[849, 0], 'snr849L':self.snr_s[849, 1], 
                        'snr850R':self.snr_s[850, 0], 'snr850L':self.snr_s[850, 1], 'snr851R':self.snr_s[851, 0], 'snr851L':self.snr_s[851, 1], 
                        'snr852R':self.snr_s[852, 0], 'snr852L':self.snr_s[852, 1], 'snr853R':self.snr_s[853, 0], 'snr853L':self.snr_s[853, 1], 
                        'snr854R':self.snr_s[854, 0], 'snr854L':self.snr_s[854, 1], 'snr855R':self.snr_s[855, 0], 'snr855L':self.snr_s[855, 1], 
                        'snr856R':self.snr_s[856, 0], 'snr856L':self.snr_s[856, 1], 'snr857R':self.snr_s[857, 0], 'snr857L':self.snr_s[857, 1], 
                        'snr858R':self.snr_s[858, 0], 'snr858L':self.snr_s[858, 1], 'snr859R':self.snr_s[859, 0], 'snr859L':self.snr_s[859, 1], 
                        'snr860R':self.snr_s[860, 0], 'snr860L':self.snr_s[860, 1], 'snr861R':self.snr_s[861, 0], 'snr861L':self.snr_s[861, 1], 
                        'snr862R':self.snr_s[862, 0], 'snr862L':self.snr_s[862, 1], 'snr863R':self.snr_s[863, 0], 'snr863L':self.snr_s[863, 1], 
                        'snr864R':self.snr_s[864, 0], 'snr864L':self.snr_s[864, 1], 'snr865R':self.snr_s[865, 0], 'snr865L':self.snr_s[865, 1], 
                        'snr866R':self.snr_s[866, 0], 'snr866L':self.snr_s[866, 1], 'snr867R':self.snr_s[867, 0], 'snr867L':self.snr_s[867, 1], 
                        'snr868R':self.snr_s[868, 0], 'snr868L':self.snr_s[868, 1], 'snr869R':self.snr_s[869, 0], 'snr869L':self.snr_s[869, 1], 
                        'snr870R':self.snr_s[870, 0], 'snr870L':self.snr_s[870, 1], 'snr871R':self.snr_s[871, 0], 'snr871L':self.snr_s[871, 1], 
                        'snr872R':self.snr_s[872, 0], 'snr872L':self.snr_s[872, 1], 'snr873R':self.snr_s[873, 0], 'snr873L':self.snr_s[873, 1], 
                        'snr874R':self.snr_s[874, 0], 'snr874L':self.snr_s[874, 1], 'snr875R':self.snr_s[875, 0], 'snr875L':self.snr_s[875, 1], 
                        'snr876R':self.snr_s[876, 0], 'snr876L':self.snr_s[876, 1], 'snr877R':self.snr_s[877, 0], 'snr877L':self.snr_s[877, 1], 
                        'snr878R':self.snr_s[878, 0], 'snr878L':self.snr_s[878, 1], 'snr879R':self.snr_s[879, 0], 'snr879L':self.snr_s[879, 1], 
                        'snr880R':self.snr_s[880, 0], 'snr880L':self.snr_s[880, 1], 'snr881R':self.snr_s[881, 0], 'snr881L':self.snr_s[881, 1], 
                        'snr882R':self.snr_s[882, 0], 'snr882L':self.snr_s[882, 1], 'snr883R':self.snr_s[883, 0], 'snr883L':self.snr_s[883, 1], 
                        'snr884R':self.snr_s[884, 0], 'snr884L':self.snr_s[884, 1], 'snr885R':self.snr_s[885, 0], 'snr885L':self.snr_s[885, 1], 
                        'snr886R':self.snr_s[886, 0], 'snr886L':self.snr_s[886, 1], 'snr887R':self.snr_s[887, 0], 'snr887L':self.snr_s[887, 1], 
                        'snr888R':self.snr_s[888, 0], 'snr888L':self.snr_s[888, 1], 'snr889R':self.snr_s[889, 0], 'snr889L':self.snr_s[889, 1], 
                        'snr890R':self.snr_s[890, 0], 'snr890L':self.snr_s[890, 1], 'snr891R':self.snr_s[891, 0], 'snr891L':self.snr_s[891, 1], 
                        'snr892R':self.snr_s[892, 0], 'snr892L':self.snr_s[892, 1], 'snr893R':self.snr_s[893, 0], 'snr893L':self.snr_s[893, 1], 
                        'snr894R':self.snr_s[894, 0], 'snr894L':self.snr_s[894, 1], 'snr895R':self.snr_s[895, 0], 'snr895L':self.snr_s[895, 1], 
                        'snr896R':self.snr_s[896, 0], 'snr896L':self.snr_s[896, 1], 'snr897R':self.snr_s[897, 0], 'snr897L':self.snr_s[897, 1], 
                        'snr898R':self.snr_s[898, 0], 'snr898L':self.snr_s[898, 1], 'snr899R':self.snr_s[899, 0], 'snr899L':self.snr_s[899, 1], 
                        'snr900R':self.snr_s[900, 0], 'snr900L':self.snr_s[900, 1], 'snr901R':self.snr_s[901, 0], 'snr901L':self.snr_s[901, 1], 
                        'snr902R':self.snr_s[902, 0], 'snr902L':self.snr_s[902, 1], 'snr903R':self.snr_s[903, 0], 'snr903L':self.snr_s[903, 1], 
                        'snr904R':self.snr_s[904, 0], 'snr904L':self.snr_s[904, 1], 'snr905R':self.snr_s[905, 0], 'snr905L':self.snr_s[905, 1], 
                        'snr906R':self.snr_s[906, 0], 'snr906L':self.snr_s[906, 1], 'snr907R':self.snr_s[907, 0], 'snr907L':self.snr_s[907, 1], 
                        'snr908R':self.snr_s[908, 0], 'snr908L':self.snr_s[908, 1], 'snr909R':self.snr_s[909, 0], 'snr909L':self.snr_s[909, 1], 
                        'snr910R':self.snr_s[910, 0], 'snr910L':self.snr_s[910, 1], 'snr911R':self.snr_s[911, 0], 'snr911L':self.snr_s[911, 1], 
                        'snr912R':self.snr_s[912, 0], 'snr912L':self.snr_s[912, 1], 'snr913R':self.snr_s[913, 0], 'snr913L':self.snr_s[913, 1], 
                        'snr914R':self.snr_s[914, 0], 'snr914L':self.snr_s[914, 1], 'snr915R':self.snr_s[915, 0], 'snr915L':self.snr_s[915, 1], 
                        'snr916R':self.snr_s[916, 0], 'snr916L':self.snr_s[916, 1], 'snr917R':self.snr_s[917, 0], 'snr917L':self.snr_s[917, 1], 
                        'snr918R':self.snr_s[918, 0], 'snr918L':self.snr_s[918, 1], 'snr919R':self.snr_s[919, 0], 'snr919L':self.snr_s[919, 1], 
                        'snr920R':self.snr_s[920, 0], 'snr920L':self.snr_s[920, 1], 'snr921R':self.snr_s[921, 0], 'snr921L':self.snr_s[921, 1], 
                        'snr922R':self.snr_s[922, 0], 'snr922L':self.snr_s[922, 1], 'snr923R':self.snr_s[923, 0], 'snr923L':self.snr_s[923, 1], 
                        'snr924R':self.snr_s[924, 0], 'snr924L':self.snr_s[924, 1], 'snr925R':self.snr_s[925, 0], 'snr925L':self.snr_s[925, 1], 
                        'snr926R':self.snr_s[926, 0], 'snr926L':self.snr_s[926, 1], 'snr927R':self.snr_s[927, 0], 'snr927L':self.snr_s[927, 1], 
                        'snr928R':self.snr_s[928, 0], 'snr928L':self.snr_s[928, 1], 'snr929R':self.snr_s[929, 0], 'snr929L':self.snr_s[929, 1], 
                        'snr930R':self.snr_s[930, 0], 'snr930L':self.snr_s[930, 1], 'snr931R':self.snr_s[931, 0], 'snr931L':self.snr_s[931, 1], 
                        'snr932R':self.snr_s[932, 0], 'snr932L':self.snr_s[932, 1], 'snr933R':self.snr_s[933, 0], 'snr933L':self.snr_s[933, 1], 
                        'snr934R':self.snr_s[934, 0], 'snr934L':self.snr_s[934, 1], 'snr935R':self.snr_s[935, 0], 'snr935L':self.snr_s[935, 1], 
                        'snr936R':self.snr_s[936, 0], 'snr936L':self.snr_s[936, 1], 'snr937R':self.snr_s[937, 0], 'snr937L':self.snr_s[937, 1], 
                        'snr938R':self.snr_s[938, 0], 'snr938L':self.snr_s[938, 1], 'snr939R':self.snr_s[939, 0], 'snr939L':self.snr_s[939, 1], 
                        'snr940R':self.snr_s[940, 0], 'snr940L':self.snr_s[940, 1], 'snr941R':self.snr_s[941, 0], 'snr941L':self.snr_s[941, 1], 
                        'snr942R':self.snr_s[942, 0], 'snr942L':self.snr_s[942, 1], 'snr943R':self.snr_s[943, 0], 'snr943L':self.snr_s[943, 1], 
                        'snr944R':self.snr_s[944, 0], 'snr944L':self.snr_s[944, 1], 'snr945R':self.snr_s[945, 0], 'snr945L':self.snr_s[945, 1], 
                        'snr946R':self.snr_s[946, 0], 'snr946L':self.snr_s[946, 1], 'snr947R':self.snr_s[947, 0], 'snr947L':self.snr_s[947, 1], 
                        'snr948R':self.snr_s[948, 0], 'snr948L':self.snr_s[948, 1], 'snr949R':self.snr_s[949, 0], 'snr949L':self.snr_s[949, 1], 
                        'snr950R':self.snr_s[950, 0], 'snr950L':self.snr_s[950, 1], 'snr951R':self.snr_s[951, 0], 'snr951L':self.snr_s[951, 1], 
                        'snr952R':self.snr_s[952, 0], 'snr952L':self.snr_s[952, 1], 'snr953R':self.snr_s[953, 0], 'snr953L':self.snr_s[953, 1], 
                        'snr954R':self.snr_s[954, 0], 'snr954L':self.snr_s[954, 1], 'snr955R':self.snr_s[955, 0], 'snr955L':self.snr_s[955, 1], 
                        'snr956R':self.snr_s[956, 0], 'snr956L':self.snr_s[956, 1], 'snr957R':self.snr_s[957, 0], 'snr957L':self.snr_s[957, 1], 
                        'snr958R':self.snr_s[958, 0], 'snr958L':self.snr_s[958, 1], 'snr959R':self.snr_s[959, 0], 'snr959L':self.snr_s[959, 1], 
                        'snr960R':self.snr_s[960, 0], 'snr960L':self.snr_s[960, 1], 'snr961R':self.snr_s[961, 0], 'snr961L':self.snr_s[961, 1], 
                        'snr962R':self.snr_s[962, 0], 'snr962L':self.snr_s[962, 1], 'snr963R':self.snr_s[963, 0], 'snr963L':self.snr_s[963, 1], 
                        'snr964R':self.snr_s[964, 0], 'snr964L':self.snr_s[964, 1], 'snr965R':self.snr_s[965, 0], 'snr965L':self.snr_s[965, 1], 
                        'snr966R':self.snr_s[966, 0], 'snr966L':self.snr_s[966, 1], 'snr967R':self.snr_s[967, 0], 'snr967L':self.snr_s[967, 1], 
                        'snr968R':self.snr_s[968, 0], 'snr968L':self.snr_s[968, 1], 'snr969R':self.snr_s[969, 0], 'snr969L':self.snr_s[969, 1], 
                        'snr970R':self.snr_s[970, 0], 'snr970L':self.snr_s[970, 1], 'snr971R':self.snr_s[971, 0], 'snr971L':self.snr_s[971, 1], 
                        'snr972R':self.snr_s[972, 0], 'snr972L':self.snr_s[972, 1], 'snr973R':self.snr_s[973, 0], 'snr973L':self.snr_s[973, 1], 
                        'snr974R':self.snr_s[974, 0], 'snr974L':self.snr_s[974, 1], 'snr975R':self.snr_s[975, 0], 'snr975L':self.snr_s[975, 1], 
                        'snr976R':self.snr_s[976, 0], 'snr976L':self.snr_s[976, 1], 'snr977R':self.snr_s[977, 0], 'snr977L':self.snr_s[977, 1], 
                        'snr978R':self.snr_s[978, 0], 'snr978L':self.snr_s[978, 1], 'snr979R':self.snr_s[979, 0], 'snr979L':self.snr_s[979, 1], 
                        'snr980R':self.snr_s[980, 0], 'snr980L':self.snr_s[980, 1], 'snr981R':self.snr_s[981, 0], 'snr981L':self.snr_s[981, 1], 
                        'snr982R':self.snr_s[982, 0], 'snr982L':self.snr_s[982, 1], 'snr983R':self.snr_s[983, 0], 'snr983L':self.snr_s[983, 1], 
                        'snr984R':self.snr_s[984, 0], 'snr984L':self.snr_s[984, 1], 'snr985R':self.snr_s[985, 0], 'snr985L':self.snr_s[985, 1], 
                        'snr986R':self.snr_s[986, 0], 'snr986L':self.snr_s[986, 1], 'snr987R':self.snr_s[987, 0], 'snr987L':self.snr_s[987, 1], 
                        'snr988R':self.snr_s[988, 0], 'snr988L':self.snr_s[988, 1], 'snr989R':self.snr_s[989, 0], 'snr989L':self.snr_s[989, 1], 
                        'snr990R':self.snr_s[990, 0], 'snr990L':self.snr_s[990, 1], 'snr991R':self.snr_s[991, 0], 'snr991L':self.snr_s[991, 1], 
                        'snr992R':self.snr_s[992, 0], 'snr992L':self.snr_s[992, 1], 'snr993R':self.snr_s[993, 0], 'snr993L':self.snr_s[993, 1], 
                        'snr994R':self.snr_s[994, 0], 'snr994L':self.snr_s[994, 1], 'snr995R':self.snr_s[995, 0], 'snr995L':self.snr_s[995, 1], 
                        'snr996R':self.snr_s[996, 0], 'snr996L':self.snr_s[996, 1], 'snr997R':self.snr_s[997, 0], 'snr997L':self.snr_s[997, 1], 
                        'snr998R':self.snr_s[998, 0], 'snr998L':self.snr_s[998, 1], 'snr999R':self.snr_s[999, 0], 'snr999L':self.snr_s[999, 1], 
                        'snr1000R':self.snr_s[1000, 0], 'snr1000L':self.snr_s[1000, 1], 'snr1001R':self.snr_s[1001, 0], 'snr1001L':self.snr_s[1001, 1], 
                        'snr1002R':self.snr_s[1002, 0], 'snr1002L':self.snr_s[1002, 1], 'snr1003R':self.snr_s[1003, 0], 'snr1003L':self.snr_s[1003, 1], 
                        'snr1004R':self.snr_s[1004, 0], 'snr1004L':self.snr_s[1004, 1], 'snr1005R':self.snr_s[1005, 0], 'snr1005L':self.snr_s[1005, 1], 
                        'snr1006R':self.snr_s[1006, 0], 'snr1006L':self.snr_s[1006, 1], 'snr1007R':self.snr_s[1007, 0], 'snr1007L':self.snr_s[1007, 1], 
                        'snr1008R':self.snr_s[1008, 0], 'snr1008L':self.snr_s[1008, 1], 'snr1009R':self.snr_s[1009, 0], 'snr1009L':self.snr_s[1009, 1], 
                        'snr1010R':self.snr_s[1010, 0], 'snr1010L':self.snr_s[1010, 1], 'snr1011R':self.snr_s[1011, 0], 'snr1011L':self.snr_s[1011, 1], 
                        'snr1012R':self.snr_s[1012, 0], 'snr1012L':self.snr_s[1012, 1], 'snr1013R':self.snr_s[1013, 0], 'snr1013L':self.snr_s[1013, 1], 
                        'snr1014R':self.snr_s[1014, 0], 'snr1014L':self.snr_s[1014, 1], 'snr1015R':self.snr_s[1015, 0], 'snr1015L':self.snr_s[1015, 1], 
                        'snr1016R':self.snr_s[1016, 0], 'snr1016L':self.snr_s[1016, 1], 'snr1017R':self.snr_s[1017, 0], 'snr1017L':self.snr_s[1017, 1], 
                        'snr1018R':self.snr_s[1018, 0], 'snr1018L':self.snr_s[1018, 1], 'snr1019R':self.snr_s[1019, 0], 'snr1019L':self.snr_s[1019, 1], 
                        'snr1020R':self.snr_s[1020, 0], 'snr1020L':self.snr_s[1020, 1], 'snr1021R':self.snr_s[1021, 0], 'snr1021L':self.snr_s[1021, 1], 
                        'snr1022R':self.snr_s[1022, 0], 'snr1022L':self.snr_s[1022, 1], 'snr1023R':self.snr_s[1023, 0], 'snr1023L':self.snr_s[1023, 1], 
                        'snr1024R':self.snr_s[1024, 0], 'snr1024L':self.snr_s[1024, 1], 'snr1025R':self.snr_s[1025, 0], 'snr1025L':self.snr_s[1025, 1], 
                        'snr1026R':self.snr_s[1026, 0], 'snr1026L':self.snr_s[1026, 1], 'snr1027R':self.snr_s[1027, 0], 'snr1027L':self.snr_s[1027, 1], 
                        'snr1028R':self.snr_s[1028, 0], 'snr1028L':self.snr_s[1028, 1], 'snr1029R':self.snr_s[1029, 0], 'snr1029L':self.snr_s[1029, 1], 
                        'snr1030R':self.snr_s[1030, 0], 'snr1030L':self.snr_s[1030, 1], 'snr1031R':self.snr_s[1031, 0], 'snr1031L':self.snr_s[1031, 1], 
                        'snr1032R':self.snr_s[1032, 0], 'snr1032L':self.snr_s[1032, 1], 'snr1033R':self.snr_s[1033, 0], 'snr1033L':self.snr_s[1033, 1], 
                        'snr1034R':self.snr_s[1034, 0], 'snr1034L':self.snr_s[1034, 1], 'snr1035R':self.snr_s[1035, 0], 'snr1035L':self.snr_s[1035, 1], 
                        'snr1036R':self.snr_s[1036, 0], 'snr1036L':self.snr_s[1036, 1], 'snr1037R':self.snr_s[1037, 0], 'snr1037L':self.snr_s[1037, 1], 
                        'snr1038R':self.snr_s[1038, 0], 'snr1038L':self.snr_s[1038, 1], 'snr1039R':self.snr_s[1039, 0], 'snr1039L':self.snr_s[1039, 1], 
                        'snr1040R':self.snr_s[1040, 0], 'snr1040L':self.snr_s[1040, 1], 'snr1041R':self.snr_s[1041, 0], 'snr1041L':self.snr_s[1041, 1], 
                        'snr1042R':self.snr_s[1042, 0], 'snr1042L':self.snr_s[1042, 1], 'snr1043R':self.snr_s[1043, 0], 'snr1043L':self.snr_s[1043, 1], 
                        'snr1044R':self.snr_s[1044, 0], 'snr1044L':self.snr_s[1044, 1], 'snr1045R':self.snr_s[1045, 0], 'snr1045L':self.snr_s[1045, 1], 
                        'snr1046R':self.snr_s[1046, 0], 'snr1046L':self.snr_s[1046, 1], 'snr1047R':self.snr_s[1047, 0], 'snr1047L':self.snr_s[1047, 1], 
                        'snr1048R':self.snr_s[1048, 0], 'snr1048L':self.snr_s[1048, 1], 'snr1049R':self.snr_s[1049, 0], 'snr1049L':self.snr_s[1049, 1], 
                        'snr1050R':self.snr_s[1050, 0], 'snr1050L':self.snr_s[1050, 1], 'snr1051R':self.snr_s[1051, 0], 'snr1051L':self.snr_s[1051, 1], 
                        'snr1052R':self.snr_s[1052, 0], 'snr1052L':self.snr_s[1052, 1], 'snr1053R':self.snr_s[1053, 0], 'snr1053L':self.snr_s[1053, 1], 
                        'snr1054R':self.snr_s[1054, 0], 'snr1054L':self.snr_s[1054, 1], 'snr1055R':self.snr_s[1055, 0], 'snr1055L':self.snr_s[1055, 1], 
                        'snr1056R':self.snr_s[1056, 0], 'snr1056L':self.snr_s[1056, 1], 'snr1057R':self.snr_s[1057, 0], 'snr1057L':self.snr_s[1057, 1], 
                        'snr1058R':self.snr_s[1058, 0], 'snr1058L':self.snr_s[1058, 1], 'snr1059R':self.snr_s[1059, 0], 'snr1059L':self.snr_s[1059, 1], 
                        'snr1060R':self.snr_s[1060, 0], 'snr1060L':self.snr_s[1060, 1], 'snr1061R':self.snr_s[1061, 0], 'snr1061L':self.snr_s[1061, 1], 
                        'snr1062R':self.snr_s[1062, 0], 'snr1062L':self.snr_s[1062, 1], 'snr1063R':self.snr_s[1063, 0], 'snr1063L':self.snr_s[1063, 1], 
                        'snr1064R':self.snr_s[1064, 0], 'snr1064L':self.snr_s[1064, 1], 'snr1065R':self.snr_s[1065, 0], 'snr1065L':self.snr_s[1065, 1], 
                        'snr1066R':self.snr_s[1066, 0], 'snr1066L':self.snr_s[1066, 1], 'snr1067R':self.snr_s[1067, 0], 'snr1067L':self.snr_s[1067, 1], 
                        'snr1068R':self.snr_s[1068, 0], 'snr1068L':self.snr_s[1068, 1], 'snr1069R':self.snr_s[1069, 0], 'snr1069L':self.snr_s[1069, 1], 
                        'snr1070R':self.snr_s[1070, 0], 'snr1070L':self.snr_s[1070, 1], 'snr1071R':self.snr_s[1071, 0], 'snr1071L':self.snr_s[1071, 1], 
                        'snr1072R':self.snr_s[1072, 0], 'snr1072L':self.snr_s[1072, 1], 'snr1073R':self.snr_s[1073, 0], 'snr1073L':self.snr_s[1073, 1], 
                        'snr1074R':self.snr_s[1074, 0], 'snr1074L':self.snr_s[1074, 1], 'snr1075R':self.snr_s[1075, 0], 'snr1075L':self.snr_s[1075, 1], 
                        'snr1076R':self.snr_s[1076, 0], 'snr1076L':self.snr_s[1076, 1], 'snr1077R':self.snr_s[1077, 0], 'snr1077L':self.snr_s[1077, 1], 
                        'snr1078R':self.snr_s[1078, 0], 'snr1078L':self.snr_s[1078, 1], 'snr1079R':self.snr_s[1079, 0], 'snr1079L':self.snr_s[1079, 1], 
                        'snr1080R':self.snr_s[1080, 0], 'snr1080L':self.snr_s[1080, 1], 'snr1081R':self.snr_s[1081, 0], 'snr1081L':self.snr_s[1081, 1], 
                        'snr1082R':self.snr_s[1082, 0], 'snr1082L':self.snr_s[1082, 1], 'snr1083R':self.snr_s[1083, 0], 'snr1083L':self.snr_s[1083, 1], 
                        'snr1084R':self.snr_s[1084, 0], 'snr1084L':self.snr_s[1084, 1], 'snr1085R':self.snr_s[1085, 0], 'snr1085L':self.snr_s[1085, 1], 
                        'snr1086R':self.snr_s[1086, 0], 'snr1086L':self.snr_s[1086, 1], 'snr1087R':self.snr_s[1087, 0], 'snr1087L':self.snr_s[1087, 1], 
                        'snr1088R':self.snr_s[1088, 0], 'snr1088L':self.snr_s[1088, 1], 'snr1089R':self.snr_s[1089, 0], 'snr1089L':self.snr_s[1089, 1], 
                        'snr1090R':self.snr_s[1090, 0], 'snr1090L':self.snr_s[1090, 1], 'snr1091R':self.snr_s[1091, 0], 'snr1091L':self.snr_s[1091, 1], 
                        'snr1092R':self.snr_s[1092, 0], 'snr1092L':self.snr_s[1092, 1], 'snr1093R':self.snr_s[1093, 0], 'snr1093L':self.snr_s[1093, 1], 
                        'snr1094R':self.snr_s[1094, 0], 'snr1094L':self.snr_s[1094, 1], 'snr1095R':self.snr_s[1095, 0], 'snr1095L':self.snr_s[1095, 1], 
                        'snr1096R':self.snr_s[1096, 0], 'snr1096L':self.snr_s[1096, 1], 'snr1097R':self.snr_s[1097, 0], 'snr1097L':self.snr_s[1097, 1], 
                        'snr1098R':self.snr_s[1098, 0], 'snr1098L':self.snr_s[1098, 1], 'snr1099R':self.snr_s[1099, 0], 'snr1099L':self.snr_s[1099, 1], 
                        'snr1100R':self.snr_s[1100, 0], 'snr1100L':self.snr_s[1100, 1], 'snr1101R':self.snr_s[1101, 0], 'snr1101L':self.snr_s[1101, 1], 
                        'snr1102R':self.snr_s[1102, 0], 'snr1102L':self.snr_s[1102, 1], 'snr1103R':self.snr_s[1103, 0], 'snr1103L':self.snr_s[1103, 1], 
                        'snr1104R':self.snr_s[1104, 0], 'snr1104L':self.snr_s[1104, 1], 'snr1105R':self.snr_s[1105, 0], 'snr1105L':self.snr_s[1105, 1], 
                        'snr1106R':self.snr_s[1106, 0], 'snr1106L':self.snr_s[1106, 1], 'snr1107R':self.snr_s[1107, 0], 'snr1107L':self.snr_s[1107, 1], 
                        'snr1108R':self.snr_s[1108, 0], 'snr1108L':self.snr_s[1108, 1], 'snr1109R':self.snr_s[1109, 0], 'snr1109L':self.snr_s[1109, 1], 
                        'snr1110R':self.snr_s[1110, 0], 'snr1110L':self.snr_s[1110, 1], 'snr1111R':self.snr_s[1111, 0], 'snr1111L':self.snr_s[1111, 1], 
                        'snr1112R':self.snr_s[1112, 0], 'snr1112L':self.snr_s[1112, 1], 'snr1113R':self.snr_s[1113, 0], 'snr1113L':self.snr_s[1113, 1], 
                        'snr1114R':self.snr_s[1114, 0], 'snr1114L':self.snr_s[1114, 1], 'snr1115R':self.snr_s[1115, 0], 'snr1115L':self.snr_s[1115, 1], 
                        'snr1116R':self.snr_s[1116, 0], 'snr1116L':self.snr_s[1116, 1], 'snr1117R':self.snr_s[1117, 0], 'snr1117L':self.snr_s[1117, 1], 
                        'snr1118R':self.snr_s[1118, 0], 'snr1118L':self.snr_s[1118, 1], 'snr1119R':self.snr_s[1119, 0], 'snr1119L':self.snr_s[1119, 1], 
                        'snr1120R':self.snr_s[1120, 0], 'snr1120L':self.snr_s[1120, 1], 'snr1121R':self.snr_s[1121, 0], 'snr1121L':self.snr_s[1121, 1], 
                        'snr1122R':self.snr_s[1122, 0], 'snr1122L':self.snr_s[1122, 1], 'snr1123R':self.snr_s[1123, 0], 'snr1123L':self.snr_s[1123, 1], 
                        'snr1124R':self.snr_s[1124, 0], 'snr1124L':self.snr_s[1124, 1], 'snr1125R':self.snr_s[1125, 0], 'snr1125L':self.snr_s[1125, 1], 
                        'snr1126R':self.snr_s[1126, 0], 'snr1126L':self.snr_s[1126, 1], 'snr1127R':self.snr_s[1127, 0], 'snr1127L':self.snr_s[1127, 1], 
                        'snr1128R':self.snr_s[1128, 0], 'snr1128L':self.snr_s[1128, 1], 'snr1129R':self.snr_s[1129, 0], 'snr1129L':self.snr_s[1129, 1], 
                        'snr1130R':self.snr_s[1130, 0], 'snr1130L':self.snr_s[1130, 1], 'snr1131R':self.snr_s[1131, 0], 'snr1131L':self.snr_s[1131, 1], 
                        'snr1132R':self.snr_s[1132, 0], 'snr1132L':self.snr_s[1132, 1], 'snr1133R':self.snr_s[1133, 0], 'snr1133L':self.snr_s[1133, 1], 
                        'snr1134R':self.snr_s[1134, 0], 'snr1134L':self.snr_s[1134, 1], 'snr1135R':self.snr_s[1135, 0], 'snr1135L':self.snr_s[1135, 1], 
                        'snr1136R':self.snr_s[1136, 0], 'snr1136L':self.snr_s[1136, 1], 'snr1137R':self.snr_s[1137, 0], 'snr1137L':self.snr_s[1137, 1], 
                        'snr1138R':self.snr_s[1138, 0], 'snr1138L':self.snr_s[1138, 1], 'snr1139R':self.snr_s[1139, 0], 'snr1139L':self.snr_s[1139, 1], 
                        'snr1140R':self.snr_s[1140, 0], 'snr1140L':self.snr_s[1140, 1], 'snr1141R':self.snr_s[1141, 0], 'snr1141L':self.snr_s[1141, 1], 
                        'snr1142R':self.snr_s[1142, 0], 'snr1142L':self.snr_s[1142, 1], 'snr1143R':self.snr_s[1143, 0], 'snr1143L':self.snr_s[1143, 1], 
                        'snr1144R':self.snr_s[1144, 0], 'snr1144L':self.snr_s[1144, 1], 'snr1145R':self.snr_s[1145, 0], 'snr1145L':self.snr_s[1145, 1], 
                        'snr1146R':self.snr_s[1146, 0], 'snr1146L':self.snr_s[1146, 1], 'snr1147R':self.snr_s[1147, 0], 'snr1147L':self.snr_s[1147, 1], 
                        'snr1148R':self.snr_s[1148, 0], 'snr1148L':self.snr_s[1148, 1], 'snr1149R':self.snr_s[1149, 0], 'snr1149L':self.snr_s[1149, 1], 
                        'snr1150R':self.snr_s[1150, 0], 'snr1150L':self.snr_s[1150, 1], 'snr1151R':self.snr_s[1151, 0], 'snr1151L':self.snr_s[1151, 1], 
                        'snr1152R':self.snr_s[1152, 0], 'snr1152L':self.snr_s[1152, 1], 'snr1153R':self.snr_s[1153, 0], 'snr1153L':self.snr_s[1153, 1], 
                        'snr1154R':self.snr_s[1154, 0], 'snr1154L':self.snr_s[1154, 1], 'snr1155R':self.snr_s[1155, 0], 'snr1155L':self.snr_s[1155, 1], 
                        'snr1156R':self.snr_s[1156, 0], 'snr1156L':self.snr_s[1156, 1], 'snr1157R':self.snr_s[1157, 0], 'snr1157L':self.snr_s[1157, 1], 
                        'snr1158R':self.snr_s[1158, 0], 'snr1158L':self.snr_s[1158, 1], 'snr1159R':self.snr_s[1159, 0], 'snr1159L':self.snr_s[1159, 1], 
                        'snr1160R':self.snr_s[1160, 0], 'snr1160L':self.snr_s[1160, 1], 'snr1161R':self.snr_s[1161, 0], 'snr1161L':self.snr_s[1161, 1], 
                        'snr1162R':self.snr_s[1162, 0], 'snr1162L':self.snr_s[1162, 1], 'snr1163R':self.snr_s[1163, 0], 'snr1163L':self.snr_s[1163, 1], 
                        'snr1164R':self.snr_s[1164, 0], 'snr1164L':self.snr_s[1164, 1], 'snr1165R':self.snr_s[1165, 0], 'snr1165L':self.snr_s[1165, 1], 
                        'snr1166R':self.snr_s[1166, 0], 'snr1166L':self.snr_s[1166, 1], 'snr1167R':self.snr_s[1167, 0], 'snr1167L':self.snr_s[1167, 1], 
                        'snr1168R':self.snr_s[1168, 0], 'snr1168L':self.snr_s[1168, 1], 'snr1169R':self.snr_s[1169, 0], 'snr1169L':self.snr_s[1169, 1], 
                        'snr1170R':self.snr_s[1170, 0], 'snr1170L':self.snr_s[1170, 1], 'snr1171R':self.snr_s[1171, 0], 'snr1171L':self.snr_s[1171, 1], 
                        'snr1172R':self.snr_s[1172, 0], 'snr1172L':self.snr_s[1172, 1], 'snr1173R':self.snr_s[1173, 0], 'snr1173L':self.snr_s[1173, 1], 
                        'snr1174R':self.snr_s[1174, 0], 'snr1174L':self.snr_s[1174, 1], 'snr1175R':self.snr_s[1175, 0], 'snr1175L':self.snr_s[1175, 1], 
                        'snr1176R':self.snr_s[1176, 0], 'snr1176L':self.snr_s[1176, 1], 'snr1177R':self.snr_s[1177, 0], 'snr1177L':self.snr_s[1177, 1], 
                        'snr1178R':self.snr_s[1178, 0], 'snr1178L':self.snr_s[1178, 1], 'snr1179R':self.snr_s[1179, 0], 'snr1179L':self.snr_s[1179, 1], 
                        'snr1180R':self.snr_s[1180, 0], 'snr1180L':self.snr_s[1180, 1], 'snr1181R':self.snr_s[1181, 0], 'snr1181L':self.snr_s[1181, 1], 
                        'snr1182R':self.snr_s[1182, 0], 'snr1182L':self.snr_s[1182, 1], 'snr1183R':self.snr_s[1183, 0], 'snr1183L':self.snr_s[1183, 1], 
                        'snr1184R':self.snr_s[1184, 0], 'snr1184L':self.snr_s[1184, 1], 'snr1185R':self.snr_s[1185, 0], 'snr1185L':self.snr_s[1185, 1], 
                        'snr1186R':self.snr_s[1186, 0], 'snr1186L':self.snr_s[1186, 1], 'snr1187R':self.snr_s[1187, 0], 'snr1187L':self.snr_s[1187, 1], 
                        'snr1188R':self.snr_s[1188, 0], 'snr1188L':self.snr_s[1188, 1], 'snr1189R':self.snr_s[1189, 0], 'snr1189L':self.snr_s[1189, 1], 
                        'snr1190R':self.snr_s[1190, 0], 'snr1190L':self.snr_s[1190, 1], 'snr1191R':self.snr_s[1191, 0], 'snr1191L':self.snr_s[1191, 1], 
                        'snr1192R':self.snr_s[1192, 0], 'snr1192L':self.snr_s[1192, 1], 'snr1193R':self.snr_s[1193, 0], 'snr1193L':self.snr_s[1193, 1], 
                        'snr1194R':self.snr_s[1194, 0], 'snr1194L':self.snr_s[1194, 1], 'snr1195R':self.snr_s[1195, 0], 'snr1195L':self.snr_s[1195, 1], 
                        'snr1196R':self.snr_s[1196, 0], 'snr1196L':self.snr_s[1196, 1], 'snr1197R':self.snr_s[1197, 0], 'snr1197L':self.snr_s[1197, 1], 
                        'snr1198R':self.snr_s[1198, 0], 'snr1198L':self.snr_s[1198, 1], 'snr1199R':self.snr_s[1199, 0], 'snr1199L':self.snr_s[1199, 1], 
                        'snr1200R':self.snr_s[1200, 0], 'snr1200L':self.snr_s[1200, 1], 'snr1201R':self.snr_s[1201, 0], 'snr1201L':self.snr_s[1201, 1], 
                        'snr1202R':self.snr_s[1202, 0], 'snr1202L':self.snr_s[1202, 1], 'snr1203R':self.snr_s[1203, 0], 'snr1203L':self.snr_s[1203, 1], 
                        'snr1204R':self.snr_s[1204, 0], 'snr1204L':self.snr_s[1204, 1], 'snr1205R':self.snr_s[1205, 0], 'snr1205L':self.snr_s[1205, 1], 
                        'snr1206R':self.snr_s[1206, 0], 'snr1206L':self.snr_s[1206, 1], 'snr1207R':self.snr_s[1207, 0], 'snr1207L':self.snr_s[1207, 1], 
                        'snr1208R':self.snr_s[1208, 0], 'snr1208L':self.snr_s[1208, 1], 'snr1209R':self.snr_s[1209, 0], 'snr1209L':self.snr_s[1209, 1], 
                        'snr1210R':self.snr_s[1210, 0], 'snr1210L':self.snr_s[1210, 1], 'snr1211R':self.snr_s[1211, 0], 'snr1211L':self.snr_s[1211, 1], 
                        'snr1212R':self.snr_s[1212, 0], 'snr1212L':self.snr_s[1212, 1], 'snr1213R':self.snr_s[1213, 0], 'snr1213L':self.snr_s[1213, 1], 
                        'snr1214R':self.snr_s[1214, 0], 'snr1214L':self.snr_s[1214, 1], 'snr1215R':self.snr_s[1215, 0], 'snr1215L':self.snr_s[1215, 1], 
                        'snr1216R':self.snr_s[1216, 0], 'snr1216L':self.snr_s[1216, 1], 'snr1217R':self.snr_s[1217, 0], 'snr1217L':self.snr_s[1217, 1], 
                        'snr1218R':self.snr_s[1218, 0], 'snr1218L':self.snr_s[1218, 1], 'snr1219R':self.snr_s[1219, 0], 'snr1219L':self.snr_s[1219, 1], 
                        'snr1220R':self.snr_s[1220, 0], 'snr1220L':self.snr_s[1220, 1], 'snr1221R':self.snr_s[1221, 0], 'snr1221L':self.snr_s[1221, 1], 
                        'snr1222R':self.snr_s[1222, 0], 'snr1222L':self.snr_s[1222, 1], 'snr1223R':self.snr_s[1223, 0], 'snr1223L':self.snr_s[1223, 1], 
                        'snr1224R':self.snr_s[1224, 0], 'snr1224L':self.snr_s[1224, 1], 'snr1225R':self.snr_s[1225, 0], 'snr1225L':self.snr_s[1225, 1], 
                        'snr1226R':self.snr_s[1226, 0], 'snr1226L':self.snr_s[1226, 1], 'snr1227R':self.snr_s[1227, 0], 'snr1227L':self.snr_s[1227, 1], 
                        'snr1228R':self.snr_s[1228, 0], 'snr1228L':self.snr_s[1228, 1], 'snr1229R':self.snr_s[1229, 0], 'snr1229L':self.snr_s[1229, 1], 
                        'snr1230R':self.snr_s[1230, 0], 'snr1230L':self.snr_s[1230, 1], 'snr1231R':self.snr_s[1231, 0], 'snr1231L':self.snr_s[1231, 1], 
                        'snr1232R':self.snr_s[1232, 0], 'snr1232L':self.snr_s[1232, 1], 'snr1233R':self.snr_s[1233, 0], 'snr1233L':self.snr_s[1233, 1], 
                        'snr1234R':self.snr_s[1234, 0], 'snr1234L':self.snr_s[1234, 1], 'snr1235R':self.snr_s[1235, 0], 'snr1235L':self.snr_s[1235, 1], 
                        'snr1236R':self.snr_s[1236, 0], 'snr1236L':self.snr_s[1236, 1], 'snr1237R':self.snr_s[1237, 0], 'snr1237L':self.snr_s[1237, 1], 
                        'snr1238R':self.snr_s[1238, 0], 'snr1238L':self.snr_s[1238, 1], 'snr1239R':self.snr_s[1239, 0], 'snr1239L':self.snr_s[1239, 1], 
                        'snr1240R':self.snr_s[1240, 0], 'snr1240L':self.snr_s[1240, 1], 'snr1241R':self.snr_s[1241, 0], 'snr1241L':self.snr_s[1241, 1], 
                        'snr1242R':self.snr_s[1242, 0], 'snr1242L':self.snr_s[1242, 1], 'snr1243R':self.snr_s[1243, 0], 'snr1243L':self.snr_s[1243, 1], 
                        'snr1244R':self.snr_s[1244, 0], 'snr1244L':self.snr_s[1244, 1], 'snr1245R':self.snr_s[1245, 0], 'snr1245L':self.snr_s[1245, 1], 
                        'snr1246R':self.snr_s[1246, 0], 'snr1246L':self.snr_s[1246, 1], 'snr1247R':self.snr_s[1247, 0], 'snr1247L':self.snr_s[1247, 1], 
                        'snr1248R':self.snr_s[1248, 0], 'snr1248L':self.snr_s[1248, 1], 'snr1249R':self.snr_s[1249, 0], 'snr1249L':self.snr_s[1249, 1], 
                        'snr1250R':self.snr_s[1250, 0], 'snr1250L':self.snr_s[1250, 1], 'snr1251R':self.snr_s[1251, 0], 'snr1251L':self.snr_s[1251, 1], 
                        'snr1252R':self.snr_s[1252, 0], 'snr1252L':self.snr_s[1252, 1], 'snr1253R':self.snr_s[1253, 0], 'snr1253L':self.snr_s[1253, 1], 
                        'snr1254R':self.snr_s[1254, 0], 'snr1254L':self.snr_s[1254, 1], 'snr1255R':self.snr_s[1255, 0], 'snr1255L':self.snr_s[1255, 1], 
                        'snr1256R':self.snr_s[1256, 0], 'snr1256L':self.snr_s[1256, 1], 'snr1257R':self.snr_s[1257, 0], 'snr1257L':self.snr_s[1257, 1], 
                        'snr1258R':self.snr_s[1258, 0], 'snr1258L':self.snr_s[1258, 1], 'snr1259R':self.snr_s[1259, 0], 'snr1259L':self.snr_s[1259, 1], 
                        'snr1260R':self.snr_s[1260, 0], 'snr1260L':self.snr_s[1260, 1], 'snr1261R':self.snr_s[1261, 0], 'snr1261L':self.snr_s[1261, 1], 
                        'snr1262R':self.snr_s[1262, 0], 'snr1262L':self.snr_s[1262, 1], 'snr1263R':self.snr_s[1263, 0], 'snr1263L':self.snr_s[1263, 1], 
                        'snr1264R':self.snr_s[1264, 0], 'snr1264L':self.snr_s[1264, 1], 'snr1265R':self.snr_s[1265, 0], 'snr1265L':self.snr_s[1265, 1], 
                        'snr1266R':self.snr_s[1266, 0], 'snr1266L':self.snr_s[1266, 1], 'snr1267R':self.snr_s[1267, 0], 'snr1267L':self.snr_s[1267, 1], 
                        'snr1268R':self.snr_s[1268, 0], 'snr1268L':self.snr_s[1268, 1], 'snr1269R':self.snr_s[1269, 0], 'snr1269L':self.snr_s[1269, 1], 
                        'snr1270R':self.snr_s[1270, 0], 'snr1270L':self.snr_s[1270, 1], 'snr1271R':self.snr_s[1271, 0], 'snr1271L':self.snr_s[1271, 1], 
                        'snr1272R':self.snr_s[1272, 0], 'snr1272L':self.snr_s[1272, 1], 'snr1273R':self.snr_s[1273, 0], 'snr1273L':self.snr_s[1273, 1], 
                        'snr1274R':self.snr_s[1274, 0], 'snr1274L':self.snr_s[1274, 1], 'snr1275R':self.snr_s[1275, 0], 'snr1275L':self.snr_s[1275, 1], 
                        'snr1276R':self.snr_s[1276, 0], 'snr1276L':self.snr_s[1276, 1], 'snr1277R':self.snr_s[1277, 0], 'snr1277L':self.snr_s[1277, 1], 
                        'snr1278R':self.snr_s[1278, 0], 'snr1278L':self.snr_s[1278, 1], 'snr1279R':self.snr_s[1279, 0], 'snr1279L':self.snr_s[1279, 1], 
                        'snr1280R':self.snr_s[1280, 0], 'snr1280L':self.snr_s[1280, 1], 'snr1281R':self.snr_s[1281, 0], 'snr1281L':self.snr_s[1281, 1], 
                        'snr1282R':self.snr_s[1282, 0], 'snr1282L':self.snr_s[1282, 1], 'snr1283R':self.snr_s[1283, 0], 'snr1283L':self.snr_s[1283, 1], 
                        'snr1284R':self.snr_s[1284, 0], 'snr1284L':self.snr_s[1284, 1], 'snr1285R':self.snr_s[1285, 0], 'snr1285L':self.snr_s[1285, 1], 
                        'snr1286R':self.snr_s[1286, 0], 'snr1286L':self.snr_s[1286, 1], 'snr1287R':self.snr_s[1287, 0], 'snr1287L':self.snr_s[1287, 1], 
                        'snr1288R':self.snr_s[1288, 0], 'snr1288L':self.snr_s[1288, 1], 'snr1289R':self.snr_s[1289, 0], 'snr1289L':self.snr_s[1289, 1], 
                        'snr1290R':self.snr_s[1290, 0], 'snr1290L':self.snr_s[1290, 1], 'snr1291R':self.snr_s[1291, 0], 'snr1291L':self.snr_s[1291, 1], 
                        'snr1292R':self.snr_s[1292, 0], 'snr1292L':self.snr_s[1292, 1], 'snr1293R':self.snr_s[1293, 0], 'snr1293L':self.snr_s[1293, 1], 
                        'snr1294R':self.snr_s[1294, 0], 'snr1294L':self.snr_s[1294, 1], 'snr1295R':self.snr_s[1295, 0], 'snr1295L':self.snr_s[1295, 1]

"""