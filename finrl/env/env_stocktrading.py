import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df, 
                stock_dim,
                hmax,                
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None,
                make_plots = False, 
                print_verbosity = 10,
                day = 0, iteration=''):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_space,)) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,))
        self.data = self.df.loc[self.day,:]
        self.terminal = False     
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        # initalize state
        self.state = self._initiate_state()
        
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory=[]
        self.actions_memory_=[] ## boris
        self.date_memory=[self._get_date()]
        #self.reset()
        self._seed()
        self.actions_ = []


    def _sell_stock(self, index, action):

        price = self.state[index+1]
        quantity = self.state[index+self.stock_dim+1]   

        if (self.turbulence_threshold is None) or (self.turbulence < self.turbulence_threshold):
            sell_quantity = min(abs(action), quantity)                 
        else:
            sell_quantity = quantity   # if turbulence goes over threshold, just clear out all positions                
        
        #update balance
        # print(f'take sell action')
        # print(f'price: {price:.2f}, sell_quantity: {sell_quantity:.2f}, money: {price * sell_quantity * (1 - self.transaction_cost_pct):.2f}')        
        self.state[0] += price * sell_quantity * (1 - self.transaction_cost_pct)
        self.state[index+self.stock_dim+1] -= sell_quantity
        self.cost += price * sell_quantity * (self.transaction_cost_pct)
        self.actions_memory_[-1][index] = - sell_quantity
        self.trades += 1

    
    def _buy_stock(self, index, action):

        price = self.state[index+1]
        quantity = self.state[index+self.stock_dim+1]            
        
        if (self.turbulence_threshold is None) or (self.turbulence < self.turbulence_threshold):                        
            available_quantity = self.state[0] // price    
            buy_quantity = min(available_quantity, action)
        else:
            buy_quantity = 0   # if turbulence goes over threshold, just don't buy

        #update balance
        # print(f'take buy action')
        # print(f'price: {price:.2f}, buy_quantity: {buy_quantity:.2f}, money: {price * buy_quantity * (1 + self.transaction_cost_pct):.2f}') 
        self.state[0] -= price * buy_quantity * (1 + self.transaction_cost_pct)
        self.state[index+self.stock_dim+1] += buy_quantity
        self.cost += price * buy_quantity * self.transaction_cost_pct
        self.actions_memory_[-1][index] = buy_quantity
        self.trades += 1


    def _make_plot(self):
        plt.plot(self.asset_memory,'r')
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()


    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()            
            end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = self.state[0]+sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))- self.initial_amount 
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            if df_total_value['daily_return'].std() !=0:
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                      df_total_value['daily_return'].std()
            df_rewards = pd.DataFrame(self.rewards_memory)
            if self.episode%self.print_verbosity ==0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset:{self.asset_memory[0]:0.2f}")           
                print(f"end_total_asset:{end_total_asset:0.2f}")
                print(f"total_reward:{tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value['daily_return'].std() !=0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")
            return self.state, self.reward, self.terminal,{}

        else:

            actions = actions * self.hmax
            actions = actions.flatten().astype(int) ## boris
            self.actions_memory.append(actions)
            self.actions_memory_.append(actions * 0) ## boris
            if self.turbulence_threshold is not None:
                if self.turbulence>=self.turbulence_threshold:
                    actions=np.array([-self.hmax]*self.stock_dim)
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
#             print("begin_total_asset:{}".format(begin_total_asset))  
            # print("=========")
            # print(f'begin money: {self.state[0]:.2f}')
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
#                 print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
#                 print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])
    
            # print(f'end money: {self.state[0]:.2f}')

            self.day += 1
            self.data = self.df.loc[self.day,:]    
            if self.turbulence_threshold is not None:     
                self.turbulence = self.data['turbulence'].values[0]
            self.state =  self._update_state()
            
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            
#             print("end_total_asset:{}".format(end_total_asset))
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset            
            self.rewards_memory.append(self.reward)
            self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        #self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self._get_date()]
        #initiate state
        self.state = self._initiate_state()
        self.episode+=1
        return self.state
    
    def render(self, mode='human',close=False):
        return self.state

    def _initiate_state(self):
        if len(self.df.tic.unique())>1:
            # for multiple stock
            state = [self.initial_amount] + \
                     self.data.close.values.tolist() + \
                     [0]*self.stock_dim  + \
                     sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
        else:
            # for single stock
            state = [self.initial_amount] + \
                    [self.data.close] + \
                    [0]*self.stock_dim  + \
                    sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
        return state

    def _update_state(self):
        if len(self.df.tic.unique())>1:
            # for multiple stock
            state =  [self.state[0]] + \
                      self.data.close.values.tolist() + \
                      list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                      sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])

        else:
            # for single stock
            state =  [self.state[0]] + \
                     [self.data.close] + \
                     list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                     sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
                          
        return state

    def _get_date(self):
        if len(self.df.tic.unique())>1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique())>1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']
            
            action_list = self.actions_memory_ ## boris
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = np.array(self.actions_memory_).flatten() ## boris
            df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return self, obs ## boris
