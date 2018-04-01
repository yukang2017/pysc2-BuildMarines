from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features

from agents.network import build_net
import utils as U

#scripted agents
from pysc2.agents import base_agent

_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id # 建立兵营的id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id # 建立供应站的id
_NOOP = actions.FUNCTIONS.no_op.id # 什么都不做
_SELECT_POINT = actions.FUNCTIONS.select_point.id # 选择一个点
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id # 训练枪兵的快捷键
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id # 设置集结点

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index #


# 不同单位的代码，比如scv的代码为45
_TERRAN_BARRACKS = 21
_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLYDEPOT = 19
_TERRAN_SCV = 45

#
_PLAYER_SELF = 1
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_SCREEN = [0]
_MINIMAP = [1]
_QUEUED = [1]

class BuildMarines_scripted(base_agent.BaseAgent):
    base_top_left = True
    supply_depot_built = False
    scv_selected = False
    barracks_built = False
    barracks_selected = False
    supply_depot_building = False
    num_supply_depot = 0
    last_build_x = 0
    last_build_y = 0
    def __init__(self):    
        self.valid_actions = []
    
    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]
    
    def step(self, obs):
        #super(BuildMarines_scripted, self).step(obs)
        
        new_valid_actions = obs.observation['available_actions']
        self.valid_actions = np.sort(np.unique(np.append(self.valid_actions,new_valid_actions)))
        # 调整游戏合适的速度来观察
        #time.sleep(0.05)
        
        if self.barracks_built and obs.observation['player'][_SUPPLY_USED] < obs.observation['player'][_SUPPLY_MAX]:                
            self.supply_depot_building = False

        if not self.supply_depot_built:
        # 选择scv
            if not self.scv_selected:
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                #print('unit_type = ',unit_type.nonzero())
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                
                if len(unit_x) == 0:
                    return actions.FunctionCall(_NOOP, [])

                target = [unit_x[0], unit_y[0]]
                
                self.scv_selected = True
                self.barracks_selected = False        
                action = actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])
                #print(action)
                return action
            # 建供给站
            if _BUILD_SUPPLYDEPOT in obs.observation['available_actions'] and not self.supply_depot_building:
               unit_type = obs.observation['screen'][_UNIT_TYPE]
               unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
               
               if self.num_supply_depot == 0:               
                   self.last_build_x = int(unit_x.mean())
                   self.last_build_y = int(unit_y.mean())               
                   target = self.transformLocation(self.last_build_x, 15 , self.last_build_y, 0)                
               else:
                   target = self.transformLocation(self.last_build_x, (self.num_supply_depot%2)*(10) , self.last_build_y, int((-1)**self.num_supply_depot*(self.num_supply_depot%3)*(10)))                
                   self.last_build_x = target[0]
                   self.last_build_y = target[1]
               self.num_supply_depot += 1 
               
               self.supply_depot_built = True
               action = actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_SCREEN, target])
               self.supply_depot_building = True
               self.barracks_selected = False        
               return action
        # 建兵营
        elif not self.barracks_built:
            if _BUILD_BARRACKS in obs.observation['available_actions']:
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                    
                target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), -20)
                    
                self.barracks_built = True
                action = actions.FunctionCall(_BUILD_BARRACKS, [_SCREEN, target])
                self.barracks_selected = False        
                return action
            
        # 选择兵营
        if not self.barracks_selected:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                                    
            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]        
                self.barracks_selected = True        
                action = actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])
                return action            
        # 快速造枪兵
        if self.barracks_built and obs.observation['player'][_SUPPLY_USED] < obs.observation['player'][_SUPPLY_MAX]:                
            if _TRAIN_MARINE in obs.observation['available_actions']:
                action = actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
                return action   
        elif obs.observation['player'][_SUPPLY_USED] >= obs.observation['player'][_SUPPLY_MAX]:
            self.supply_depot_built = False
            self.scv_selected = False
            
        return actions.FunctionCall(_NOOP, [])
    
class A3CAgent(object):
  """An agent specifically for solving the mini-game maps."""
  def __init__(self, training, msize, ssize, name='A3C/A3CAgent'):#
    self.name = name
    self.training = training
    self.summary = []
    # Minimap size, screen size and info size
    assert msize == ssize
    self.msize = msize
    self.ssize = ssize
    self.isize = len(actions.FUNCTIONS)
    
    self.scriptAgent = BuildMarines_scripted()
    self.less_actions = actions.FUNCTIONS
    
  def setup(self, sess, summary_writer):
    self.sess = sess
    self.summary_writer = summary_writer


  def initialize(self):
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)


  def reset(self):
    # Epsilon schedule
    self.epsilon = [0.05, 0.2]


  def build_model(self, reuse, dev, ntype):
    with tf.variable_scope(self.name) and tf.device(dev):
      if reuse:
        tf.get_variable_scope().reuse_variables()
        assert tf.get_variable_scope()
      if(len(self.scriptAgent.valid_actions)>0):
          self.less_actions = self.scriptAgent.valid_actions
      # Set inputs of networks
      self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize], name='minimap')
      self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
      self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

      # Build networks
      #net = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, len(actions.FUNCTIONS), ntype)
      net = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, len(self.less_actions), ntype)
      self.spatial_action, self.non_spatial_action, self.value = net

      # Set targets and masks
      self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
      self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize**2], name='spatial_action_selected')
      
      #self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='valid_non_spatial_action')
      #self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='non_spatial_action_selected')
      self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(self.less_actions)], name='valid_non_spatial_action')
      self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(self.less_actions)], name='non_spatial_action_selected')
      self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

      # Compute log probability
      spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
      spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
      non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
      valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action, axis=1)
      valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
      non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
      non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
      self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
      self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

      # Compute losses, more details in https://arxiv.org/abs/1602.01783
      # Policy loss and value loss
      action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
      advantage = tf.stop_gradient(self.value_target - self.value)
      policy_loss = - tf.reduce_mean(action_log_prob * advantage)
      value_loss = - tf.reduce_mean(self.value * advantage)
      self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
      self.summary.append(tf.summary.scalar('value_loss', value_loss))

      # TODO: policy penalty
      loss = policy_loss + value_loss

      # Build the optimizer
      self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
      opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
      grads = opt.compute_gradients(loss)
      cliped_grad = []
      for grad, var in grads:
        self.summary.append(tf.summary.histogram(var.op.name, var))
        self.summary.append(tf.summary.histogram(var.op.name+'/grad', grad))
        grad = tf.clip_by_norm(grad, 10.0)
        cliped_grad.append([grad, var])
      self.train_op = opt.apply_gradients(cliped_grad)
      self.summary_op = tf.summary.merge(self.summary)

      self.saver = tf.train.Saver(max_to_keep=100)


  def step(self, obs):
    
    minimap = np.array(obs.observation['minimap'], dtype=np.float32)
    minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
    screen = np.array(obs.observation['screen'], dtype=np.float32)
    screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
    # TODO: only use available actions
    info = np.zeros([1, self.isize], dtype=np.float32)
    info[0, obs.observation['available_actions']] = 1
    #print('info = ',info)
    
    feed = {self.minimap: minimap,
            self.screen: screen,
            self.info: info}
    non_spatial_action, spatial_action = self.sess.run(
      [self.non_spatial_action, self.spatial_action],
      feed_dict=feed)
      
    # Select an action and a spatial target
    non_spatial_action = non_spatial_action.ravel()
    spatial_action = spatial_action.ravel()
    valid_actions = obs.observation['available_actions']
    
    #print('valid_actions = ',valid_actions)
    #print('self.less_actions = ',self.less_actions)
    
    #找valid_actions中各元素在self.less_actions中的脚标
    valid_actions_idx = []
    for i in range(len(valid_actions)):
        for j in range(len(self.less_actions)):
            if(self.less_actions[j]==valid_actions[i]):
                valid_actions_idx.append(j)
    #valid_actions_idx = np.sort(valid_actions_idx)
    act_id = int(self.less_actions[np.argmax(non_spatial_action[valid_actions_idx])])
    
    #print('valid_actions_idx = ',valid_actions_idx)
    #print('np.argmax(non_spatial_action[valid_actions_idx]) = ', np.argmax(non_spatial_action[valid_actions_idx]))
    
    #print('act_id = ',act_id)
    target = np.argmax(spatial_action)
    target = [int(target // self.ssize), int(target % self.ssize)]

#if False:
#      print(actions.FUNCTIONS[act_id].name, target)

    # Epsilon greedy exploration
    if self.training and np.random.rand() < self.epsilon[0]:
      act_id = np.random.choice(valid_actions)
    if self.training and np.random.rand() < self.epsilon[1]:
      dy = np.random.randint(-4, 5)
      target[0] = int(max(0, min(self.ssize-1, target[0]+dy)))
      dx = np.random.randint(-4, 5)
      target[1] = int(max(0, min(self.ssize-1, target[1]+dx)))

    # Set act_id and act_args
    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
      if arg.name in ('screen', 'minimap', 'screen2'):
        act_args.append([target[1], target[0]])
      else:
        act_args.append([0])  # TODO: Be careful
    if(not act_id in valid_actions):
        return actions.FunctionCall(_NOOP, [])

    return actions.FunctionCall(act_id, act_args)


  def update(self, rbs, disc, lr, cter):
    # Compute R, which is value of the last observation
    obs = rbs[-1][-1]
    if obs.last():
      R = 0
    else:
      minimap = np.array(obs.observation['minimap'], dtype=np.float32)
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1

      feed = {self.minimap: minimap,
              self.screen: screen,
              self.info: info}
      R = self.sess.run(self.value, feed_dict=feed)[0]

    # Compute targets and masks
    minimaps = []
    screens = []
    infos = []

    value_target = np.zeros([len(rbs)], dtype=np.float32)
    value_target[-1] = R

    valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)
    spatial_action_selected = np.zeros([len(rbs), self.ssize**2], dtype=np.float32)
    
    #valid_non_spatial_action = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)
    #non_spatial_action_selected = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)#less_actions
    valid_non_spatial_action = np.zeros([len(rbs), len(self.less_actions)], dtype=np.float32)
    non_spatial_action_selected = np.zeros([len(rbs), len(self.less_actions)], dtype=np.float32)
    
    rbs.reverse()
    for i, [obs, action, next_obs] in enumerate(rbs):
      minimap = np.array(obs.observation['minimap'], dtype=np.float32)
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1

      minimaps.append(minimap)
      screens.append(screen)
      infos.append(info)

      reward = obs.reward
      act_id = action.function
      act_args = action.arguments

      value_target[i] = reward + disc * value_target[i-1]

      valid_actions = obs.observation["available_actions"]
      #idx
      valid_actions_idx = []
      act_id_idx = -1
      for i in range(len(valid_actions)):
          for j in range(len(self.less_actions)):
              if(self.less_actions[j]==valid_actions[i]):
                  valid_actions_idx.append(j)
                  
      for j in range(len(self.less_actions)):
          if(self.less_actions[j]==act_id):
              act_id_idx = j
              
      valid_non_spatial_action[i, valid_actions_idx] = 1
      non_spatial_action_selected[i, act_id_idx] = 1

      args = actions.FUNCTIONS[act_id].args
      for arg, act_arg in zip(args, act_args):
        if arg.name in ('screen', 'minimap', 'screen2'):
          ind = act_arg[1] * self.ssize + act_arg[0]
          valid_spatial_action[i] = 1
          spatial_action_selected[i, ind] = 1

    minimaps = np.concatenate(minimaps, axis=0)
    screens = np.concatenate(screens, axis=0)
    infos = np.concatenate(infos, axis=0)

    # Train
    feed = {self.minimap: minimaps,
            self.screen: screens,
            self.info: infos,
            self.value_target: value_target,
            self.valid_spatial_action: valid_spatial_action,
            self.spatial_action_selected: spatial_action_selected,
            self.valid_non_spatial_action: valid_non_spatial_action,
            self.non_spatial_action_selected: non_spatial_action_selected,
            self.learning_rate: lr}
    _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)
    self.summary_writer.add_summary(summary, cter)


  def save_model(self, path, count):
    self.saver.save(self.sess, path+'/model.pkl', count)


  def load_model(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    return int(ckpt.model_checkpoint_path.split('-')[-1])
