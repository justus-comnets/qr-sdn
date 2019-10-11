"""
   config values for learning
"""
from enum import Enum

class QMode(Enum):
  SHORTEST_PATH = -1
  MULTI_ARMED_BANDIT_NO_WEIGHT = 1
  MULTI_ARMED_BANDIT_WITH_WEIGHT = 2
  Q_LEARNING = 3
  SARSA = 4

class ExplorationMode(Enum):
  CONSTANT_EPS = 0
  FALLING_EPS = 1
  SOFTMAX = 2
  UCB = 3

class BiasRL(Enum):
  SPF = 1
  RANDOM = 2

class ActionMode(Enum):
  ONE_FLOW = 1
  DIRECT_CHANGE = 2

class RewardMode(Enum):
  ONLY_LAT = 1
  LAT_UTILISATION = 2

class Config(object):

  ################### Learning ########################
  qMode = QMode.Q_LEARNING
  alpha = 0.8
  gamma = 0.8
  epsilon = 0.3
  temperature = 3
  # for UCB
  explorationDegree = 50.0

  # how long to wait until starting to gather new rewards
  delay_reward = 2

  # how many rewards are gathred before considering taking a new action
  measurements_for_reward = 1

  # duration to stay in one load level by iperf
  duration_iperf_per_load_level_minutes = 15

  #
  exploration_mode = ExplorationMode.SOFTMAX

  # action mode
  action_mode = ActionMode.ONE_FLOW

  # if LoadLevel Test Case
  resetQTestFlag = True

  # splitting up - each load level different log file
  splitUpLoadLevelsFlag = True

  # if merging QTables when new flow joins
  mergingQTableFlag = False

  # if initialise with shortest path first or with a random selected path
  bias = BiasRL.RANDOM

  # load level
  #load_levels = [3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8]
  load_levels = [4, 10]

  # where to save the logs
  log_path = '../logs'

  # how many rewards sould be taken until building an average for the saved reward
  savingRewardCounter = 1

  # number of iterations per measurement
  iterations = 20

  # style of reward
  reward_mode = RewardMode.ONLY_LAT

  ################### Remote Controller ########################

  # update interval latency in seconds
  interval_update_latency = 1

  # sending to leanring module interval in seconds
  interval_communication_processes = 1

  # update interval flow and port statictics
  interval_controller_switch_latency = 0.5

  ################## Mininet #########################
  # queue lenght
  queue_lenght = 30
  # size (bytes) packet iperf udp
  size_iperf_pkt_bytes = 100
  # bandwith, in Mbit/s
  bw_max_dict = {1: {2: 4.0, 3: 3.0}, 2: {4: 4.0}, 3: {4: 3.0}}