import time
import os
import copy
import math
import itertools as it
import numpy as np
import json
import random
import sys
import csv

sys.path.append("..")
sys.path.append(".")
from config import Config
from config import ExplorationMode
from config import QMode
from config import ActionMode
from config import RewardMode
from datetime import datetime

MAX_LENGHT_DIFFSET = 2
MAX_PAST_REWARDS = 5


# modes:

###############################################################
########## Class for learning started by controller############
###############################################################

def learningModule(pipe, ):
    """
    main function that performs the learning and decision taking based on reinforcement leanring
    @param pipe: connection to remote controller
    """
    print('process id:', os.getpid())
    #### RL Parameters
    # learning rate alpha
    alpha = Config.alpha
    # discount factor gamma
    gamma = Config.gamma
    # exploration probability epsilon (0-1.0)
    epsilon = Config.epsilon
    # leanring mode: q-leanring, multiarmed bandit (constant )
    learning_mode = Config.qMode
    # defines if the exploration is eps_greedy constant, with falling eps or softmax
    exploration_mode = Config.exploration_mode

    # time steps that need to be waited until reward can be gathered
    # necessary because of the delayed reward
    delayedRewardCounter = 0
    # list of currently installed flows
    # necessary to check if new flows joined
    temp_flows = []
    # Q - Table
    Q = {}
    # how many rewards are gathred before considering taking a new action
    measurements_for_reward = Config.measurements_for_reward
    # how long to wait until starting to gather new rewards
    delay_reward = Config.delay_reward
    # running time per load level
    duration_iperf_per_load_level_minutes = Config.duration_iperf_per_load_level_minutes
    # load levels
    load_levels = Config.load_levels
    # total iterations per measurement
    iterations = Config.iterations
    # temperature (for softmax)
    temperature = Config.temperature
    # exploration degree (for UCB)
    explorationDegree = Config.explorationDegree
    # total running time
    duration_per_minutes = ((len(load_levels) * duration_iperf_per_load_level_minutes)) * iterations
    # if Q-tables should be merged
    mergingQTableFlag = Config.mergingQTableFlag

    # fill-up-arrays / dicts
    average_latency_list = []
    rewards_list = []
    reward_saving_list = []
    savingValueArray = []
    previous_state = []
    previous_action = {}
    current_state = {}

    # Iterators
    savingIterator = 0
    generalIterator = 0
    Interval_saving_q = 500

    # read Load levels
    loadLevels = Config.load_levels

    # if its a test with resetting Q values when changing load levels
    reset_q_test = Config.resetQTestFlag

    # if splitting up load level files
    splitUpLoadLevels = Config.splitUpLoadLevelsFlag

    # log folder
    logPath = Config.log_path

    # changing load level flag
    reset_load_flag = False

    startingTime = time.time()

    # load level difines how high the network capacity can be
    load_level = loadLevels[0]

    # current iterations
    iterationsLevel = 0

    # how many rewards sould be taken until building an average for the saved reward
    savingRewardCounter = 1

    # if direct state change or only one flow
    action_mode = Config.action_mode

    # reward mode
    reward_mode = Config.reward_mode

    # bw_max_dict (the maximum bandwidth)
    bw_max_dict = Config.bw_max_dict

    # check if iterations need to be saved in different file
    if iterations > 1:
        iteration_split_up_flag = True
    else:
        iteration_split_up_flag = False

    # if SARSA, the next action is given when calculating the next action
    next_action = ''

    # clean up save files
    clearing_save_file(logPath, load_level, 'reward_controller', splitUpLoadLevels, iteration_split_up_flag,
                       iterationsLevel)
    clearing_save_file(logPath, load_level, 'average_latency', splitUpLoadLevels, iteration_split_up_flag,
                       iterationsLevel)
    time_now = time.time()
    time_finish = time_now + (duration_per_minutes * 60)
    dt_object = datetime.fromtimestamp(time_finish)
    print(
        "STARTING LEARNING | Mode: {} | time: {}min | exp. finishing: {} | alpha: {} | epsilon: {} | temperature; {} | Exploration Mode: {}".format(
            learning_mode,
            duration_per_minutes, dt_object, alpha,
            epsilon, temperature, exploration_mode))
    # the load level that arrives from the controller, gathered from the pipe
    # loadLevelController = 0
    while True:
        # gathers the data from the pipe (controller <-> learning module)
        elements = pipe.recv()
        # if recieved sth
        if len(elements) > 0:
            # if received latency measurement values (init)
            if len(elements['currentCombination']) > 0:
                # actual combination of flows
                current_combination = elements['currentCombination']
                # possible paths per flow
                paths_per_flow = elements['paths_per_flow']
                # dictionary with latency values between the links
                latencydict = elements['latencyDict']
                # wether load lvel changed
                reset_load_flag = elements['resetFlag']
                # load level gathered from  the mininet file (via the controller)
                loadLevelController = elements['loadLevel']
                # flag for resetting the iterations
                reset_iteration_flag = elements['iterationFlag']
                # iterations level (int)
                iterationController = elements['iteration']
                # stopFlag
                stopFlag = elements['stopFlag']

                if stopFlag:
                    saveQ(Q, iterationsLevel)
                    print("Exited after {} steps (last load level)".format(generalIterator))
                    break

                if reset_load_flag:
                    load_level = loadLevelController
                    print("change in load level. new load level: {}".format(load_level))
                    # if it should be saved in different files
                    if splitUpLoadLevels:
                        if not reset_iteration_flag:
                            clearing_save_file(logPath, load_level, 'reward_controller', splitUpLoadLevels,
                                               iteration_split_up_flag, iterationsLevel)
                            clearing_save_file(logPath, load_level, 'average_latency', splitUpLoadLevels,
                                               iteration_split_up_flag, iterationsLevel)
                        # resetting the Q-Table (restart of learning process)
                        if reset_q_test:
                            print("xxxxxxxx RESETTING Q LoadLevel: {} Iteration: {} xxxxxxxxxxxxx".format(load_level,
                                                                                                          iterationsLevel))
                            Q, actions, state_transitions = update_Q_table({}, copied_paths_per_flow,
                                                                           mergingQTableFlag, action_mode)
                        next_action = ''
                        temp_flows = []
                        generalIterator = 0
                        savingIterator = 0
                        rewards_list.clear()
                        reward_saving_list.clear()
                        average_latency_list.clear()

                if reset_iteration_flag:
                    saveQ(Q, iterationsLevel)
                    iterationsLevel = iterationController
                    # if not reset_load_flag:
                    clearing_save_file(logPath, load_level, 'reward_controller', splitUpLoadLevels,
                                       iteration_split_up_flag, iterationsLevel)
                    clearing_save_file(logPath, load_level, 'average_latency', splitUpLoadLevels,
                                       iteration_split_up_flag, iterationsLevel)
                    # resetting the Q-Table (restart of learning process)
                    if reset_q_test:
                        print("xxxxxxxx RESETTING Q LoadLevel: {} Iteration: {} xxxxxxxxxxxxx".format(load_level,
                                                                                                      iterationsLevel))
                        Q, actions, state_transitions = update_Q_table({}, copied_paths_per_flow,
                                                                       mergingQTableFlag, action_mode)
                    generalIterator = 0
                    savingIterator = 0
                    next_action = ''
                    temp_flows = []
                    rewards_list.clear()
                    reward_saving_list.clear()
                    average_latency_list.clear()
                    print("xxxxxxxxxxx Iteration: {} xxxxxxxxxxxxxxxxxxxxxxxxxx".format(iterationsLevel))
                    continue

                # perfectStateStr = '{"10.0.0.12_10.0.0.42": [1, 3, 4], "10.0.0.11_10.0.0.41": [1, 2, 4], "10.0.0.13_10.0.0.43": [1, 3, 4]}'
                # perfectState = json.loads(perfectStateStr)
                # if the load level is not -1 or 0
                if load_level > 0:
                    # if it is the first flow
                    if len(temp_flows) < 1:
                        copied_paths_per_flow = copy.deepcopy(paths_per_flow)
                        Q, actions, state_transitions = update_Q_table(Q, copied_paths_per_flow, mergingQTableFlag,
                                                                       action_mode)
                        current_state = current_combination
                        previous_state = {}
                        temp_flows = list(current_combination.keys())
                    else:
                        # new flow added -> update best route
                        setTempFlows = set(temp_flows)
                        setChosenPaths = set(list(current_combination.keys()))
                        # if reset flag is set -> changing of load level

                        # if flows are added/deleted etc
                        if abs(len(setChosenPaths) - len(setTempFlows)) > 0:
                            copied_paths_per_flow = copy.deepcopy(paths_per_flow)
                            # pointer to combinations
                            difference_set = setChosenPaths.difference(setTempFlows)
                            # if flows were added -> change q-table
                            Q, actions, state_transitions = update_Q_table(Q, copied_paths_per_flow, mergingQTableFlag,
                                                                           action_mode, difference_set)
                            previous_state = []
                            current_state = current_combination
                            temp_flows = list(current_combination.keys())
                            rewards_list.clear()
                            reward_saving_list.clear()
                            average_latency_list.clear()

                        # calculate the rewards
                        if reward_mode.value == RewardMode.ONLY_LAT.value:
                            reward = get_reward(current_combination, latencydict)
                        elif reward_mode.value == RewardMode.LAT_UTILISATION.value:
                            reward = get_reward_utilization(current_combination, latencydict)

                        # check if waited sufficient long time
                        # if delayedRewardCounter >= delay_reward:
                        average_latency_list.append(get_average_latency(current_combination, latencydict))
                        rewards_list.append(reward)
                        reward_saving_list.append(reward)
                        print("Average lat: {} reward: {}".format(average_latency_list, rewards_list))
                        # check if epsilon should be recalculated
                        if exploration_mode.value == ExplorationMode.FALLING_EPS.value:
                            epsilon = calc_epsilon(generalIterator)

                        # if gathered sufficient reward mesaurements
                        if len(rewards_list) >= measurements_for_reward:
                            # calc qValue
                            if len(previous_state) > 0 and len(previous_action) > 0:
                                if (learning_mode.value == QMode.MULTI_ARMED_BANDIT_NO_WEIGHT.value or
                                        learning_mode.value == QMode.MULTI_ARMED_BANDIT_WITH_WEIGHT.value):
                                    Q = update_Q_bandit(previous_state, current_combination, alpha, copy.deepcopy(Q),
                                                        np.mean(rewards_list), previous_action, learning_mode)
                                if learning_mode.value is QMode.Q_LEARNING.value:
                                    Q = update_Q_QL(previous_state, current_combination, alpha, gamma, copy.deepcopy(Q),
                                                    np.mean(rewards_list), previous_action)
                                if learning_mode.value is QMode.SARSA.value:
                                    Q, next_action = update_Q_SARSA(previous_state, current_state, alpha, gamma,
                                                                    copy.deepcopy(Q), np.mean(rewards_list),
                                                                    previous_action,
                                                                    exploration_mode, epsilon, temperature,
                                                                    explorationDegree)
                                if learning_mode.value is QMode.TD_ZERO.value:
                                    Q = update_TD_ZERO(previous_state, current_state, alpha, gamma,
                                                       copy.deepcopy(Q), np.mean(rewards_list),
                                                       previous_action)
                                    # print("NEXT_ACTION: {}".format(next_action))
                            # if not shortest path -> choose new action
                            if learning_mode.value != QMode.SHORTEST_PATH.value:
                                if learning_mode.value == QMode.TD_ZERO.value:
                                    action = get_action_TD_zero(exploration_mode, current_state, Q, epsilon,
                                                                temperature,
                                                                explorationDegree, actions, state_transitions)
                                else:
                                    if learning_mode.value == QMode.SARSA.value:
                                        # first time
                                        if len(next_action) > 0:
                                            action = copy.deepcopy(next_action)
                                        else:
                                            action = get_action(exploration_mode, current_state, Q, epsilon,
                                                                temperature, explorationDegree)
                                    else:
                                        action = get_action(exploration_mode, current_state, Q, epsilon, temperature,
                                                            explorationDegree)
                                # do the action (if it is a transition) (send it into the pipe):

                                if action_mode.value == ActionMode.ONE_FLOW.value:
                                    # print("ACTION: {}".format(action))
                                    pipe.send(action)
                                elif action_mode.value == ActionMode.DIRECT_CHANGE.value:
                                    pipe.send(action)
                                previous_action = copy.deepcopy(action)
                                previous_state = copy.deepcopy(current_state)
                                # find out next state:
                                if action_mode.value == ActionMode.DIRECT_CHANGE.value:
                                    current_state = get_next_state(state_transitions, current_state, action, True)
                                else:
                                    current_state = get_next_state(state_transitions, current_state, action)

                                print("Action: {} Next State: {} PrevState: {} PrevReward: {}".format(previous_action,
                                                                                                      current_state,
                                                                                                      previous_state,
                                                                                                      np.mean(
                                                                                                          rewards_list)))
                            # log output
                            if (generalIterator % 100) < 1:
                                print("-------number of batch: {} epsilon: {}".format(generalIterator, epsilon))

                            # saving the reward if enough gathered
                            if not (savingIterator % savingRewardCounter) and savingIterator > 0:
                                save_csv_file(logPath, load_level, 'reward_controller', np.mean(reward_saving_list),
                                              generalIterator // measurements_for_reward, splitUpLoadLevels,
                                              iteration_split_up_flag, iterationsLevel)
                                save_csv_file(logPath, load_level, 'average_latency', np.mean(average_latency_list),
                                              generalIterator // measurements_for_reward, splitUpLoadLevels,
                                              iteration_split_up_flag, iterationsLevel)
                                reward_saving_list.clear()
                                average_latency_list.clear()

                            # saving the q-table (for DEBUG or to approximate agent actions)
                            if not (savingIterator % Interval_saving_q) and savingIterator > 0:
                                # Q , savingIterator, averageReward
                                savingValueArray.append((copy.deepcopy(Q), savingIterator // measurements_for_reward,
                                                         np.mean(reward_saving_list)))
                            generalIterator = generalIterator + 1
                            savingIterator = savingIterator + 1
                            delayedRewardCounter = 0
                            rewards_list.clear()
                        delayedRewardCounter += 1

                # check if exit -> time.time are seconds
                if int((time.time() - startingTime) / 60) > duration_per_minutes:
                    saveQ(Q)
                    print("Exited after {} steps (last load level)".format(generalIterator))
                    break


def get_action(exploration_mode, current_state, Q, epsilon, temperature, explorationDegree):
    """
    choose action
    @param exploration_mode:
    @param current_state:
    @param Q:
    @param epsilon:
    @param temperature:
    @param explorationDegree:
    @return:
    """
    if exploration_mode.value == ExplorationMode.CONSTANT_EPS.value \
            or exploration_mode.value == ExplorationMode.FALLING_EPS.value:
        action = get_action_eps_greedy(current_state, Q, epsilon)
    elif exploration_mode.value == ExplorationMode.SOFTMAX.value:
        action = get_action_softmax(Q, current_state, temperature)
    elif exploration_mode.value == ExplorationMode.UCB.value:
        action = get_action_ucb(Q, current_state, explorationDegree)
    return action


def get_action_TD_zero(exploration_mode, current_state, Q, epsilon, temperature, explorationDegree, actions,
                       nextStates):
    """
    choose action
    @param exploration_mode:
    @param current_state:
    @param Q:
    @param epsilon:
    @param temperature:
    @param explorationDegree:
    @return:
    """
    if exploration_mode.value == ExplorationMode.CONSTANT_EPS.value \
            or exploration_mode.value == ExplorationMode.FALLING_EPS.value:
        action = get_action_eps_greedy(current_state, Q, epsilon)
    elif exploration_mode.value == ExplorationMode.SOFTMAX.value:
        action = get_action_softmax(Q, current_state, temperature, actions, nextStates)
    elif exploration_mode.value == ExplorationMode.UCB.value:
        action = get_action_ucb(Q, current_state, explorationDegree)
    return action


def get_next_state(stateTransitions, currentState, action, direct=False):
    """
    returns next state
    @param stateTransitions: dict that contains the next state based on the tuple between current state and action
    @param currentState
    @param action
    @return: next state
    """
    nextState = {}
    actionTuple = tuple(action)
    if Config.action_mode.value == ActionMode.DIRECT_CHANGE.value:
        if action == 'NoTrans':
            return currentState
        else:
            return action
    else:
        if actionTuple[0] == 'NoTrans':
            return currentState
        for stateTrans in stateTransitions:
            if stateTrans[0] == currentState and stateTrans[1] == actionTuple:
                nextState = stateTrans[2]
    return nextState


def get_average_latency(currentPathCombination, latencyDict):
    """
    calculates the average (latency) value of all elements of a list
    @param currentPathCombination:
    @param latencyDict:
    @return:
    """
    latencyList = getCostsOfPaths(currentPathCombination, latencyDict)
    cost = 0
    for element in latencyList:
        cost += element
    avgLat = cost / len(latencyList)
    return avgLat


def get_reward(currentPathCombination, latencyDict):
    """
    calculates the reward as a square root sum quadratic /n of the latency
    @param currentPathCombination:
    @param latencyDict:
    @return:
    """
    latencyList = getCostsOfPaths(currentPathCombination, latencyDict)
    cost = 0
    for element in latencyList:
        cost += element ** 2
    sqrootLatency = math.sqrt(cost / len(latencyList))
    return -sqrootLatency


def get_reward_utilization(currentPathCombination, latencyDict, bandwidth_dict={}, max_bw_dict={}):
    """
    TODO:
    calculates the reward as a combination of utilisation and latency
    @param currentPathCombination:
    @param latencyDict:
    @return:
    """
    return 0


def getCostsOfPaths(currentPathCombination, latencyDict):
    """
    array of path costs
    @param currentPathCombination:
    @param latencyDict:
    @return: array of path costs
    """
    valueList = []
    for path in currentPathCombination:
        cost = get_path_cost(latencyDict, currentPathCombination[path])
        valueList.append(cost)
    return valueList


def update_Q_table(prevQ, paths_per_flow, merging_q_table_flag, action_mode, joinedFlowsSet={}):
    """
    updates the q table if a new flow is joined
    @param prevQ: previous Q table
    @param paths_per_flow: possible paths for all flows
    @param merging_q_table_flag: if the Q table should be merged
    @param joinedFlowsSet: a dict of the flows that are joined
    @return: new Q-Table, new actions, new possible state transitions
    """
    t0 = time.time()
    paths_per_flow_copied = copy.deepcopy(paths_per_flow)
    paths_per_flow_filtered = filter_possible_paths_by_hops(paths_per_flow_copied, paths_per_flow, 100)
    print("got filtered flows")
    new_states = get_possible_states(copy.deepcopy(paths_per_flow_filtered))
    print("got possible states: {}".format(len(new_states)))
    if Config.qMode.value != QMode.TD_ZERO.value:
        if action_mode.value == ActionMode.ONE_FLOW.value:
            actions = get_actions_for_states(new_states, paths_per_flow_filtered)
            stateTransitions = get_state_transitions(actions)
            Q = create_new_q_table(actions, False)
        elif action_mode.value == ActionMode.DIRECT_CHANGE.value:
            actions = get_actions_for_states_direct(new_states)
            stateTransitions = get_state_transitions_direct(actions)
            Q = create_new_q_table(actions, True)
    else:
        Q = create_new_value_table(new_states)
        actions = get_actions_for_states(new_states, paths_per_flow_filtered)
        stateTransitions = get_state_transitions(actions)

    # print("Q: {}".format(Q))
    print("got actions per states: {}".format(len(actions)))
    # print("got state transitions: {}".format(time.time() - t0))
    # matching
    # create Q table

    # if Q-Table should be merged
    if merging_q_table_flag:
        if len(prevQ) > 0 and len(joinedFlowsSet) and len(joinedFlowsSet) < MAX_LENGHT_DIFFSET:
            Q = merging_qtable(prevQ, Q, joinedFlowsSet)

    print("Time to merge: {} micro_sec".format((time.time() - t0) * 10 ** 6))
    print("Action Size: {}".format(len(actions)))
    return Q, actions, stateTransitions


def merging_qtable(prevQ, newQ, differenceSet):
    """
    merging a Q table by searching similar states
    @param prevQ: previous Q table
    @param newQ: new calculated q table
    @param differenceSet:
    @return: merged Q-table
    """
    newQCopy = copy.deepcopy(newQ)
    for state in newQ:
        for action in newQ[state]:
            actionId = action[0]
            if actionId not in differenceSet:
                # find the sate with the smallest difference
                dictStateStr = ''
                dictState = json.loads(state)
                # delete the flow IDs of difference
                for difference in differenceSet:
                    dictState.pop(difference)
                # necessary to find the constellation that matches -> just json.dump does not give deterministic order
                oldQKeysDict = list(prevQ.keys())
                for oldQComb in oldQKeysDict:  # NEED TO BUILD UP TUPLE
                    oldKeysSet = json.loads(oldQComb)
                    # other variant:
                    # oldKeysSet = list(json.loads(oldQComb).items())
                    # if len(oldKeysSet.difference(set(dictState.items()))) < 1 and len(set(dictState.keys()).difference(oldKeysSet)) < 1:
                    if oldKeysSet == dictState:
                        dictStateStr = oldQComb  # json.dumps(oldQComb)
                if (len(dictStateStr) > 0):
                    # clean up the action set
                    actionOld = json.loads(action)
                    if list(actionOld)[0] not in differenceSet:
                        newQCopy[state][action] = prevQ[dictStateStr][action]
    return newQCopy


def create_new_value_table(states, initValue=-math.inf):
    """
    creates a new value table
    :param states:
    :param initValue:
    :return:
    """
    Q = {}

    if Config.exploration_mode.value == ExplorationMode.SOFTMAX.value:
        initValue = Config.softmax_init_value
    for state in states:
        stateStr = json.dumps(state, sort_keys=True)
        Q[stateStr] = [0, initValue]
    print(Q)
    return Q


#    creates a new Q table based on the actions of the possible states
def create_new_q_table(actions, direct=False):
    """
    creates new q-table based on the states and actions
    @param actions
    @return: Q table
    """
    Q = {}
    for actionElement in actions:
        state = json.dumps(actionElement[0], sort_keys=True)
        if direct:
            action = json.dumps(actionElement[1][0], sort_keys=True)
        else:
            action = json.dumps(actionElement[1], sort_keys=True)
        if state not in Q.keys():
            Q[state] = {}
        # steps
        initValue = -math.inf
        # TODO: adapt init valued based on worst path
        if (Config.exploration_mode.value == ExplorationMode.SOFTMAX.value):
            initValue = Config.softmax_init_value

        Q[state][action] = [0, initValue, []]
    # NOTE: Just a debug feature
    if len(actions) == Config.number_of_actions and len(Config.Q_array_path) > 0:
        try:
            path = Config.log_path + "/" + Config.Q_array_path
            # print(path)
            with open(path) as json_file:
                Q = json.load(json_file)
                # print("newQTable: {}".format(Q))
        except:
            print("Q-array-file not found")
    return Q


def get_state_transitions(actions):
    """
    get the next state
    @param actions:
    @return: tuple (currentstate, action, nextstate)
    """
    stateTransitionPairs = []
    for action in actions:
        currentstate = action[0]
        id = action[1][0]
        nextPath = action[1][1]
        nextState = copy.deepcopy(currentstate)
        if 'NoTrans' not in id:
            # change the state
            nextState[id] = nextPath
        stateTransitionPairs.append((currentstate, action[1], nextState))
    return stateTransitionPairs


def get_state_transitions_direct(actions):
    """
    get the next state
    @param actions:
    @return: tuple (currentstate, action, nextstate)
    """
    stateTransitionPairs = []
    for action in actions:
        currentstate = action[0]
        nextState = action[1][0]
        stateTransitionPairs.append((currentstate, action[1], nextState))
    return stateTransitionPairs


# k^n, k.. possibilities, n.. flows
# 1 flows, 2 directions: 5 possibilities -> 5^2: 16
# 2 flows, 2 directions: 5 possibilities -> 5^4: 625
# 3 flows, 2 directions: 5 possibilities -> 5^6: 15625
# 4 flows, 2 directions: 5 possibilities -> 5^8: 390625
def get_possible_states(paths_per_flow):
    """
    get all possible states
    @param paths_per_flow:
    @return: states
    """
    t0 = time.time()
    flat = [[(k, v) for v in vs] for k, vs in paths_per_flow.items()]
    combinations = [dict(items) for items in it.product(*flat)]
    states = list(combinations)
    print("calcLenght possibleStates: {} micro_sec".format((time.time() - t0) * 6))
    return states


def filter_possible_paths_by_hops(paths_per_flow, chosen_paths, bound=1):
    """
    filter possible paths by maximum hops
    kicks out too long paths
    @param paths_per_flow:
    @param chosen_paths: current chosen
    @param bound: maximum amount of hops for considering flows in comparison to minimum lenght
    @return:
    """
    for flowId in paths_per_flow:
        minimumlenght = min([len(x) for x in paths_per_flow[flowId]])
        for path in paths_per_flow[flowId]:
            if len(path) > minimumlenght + bound:
                # that is not the current chosen one
                if chosen_paths[flowId] != path:
                    paths_per_flow[flowId].remove(path)
    return paths_per_flow


def get_actions_for_states(states, paths_per_flows):
    """
    get the possible actions
    @param states:
    @param paths_per_flows:
    @return: actions
    """
    actions = []
    for state in states:
        otherPaths = copy.deepcopy(paths_per_flows)
        for flowId in state:
            # find out other combinations
            chosenPath = state[flowId]
            # all the other paths
            for otherPathById in otherPaths[flowId]:
                # kick out same paths of combinations
                if (otherPathById == chosenPath):
                    otherPaths[flowId].remove(otherPathById)
            # now, build possible next actions
            for chosenPath in otherPaths[flowId]:
                actions.append((state, (flowId, chosenPath)))
        actions.append((state, ('NoTrans', [])))
    return actions


def get_actions_for_states_direct(states):
    """
    get the possible actions for a direct change
    @param states:
    @param paths_per_flows:
    @return: actions
    """
    actions = []
    for state in states:
        stateChanges = []
        other_states = copy.deepcopy(states)
        # kick out original state
        other_states.remove(state)

        for next_state in other_states:
            # find out which states should be changed
            for flowId in state:
                pathState = state[flowId]
                pathOtherState = next_state[flowId]
                if pathState != pathOtherState:
                    stateChanges.append((flowId, pathOtherState))
            actions.append((state, (next_state, copy.deepcopy(stateChanges))))
        actions.append((state, ('NoTrans', [])))
    return actions


def get_actions_per_current_state(chosenPaths, paths_per_flow):
    """
    gets possible actions for the current state
    @param chosenPaths:
    @param paths_per_flow:
    @return: actions per state
    """
    actions = []
    otherPaths = copy.deepcopy(paths_per_flow)
    for chosenPath in chosenPaths:
        idPath = chosenPath
        selectedPath = chosenPaths[idPath]
        # cleaning up so possible actions get clear
        for path in otherPaths[idPath]:  # is it idPath?
            print("Path[0]: {}".format(path[0]))
            if path[0] == selectedPath:
                otherPaths[idPath].remove(path)
    ids = otherPaths.keys()
    for id in ids:
        for possiblePath in otherPaths[id]:
            actions.append((id, chosenPaths[id], possiblePath[0]))
    return actions


# Q(s,a) <- Q(S,a) + alpha[R + gamma*max Q(S',a)-Q(S,a)]
# tracking a non stationary problem
def update_Q_bandit(currentState, nextState, alpha, Q, reward, action, learning_mode):
    """
    updates Q table based on the multiarmed bandit method
    @param currentState:
    @param nextState:
    @param alpha: learning rate
    @param Q:
    @param reward:
    @param action:
    @param learning_mode:
    @return: updated Q table
    """
    # cambiamos
    stateNowStr = json.dumps(currentState, sort_keys=True)
    nextStateStr = json.dumps(nextState, sort_keys=True)
    actionStr = json.dumps(action, sort_keys=True)
    try:
        # if chosen Q-value is set infinity -> necessary to change
        if math.isinf(Q[stateNowStr][actionStr][1]):
            Q[stateNowStr][actionStr][1] = 0
        # weighted average: Q_(n+1) = (1-alpha)^n*Q_1 + sum(i=1 -> n) alpha * (1 - alpha)^(n-i) R_i)
        if learning_mode.value == QMode.MULTI_ARMED_BANDIT_WITH_WEIGHT.value:
            lastRewards = Q[stateNowStr][actionStr][2]
            n = len(lastRewards)
            q_n = 0
            # last one is highest weighted
            for i in range(0, n, 1):
                q_n = q_n + alpha * (1 - alpha) ** (n - i) * lastRewards[i]
            Q[stateNowStr][actionStr][1] = q_n + alpha * (reward - q_n)
            # save the previous reward (list max elements)
            Q[stateNowStr][actionStr][2].append(reward)
            # kick one out if too much
            if (len(Q[stateNowStr][actionStr][2]) > MAX_PAST_REWARDS):
                Q[stateNowStr][actionStr][2].pop(0)
        # non weighted average; Q_n+1 = Q_n + 1/n * (R_n - Q_n)
        elif learning_mode.value == QMode.MULTI_ARMED_BANDIT_NO_WEIGHT.value:
            Q[stateNowStr][actionStr][1] = Q[stateNowStr][actionStr][1] + (1 / (Q[stateNowStr][actionStr][0])) * (
                    reward - Q[stateNowStr][actionStr][1])
        # total visits
        Q[stateNowStr][actionStr][0] = Q[stateNowStr][actionStr][0] + 1
    except KeyError:
        print("Q: {}".format(Q))
        print("StateNowStr: {}".format(stateNowStr))
    return Q


# calculate new Q-Value via Q-learning
def update_Q_QL(currentState, nextState, alpha, gamma, Q, reward, action):
    """
    updates Q table based on Q-Learning
    @param currentState:
    @param nextState:
    @param alpha: learning rate
    @param gamma: discount factor
    @param Q:
    @param reward:
    @param action:
    @return:
    """
    # cambiamos
    stateNowStr = json.dumps(currentState, sort_keys=True)
    nextStateStr = json.dumps(nextState, sort_keys=True)
    actionStr = json.dumps(action, sort_keys=True)
    keyMaxValue = keywithmaxActionval(Q[nextStateStr])
    # if chosen Q-value is set infinity -> necessary to change
    if math.isinf(Q[stateNowStr][actionStr][1]):
        qAction = 0
    else:
        qAction = copy.deepcopy(Q[stateNowStr][actionStr][1])
    if math.isinf(Q[nextStateStr][keyMaxValue][1]):
        qMax_t_plus_1 = 0
    else:
        qMax_t_plus_1 = copy.deepcopy(Q[nextStateStr][keyMaxValue][1])
    try:
        Q[stateNowStr][actionStr][1] = qAction + alpha * (reward + gamma * qMax_t_plus_1 - qAction)
        Q[stateNowStr][actionStr][0] = Q[stateNowStr][actionStr][0] + 1
    except KeyError:
        print("Q: {}".format(Q))
        print("StateNowStr: {}".format(stateNowStr))
    return Q


def update_TD_ZERO(current_state, nextState, alpha, gamma, Q, reward, action):
    print("UPDATING Q")
    stateNowStr = json.dumps(current_state, sort_keys=True)
    nextStateStr = json.dumps(nextState, sort_keys=True)
    actionStr = json.dumps(action, sort_keys=True)
    try:
        # if chosen Q-value is set infinity -> necessary to change
        if math.isinf(Q[stateNowStr][1]):
            qAction = 0
        else:
            qAction = copy.deepcopy(Q[stateNowStr][1])
        valueNextAction = Q[nextStateStr][1]
        if math.isinf(valueNextAction):
            valueNextAction = 0
        else:
            valueNextAction = copy.deepcopy(valueNextAction)
        print("value next action: {}".format(valueNextAction))
        Q[stateNowStr][1] = qAction + alpha * (reward + 0.1 * valueNextAction - qAction)
        Q[stateNowStr][0] = Q[stateNowStr][0] + 1
    except KeyError:
        print("Q: {}".format(Q))
        print("StateNowStr_TD_ZERO: {}".format(stateNowStr))
    # print(Q)
    return Q


def update_Q_SARSA(current_state, nextState, alpha, gamma, Q, reward, action, exploration_mode, epsilon,
                   temperature, explorationDegree):
    """
    updates Q table based on Q-Learning
    @param nextState:
    @param alpha:
    @param gamma:
    @param Q:
    @param reward:
    @param action:
    @return:
    @param current_state:
    @param nextAction:
    @param exploration_mode:
    @param epsilon:
    @param temperature:
    @param explorationDegree:

    """
    # cambiamos
    stateNowStr = json.dumps(current_state, sort_keys=True)
    nextStateStr = json.dumps(nextState, sort_keys=True)
    actionStr = json.dumps(action, sort_keys=True)
    # get next action
    actionFollowingStateKey = json.dumps(
        get_action(exploration_mode, nextState, Q, epsilon, temperature, explorationDegree),
        sort_keys=True)
    try:
        # if chosen Q-value is set infinity -> necessary to change
        if math.isinf(Q[stateNowStr][actionStr][1]):
            qAction = 0
        else:
            qAction = copy.deepcopy(Q[stateNowStr][actionStr][1])
        if math.isinf(Q[nextStateStr][actionFollowingStateKey][1]):
            q_next_action = 0
        else:
            q_next_action = copy.deepcopy(Q[nextStateStr][actionFollowingStateKey][1])

        Q[stateNowStr][actionStr][1] = qAction + alpha * (reward + gamma * q_next_action - qAction)
        Q[stateNowStr][actionStr][0] = Q[stateNowStr][actionStr][0] + 1
    except KeyError:
        print("Q: {}".format(Q))
        print("StateNowStr: {}".format(stateNowStr))
    return Q, json.loads(actionFollowingStateKey)


### fucntions for calculating.. maybe outsource to "functions"
def get_paths(latencyDict, src, dst):
    '''
    Get all paths from src to dst using DFS algorithm
    @param latencyDict: dict of all link latencuies
    @param src:
    @param dst:
    @return: possible paths
    '''
    if src == dst:
        # host target is on the same switch
        return [[src]]
    paths = []
    stack = [(src, [src])]
    while stack:
        (node, path) = stack.pop()
        for next in set(latencyDict[node].keys()) - set(path):
            if next is dst:
                paths.append(path + [next])
            else:
                stack.append((next, path + [next]))
    return paths


# can also be changed to BWs, or to hops
def get_link_cost(latencyDict, s1, s2):
    """
    returns the link cost
    @param latencyDict:
    @param s1: switch 1
    @param s2: switch 2
    @return:
    """
    linkCost = latencyDict[s2][s1]
    return linkCost


# get the cost of a path
def get_path_cost(latencyDict, path):
    """
    gets the cost of an path
    @param latencyDict:
    @param path:
    @return:
    """
    cost = 0
    for i in range(len(path) - 1):
        cost += get_link_cost(latencyDict, path[i], path[i + 1])
    return cost


def keywithmaxActionval(actions, element=2):
    """
        a) create a list of the dict's keys and values;
        b) return the key with the max value
    """
    v = list(actions.values())
    k = list(actions.keys())
    if element == 1:
        return k[v.index(max(v, key=firstElement))]
    else:
        return k[v.index(max(v, key=scndElement))]


def firstElement(e):
    return e[0]


def scndElement(e):
    return e[1]


def calc_epsilon(steps, mode):
    return 0.507928 - 0.08 * math.log(steps)
    # return  0.507928 - 0.05993925*math.log(steps)
    # return 0.15


# TODO: implement logic
def get_action_eps_greedy(current_state, Q, e_greedy, direct=False, actions={}, state_transitions={}):
    """
    chossing action based on eps_greedy
    @param current_state:
    @param Q:
    @param e_greedy:
    @return:
    """
    # first find the actions possible
    if Config.qMode.value != QMode.TD_ZERO:
        stateString = json.dumps(current_state, sort_keys=True)
        try:
            qActions = Q[stateString]
        except KeyError:
            print("Q: {}".format(Q))
            print("StateNowStr: {}".format(current_state))
        # take max value
        actionChosen = keywithmaxActionval(qActions)
        # take random decision, if value between 0-1 is smaller than e greedy
        if random.random() < e_greedy:
            listKeys = list(qActions.keys())
            # kick out chosenaction
            listKeys.remove(actionChosen)
            actionChosen = random.choice(listKeys)
            print("xxxxxxxxxxx Chosen randomly action: {} xxxxxxxxxxxxxxx".format(actionChosen))
        return json.loads(actionChosen)
    else:
        possible_actions = actions[current_state]
        nextStateDict = {}
        for action in possible_actions:
            nextState = get_next_state(state_transitions, current_state, action, True)


def get_action_softmax(Q, currentState, tau, actions={}, state_transitions={}):
    """
    get the action based on the softmax exploration strategy
    @param Q:
    @param currentState:
    @param tau: temperature parameter
    """
    currentStateStr = json.dumps(currentState, sort_keys=True)
    if Config.qMode.value != QMode.TD_ZERO.value:
        actions = copy.deepcopy(Q[currentStateStr])
        actionsKeys = list(actions)
        try:
            total = sum([np.exp(-1 / (actions[action][1] * tau), dtype=np.float128) for action in actions])
            probs = [(np.exp(-1 / (actions[action][1] * tau), dtype=np.float128) / total) for action in actions]
        except ZeroDivisionError:
            print("actions: {}".format(actions))
            print("total: {}".format(total))
        chosenKey = np.random.choice(actionsKeys, p=probs)
        return json.loads(chosenKey)
    else:
        # print(actions)
        possible_actions = []
        for action in actions:
            # print("ACTION_PARSE: {}, current: {}".format(action,currentState))
            if action[0] == currentState:
                possible_actions.append(copy.deepcopy(action))

        nextStateDict = {}
        actionsValue = {}
        for action in possible_actions:
            nextState = get_next_state(state_transitions, currentState, action[1])
            nextstateStr = json.dumps(nextState, sort_keys=True)
            value = Q[nextstateStr][1]
            actionsValue[json.dumps(action[1], sort_keys=True)] = value
        try:
            print("ActionsValue: {}".format(actionsValue))
            total = sum([np.exp(-1 / (actionsValue[action] * tau), dtype=np.float128) for action in actionsValue])
            probs = [(np.exp(-1 / (actionsValue[action] * tau), dtype=np.float128) / total) for action in actionsValue]
            print("probs: {}".format(probs))
        except ZeroDivisionError:
            print("actions: {}".format(actions))
            print("total: {}".format(total))
        actionsKeys = list(actionsValue.keys())
        try:
            chosenKey = np.random.choice(actionsKeys, p=probs)
        except ValueError:
            print("actionsValues: {}".format(actionsValue))
            print("possible_actions: {}".format(possible_actions))
        return json.loads(chosenKey)


def get_action_ucb(Q, currentState, c, direct=False, actions={}, state_transitions={}):
    """
    get the action based on the upper confident bound
    @param Q:
    @param currentState:
    @param iteratorTotal: how many times checked in state
    @param c: degree of exploration, c > 0
    """
    currentStateStr = json.dumps(currentState, sort_keys=True)
    if Config.qMode.value != QMode.TD_ZERO.value:
        qCurrentState = Q[currentStateStr]
        actions = list(qCurrentState.keys())
        values = {}
        iteratorStateVisits = 0
        for action in actions:
            iteratorStateVisits = iteratorStateVisits + qCurrentState[action][0]
        for action in actions:
            # if never chosen -> choose it!
            if qCurrentState[action][0] == 0:
                return json.loads(action)
            values[action] = qCurrentState[action][1] + c * np.sqrt(np.log(iteratorStateVisits) /
                                                                    qCurrentState[action][0])
        return json.loads(max(values, key=values.get))


def save_csv_file(logPath, loadLevel, fileName, reward, timepoint, splitUpLoadLevels, iterationsSplitUpflag, iteration):
    """
    saving the reward or latency in a csv file
    @param logPath:
    @param loadLevel:
    @param fileName:
    @param reward:
    @param timepoint:
    @param splitUpLoadLevels:
    @param iterationsSplitUpflag:
    @param iteration: number iteration
    """
    if splitUpLoadLevels:
        loadLevelStr = '/' + str(loadLevel)
    else:
        loadLevelStr = ''
    if iterationsSplitUpflag:
        iterationLevelStr = '/' + str(iteration)
    else:
        iterationLevelStr = ''
    dirStr = '{}{}{}'.format(logPath, iterationLevelStr, loadLevelStr)
    with open('{}/{}.csv'.format(dirStr, fileName), 'a') as csvfile:
        fileWriter = csv.writer(csvfile, delimiter=',')
        fileWriter.writerow([timepoint, reward, time.time()])


def clearing_save_file(logPath, loadLevel, fileName, splitUpLoadLevels, iterationsSplitUpflag, iteration):
    """
    empties a save file
    @param logPath:
    @param loadLevel:
    @param fileName:
    @param splitUpLoadLevels:
    @param iterationsSplitUpflag:
    @param iteration: number iteration
    """
    if splitUpLoadLevels:
        loadLevelStr = '/' + str(loadLevel)
        print("LoadLevelStr: {}".format(loadLevelStr))
    else:
        loadLevelStr = ''
    if iterationsSplitUpflag:
        iterationLevelStr = '/' + str(iteration)
    else:
        iterationLevelStr = ''
    dirStr = '{}{}{}'.format(logPath, iterationLevelStr, loadLevelStr)
    if not os.path.exists(dirStr):
        os.makedirs(dirStr)
    with open('{}/{}.csv'.format(dirStr, fileName), 'w') as file:
        file.write("# iterator, reward, timestamp \n")


###################### Debugging fcuntions #############################
def saveQ(Q, iteration=-1):
    if iteration > 0:
        filePath = '../logs/{}/Q_array.json'.format(iteration)
    else:
        filePath = '../logs/Q_array.json'
    with open(filePath, 'w') as file:
        json.dump(Q, file)  # use `json.loads` to do the reverse


def saveQBest(Q):
    with open('../Q_array_best.json', 'a') as file:
        json.dump(Q, file)  # use `json.loads` to do the reverse
