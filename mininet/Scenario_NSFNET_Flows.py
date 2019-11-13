#!/usr/bin/python

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSController
from mininet.node import CPULimitedHost, Host, Node
from mininet.node import OVSKernelSwitch, UserSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink, Intf
from thread import start_new_thread
import random
import os, stat
import json
import time
import copy
import csv
import requests
import sys
sys.path.append("...")
sys.path.append("..")
sys.path.append("../controller")
sys.path.append(".")
print(os.getcwd())
print(sys.path.__str__())
from config import Config


###################################################################
############### Scenario - NSFNET    ##############################
###################################################################
# https://upload.wikimedia.org/wikipedia/commons/e/e5/NSFNET-backbone-T3.png


def reset_load_level(loadLevel):
    requests.put('http://0.0.0.0:8080/simpleswitch/params/load_level', data=json.dumps({"load_level": loadLevel}))
    requests.put('http://0.0.0.0:8080/simpleswitch/params/reset_flag', data=json.dumps({"reset_flag": True}))

def reset_iteration(iteration):
    requests.put('http://0.0.0.0:8080/simpleswitch/params/iteration', data=json.dumps({"iteration": iteration}))
    requests.put('http://0.0.0.0:8080/simpleswitch/params/iteration_flag', data=json.dumps({"iteration_flag": True}))

def stop_controller():
    requests.put('http://0.0.0.0:8080/simpleswitch/params/stop_flag', data=json.dumps({"stop_flag": True}))

def startIperf(host1, host2, amount, port, timeTotal, loadLevel):
    #host2.cmd("iperf -s -u -p {} &".format(port))
    bw = float(amount) * (float(loadLevel) / float(10))
    print("Host {} to Host {} Bw: {}".format(host1.name, host2.name, bw))
    command = "iperf -c {} -u -p {} -t {} -b {}M &".format(host2.IP(), port, timeTotal, bw)
    host1.cmd(command)

def write_in_File(fileName, logs, loadlevel, iteration_split_up_flag, iteration):
    dir = logs + '/'
    if iteration_split_up_flag:
        dir = dir + str(iteration) + '/'
    with open('{}{}.csv'.format(dir, fileName), 'a') as csvfile:
        fileWriter = csv.writer(csvfile, delimiter=',')
        fileWriter.writerow([loadlevel, time.time()])

def clearingSaveFile(fileName, logs):
    dir = logs + '/'
    with open('{}{}.csv'.format(dir, fileName), 'w') as file:
        file.write("# loadlevel, timestamp \n")

def clearingSaveFileIterations(fileName, logs, iterations):
    # cleans it all up
    for iteration in range(iterations):
        dir = logs + '/' + str(iteration) + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)
            # give folder rights
            os.chmod(dir, stat.S_IRWXO)
        with open('{}{}.csv'.format(dir, fileName), 'w') as file:
            file.write("# loadlevel, timestamp \n")

def minToSec(min):
    return min * 60

def myNetwork():

    net = Mininet( topo=None,
                   build=False,
                   ipBase='10.0.0.0/8')

    queue_lenght = Config.queue_lenght

    bw_max_dict = Config.bw_max_dict

    linkArray = []
    splitUpLoadLevelsFlag = Config.split_up_load_levels_flag
    logs = Config.log_path
    # importante! the load levels for measurements
    loadLevels = Config.load_levels
    print("LoadLevel: {}".format(loadLevels))
    timeTotal = minToSec(Config.duration_iperf_per_load_level_minutes)
    controllerIP = '127.0.0.1'
    fileName = 'timestamp_changing_load_levels_mininet'

    info( '*** Adding controller\n' )
    c0 = net.addController(name='c0',
                           controller=RemoteController,
                           ip=controllerIP,
                           protocol='tcp',
                           port=6633)

    info( '*** Add switches\n')
    s7 = net.addSwitch('s7', cls=OVSKernelSwitch)  # Houston
    s11 = net.addSwitch('s11', cls=OVSKernelSwitch)  # College Park
    s10 = net.addSwitch('s10', cls=OVSKernelSwitch)  # Cambridge
    s6 = net.addSwitch('s6', cls=OVSKernelSwitch)
    s3 = net.addSwitch('s3', cls=OVSKernelSwitch)
    s12 = net.addSwitch('s12', cls=OVSKernelSwitch)
    s9 = net.addSwitch('s9', cls=OVSKernelSwitch)
    s5 = net.addSwitch('s5', cls=OVSKernelSwitch)
    s2 = net.addSwitch('s2', cls=OVSKernelSwitch)  # Palo Alto
    s8 = net.addSwitch('s8', cls=OVSKernelSwitch)
    s1 = net.addSwitch('s1', cls=OVSKernelSwitch)
    s4 = net.addSwitch('s4', cls=OVSKernelSwitch)

    # Palo Alto - College Park   bidrectional
    # Palo Alto - Cambridge   bidrectional
    # Palo Alto - Houston   bidrectional

    info('*** Add hosts\n')
    comm_sessions = 6
    # switches = net.switches
    for i in range(1, 2*comm_sessions+1):
        h = net.addHost('h{}'.format(i), cls=Host, ip='10.0.0.{}'.format(i), defaultRoute=None)
        # s = random.choice(switches)
        # switches.remove(s)
        net.addLink(h, net.get('s{}'.format(i)))

    palo = net.get("h2")
    cambridge = net.get("h10")
    college = net.get("h11")
    houston = net.get("h7")
    # bw of the flows added
    bwFlows = 9.75
    flowArray = [[palo, college, bwFlows], [palo, cambridge, bwFlows], [palo, houston, bwFlows],
                 [college, palo, bwFlows], [cambridge, palo, bwFlows], [houston, palo, bwFlows]]


    info( '*** Add links\n')
    s1s4 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s1, s4, cls=TCLink , **s1s4)
    s1s2 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s1, s2, cls=TCLink , **s1s2)
    s2s3 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s2, s3, cls=TCLink , **s2s3)
    s3s7 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s3, s7, cls=TCLink , **s3s7)
    s7s12 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s7, s12, cls=TCLink , **s7s12)
    s4s6 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s4, s6, cls=TCLink , **s4s6)
    s5s6 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s5, s6, cls=TCLink , **s5s6)
    s6s7 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s6, s7, cls=TCLink , **s6s7)
    s5s8 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s5, s8, cls=TCLink , **s5s8)
    s8s9 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s8, s9, cls=TCLink , **s8s9)
    s9s10 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s9, s10, cls=TCLink , **s9s10)
    s10s12 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s10, s12, cls=TCLink , **s10s12)
    s11s12 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s11, s12, cls=TCLink , **s11s12)
    s9s11 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s9, s11, cls=TCLink , **s9s11)
    s2s5 = {'bw':10,'delay':'10ms', 'max_queue_size' : queue_lenght}
    net.addLink(s2, s5, cls=TCLink , **s2s5)

    info( '*** Starting network\n')
    net.build()
    info( '*** Starting controllers\n')
    for controller in net.controllers:
        controller.start()

    info( '*** Starting switches\n')
    net.get('s7').start([c0])
    net.get('s11').start([c0])
    net.get('s10').start([c0])
    net.get('s6').start([c0])
    net.get('s3').start([c0])
    net.get('s12').start([c0])
    net.get('s9').start([c0])
    net.get('s5').start([c0])
    net.get('s2').start([c0])
    net.get('s8').start([c0])
    net.get('s1').start([c0])
    net.get('s4').start([c0])

    iterations = Config.iterations
    if iterations > 1:
        iteration_split_up_flag = True
    else:
        iteration_split_up_flag = False

    # erasing previous file
    if not splitUpLoadLevelsFlag:
        if iteration_split_up_flag:
            clearingSaveFileIterations(fileName, logs, iterations)
        else:
            clearingSaveFile(fileName, logs)


    time.sleep(15)
    # if iteration_split_up_flag:
    #    reset_iteration(0)
    # incrementing the load

    # flowArray = [[h1, h2, 10.0], [h1, h2, 10.0], [h13, h43, 1.75]]
    # flowArray = []
    # for i in range(comm_sessions):
    #     hosts = net.hosts
    #     flowArray.append([hosts[i], hosts[i+comm_sessions], 9.5])
    # print(flowArray)

    for iteration in range(iterations):

        clearingSaveFileIterations(fileName, logs, iterations)
        for loadLevel in loadLevels:
            # iperf threads
            # if the load levels are not split up -> write the load level change
            if splitUpLoadLevelsFlag:
                reset_load_level(loadLevel)
            if not splitUpLoadLevelsFlag:
                write_in_File(fileName, logs, loadLevel, iteration_split_up_flag, iteration)
            # send load level
            print("(Re)starting iperf -- loadLevel:  {}".format(loadLevel))
            time.sleep(5)
            for flow in flowArray:
                start_new_thread(startIperf, (flow[0], flow[1], flow[2], 5001, timeTotal, loadLevel))
                time.sleep(0.1)

            time.sleep(timeTotal)
            # waiting additional 2 sec to reset states
            # time.sleep(2)

        # last load level past
        if not splitUpLoadLevelsFlag:
            # if iteration < iterations:
            # reset_load_level(-1)
            write_in_File(fileName, logs, -1, iteration_split_up_flag, iteration)
        if iteration_split_up_flag and iteration < iterations - 1:
            reset_iteration(iteration + 1)
            time.sleep(1)
    stop_controller()
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    myNetwork()

