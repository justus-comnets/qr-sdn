from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSController
from mininet.node import CPULimitedHost, Host, Node
from mininet.node import OVSKernelSwitch, UserSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink, Intf
from thread import start_new_thread
import os, stat
import json
import time
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


#             s1
#  h11   10ms/     \10ms    h41
#     -- s_s          s_d --
#  h1m    10ms\     /10ms   h4m
#             sm
# m switches -> m paths in between source and destination
# m hosts -> m connections
###################################################################
############### Scenario - Scaling    #############################
###################################################################

def reset_load_level(loadLevel):
    requests.put('http://0.0.0.0:8080/simpleswitch/params/load_level', data=json.dumps({"load_level": loadLevel}))
    requests.put('http://0.0.0.0:8080/simpleswitch/params/reset_flag', data=json.dumps({"reset_flag": True}))

def reset_iteration(iteration):
    requests.put('http://0.0.0.0:8080/simpleswitch/params/iteration', data=json.dumps({"iteration": iteration}))
    requests.put('http://0.0.0.0:8080/simpleswitch/params/iteration_flag', data=json.dumps({"iteration_flag": True}))

def stop_controller():
    requests.put('http://0.0.0.0:8080/simpleswitch/params/stop_flag', data=json.dumps({"stop_flag": True}))

def startIperf(host1, host2, amount, port, timeTotal):
    #host2.cmd("iperf -s -u -p {} &".format(port))
    bw = float(amount)
    print("Host {} to Host {} Bw: {}".format(host1.name, host2.name, bw))
    command = "iperf -c {} -u -p {} -t {} -b {}M &".format(host2.IP(), port, timeTotal, bw)
    host1.cmd(command)

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

def minToSec(min):
    return min * 60

def four_switches_network():
    net = Mininet(topo=None,
                  build=False,
                  ipBase='10.0.0.0/8', link=TCLink)

    scaling_amount = Config.scaling_amount

    queue_lenght = Config.queue_lenght

    bw_max_dict = Config.bw_max_dict

    # linkarray
    linkArray = []
    splitUpLoadLevelsFlag = Config.split_up_load_levels_flag
    logs = Config.log_path
    # importante! the load levels for measurements
    loadLevels = Config.load_levels
    print("LoadLevel: {}".format(loadLevels))
    timeTotal = minToSec(Config.duration_iperf_per_load_level_minutes)
    controllerIP = '127.0.0.1'
    fileName = 'timestamp_changing_load_levels_mininet'
    info('*** Adding controller\n')
    c0 = net.addController(name='c0',
                           controller=RemoteController,
                           ip=controllerIP,
                           protocol='tcp',
                           port=6633)


    s_start = net.addSwitch('s1', cls=OVSKernelSwitch)
    s_end = net.addSwitch('s{}'.format(scaling_amount+2), cls=OVSKernelSwitch)

    switchList = []
    hostList_Start = []
    hostList_End = []
    switchList.append((s_start, 's0'))
    switchList.append((s_end, 's0'))
    for i in range(1, scaling_amount+1):
        # add switch
        info('*** Add switch: {}\n'.format('s{}'.format(i+1)))
        switchList.append((net.addSwitch('s{}'.format(i+1), cls=OVSKernelSwitch), 's{}'.format(i)))
        # add host
        hostList_Start.append(((net.addHost('hs{}'.format(i), cls=Host, ip='10.0.0.{}'.format(i), defaultRoute=None)), 'hs{}'.format(i)))
        hostList_End.append((net.addHost('he{}'.format(i), cls=Host, ip='10.0.1.{}'.format(i), defaultRoute=None),'he{}'.format(i)))

        # start link switches
        linkArray.append(net.addLink(s_start, switchList[-1][0], delay='10ms', bw=i*2.0, max_queue_size=queue_lenght))
        # end link switches
        linkArray.append(net.addLink(switchList[-1][0], s_end, delay='10ms', bw=i*2.0, max_queue_size=queue_lenght))

        # link host - switch
        linkArray.append(net.addLink(hostList_Start[-1][0], s_start))
        linkArray.append(net.addLink(hostList_End[-1][0], s_end))

    info('*** Starting network\n')
    net.build()
    info('*** Starting controllers\n')
    for controller in net.controllers:
        controller.start()

    info('*** Starting switches\n')
    for switch in switchList:
        switch[0].start([c0])


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
    #if iteration_split_up_flag:
        #reset_iteration(0)
    # incrementing the load
    for iteration in range(iterations):
        clearingSaveFileIterations(fileName, logs, iterations)
        # send load level
        print("(Re)starting iperf -- i: {}".format(iteration))
        for j in range(0, scaling_amount):
            start_new_thread(startIperf, (hostList_Start[j][0], hostList_End[j][0], 1.75+j*2.0, 5001, timeTotal))
        time.sleep(timeTotal)


        if iteration_split_up_flag and iteration < iterations - 1:
            reset_iteration(iteration + 1)
            time.sleep(1)
    stop_controller()
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')

four_switches_network()
