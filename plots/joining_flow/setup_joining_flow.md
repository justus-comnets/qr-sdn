topology: Scenario_Four_switches_two_ways_6_hosts_Flow_adding_random

start
ryu-manager Controller/remote_controller.py
(with inserted config)
python Mininet/Scenario_Four_switches_two_ways_6_hosts.py

random <-> SPF
bias = BiasRL.RANDOM <-> BiasRL.SPF

merging <-> Reinit
mergingQTableFlag = True <-> False
