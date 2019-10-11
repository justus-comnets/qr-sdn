topology: Scenario_Four_switches_two_ways_6_hosts

start
ryu-manager Controller/remote_controller.py
(with inserted config)
python Mininet/Scenario_Four_switches_two_ways_6_hosts.py
