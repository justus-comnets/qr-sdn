topology: Scenario_Four_switches_two_ways_6_hosts

start
ryu-manager Controller/remote_controller.py
(with inserted config)
python Mininet/Scenario_Four_switches_two_ways_6_hosts.py

change exploration strategies depending on the desired one
change the values of temperature, epsilon, c as desired
