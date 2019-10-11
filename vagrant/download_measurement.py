import fabric as fb
import subprocess
import time

num_machines = 4


def get_ips(num=1):  # fetches IPs of vagrant machines
    dict_ips = {}
    for n in range(1, 1 + num):
        machine = "rl_routing{}".format(n)
        output = subprocess.check_output(
            ['vagrant ssh rl_routing{} -c "hostname -I" 2>/dev/null'.format(n)], shell=True)
        dict_ips[machine] = output.decode("utf-8").split(" ")[0]
    return dict_ips


dict_ips = get_ips(num_machines)
print(dict_ips)

connections = {}


timestamp = str(int(time.time()))
output = subprocess.check_output("mkdir {}".format(timestamp), shell=True)

for it, machine in enumerate(dict_ips.keys()):
    print("Machine: {}".format(it + 1))
    connections[machine] = fb.Connection(host=dict_ips[machine], user='vagrant')

    with connections[machine].cd('/home/vagrant/'):
        command = "tar -cf logs.{}.tar /home/vagrant/logs".format(it)
        result = connections[machine].run(command, pty=False)
    connections[machine].get("/home/vagrant/logs.{}.tar".format(it), local="{}/logs.{}.tar".format(timestamp, it))
