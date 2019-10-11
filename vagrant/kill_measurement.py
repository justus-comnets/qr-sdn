import fabric as fb
import subprocess

num_machines = 4


def get_ips(num=15):  # fetches IPs of vagrant machines
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


for it, machine in enumerate(dict_ips.keys()):
    print("Machine: {}".format(it + 1))
    connections[machine] = fb.Connection(host=dict_ips[machine], user='vagrant')

    command = "mn -c"
    result = connections[machine].sudo(command, pty=False)

    try:
        command = "pkill ryu-manager"
        result = connections[machine].run(command, pty=False)
    except Exception as e:
        print(e)
    try:
        command = "sudo pkill python"
        result = connections[machine].run(command, pty=False)
    except Exception as e:
        print(e)
