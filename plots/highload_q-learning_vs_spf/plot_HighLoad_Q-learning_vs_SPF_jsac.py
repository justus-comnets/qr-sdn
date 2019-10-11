import matplotlib.pyplot as plt
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("..")
import plotting

plotting.setup()

linestyles = ["solid", "dashed", (0, (3, 1, 1, 1)), "-.", (0, (1, 1)), (0, (3, 10, 1, 10))]

parser = argparse.ArgumentParser(description='Plot average latency vs. steps for different approaches')

parser.add_argument('--folder', default="./data_HighLoad_Q-learning_vs_SPF", type=str, help='Specify folder of measurements')
parser.add_argument('--save', default=".", type=str, help='Specify save directory')
parser.add_argument('--loadlevel', default="10", type=str, help='Specify load level (4 or 10)')
args = parser.parse_args()

folderpath = args.folder

dtype = {"step": "float", "latency": "float", "timestamp": "float"}
dtypenames = ["step", "latency", "timestamp"]

# check how many folders
dataDict = {}


def get_subdirs(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

colors = [(177, 179, 180), (123, 6, 100), (227, 34, 25), (240, 138, 0), (250, 187, 0), (137, 186, 23), (0, 98, 95),
          (0, 169, 212)]

algos = get_subdirs(folderpath)

for algo in algos:
    dataDict[algo] = {}
    iterations = get_subdirs(folderpath + "/" + algo)
    for it in iterations:
        dataDict[algo][it] = {"latency": [], "step": []}
        load_level = args.loadlevel

        dirStr = folderpath + "/" + algo + '/' + it + '/' + load_level + "/average_latency.csv"
        data = pd.read_csv(dirStr, delimiter=',', names=dtypenames, dtype=dtype, skiprows=1)

        if algo == "SPF":
            dataDict[algo][it]["latency"] = data["latency"]
            dataDict[algo][it]["step"] = data["step"]
        elif algo == "QR-SDN":
            dataDict[algo][it]["latency"] = data["latency"]
            dataDict[algo][it]["step"] = data["step"]


fig, ax = plt.subplots()
length = 150

for c, algo in enumerate(dataDict.keys()):
    latencies = np.array(
        [np.array(dataDict[algo][it]["latency"][:length]) for it in dataDict[algo].keys()])
    mean_latencies = np.mean(latencies, axis=0)[:length]
    upper_percentiles = np.percentile(latencies[:length], 95, axis=0)
    lower_percentiles = np.percentile(latencies[:length], 5, axis=0)

    plt.plot(range(len(mean_latencies)), mean_latencies, label="{}".format(algo), linestyle=linestyles[c])
    plt.fill_between(range(len(mean_latencies)), upper_percentiles, lower_percentiles, alpha=.2)

ax.set_xlabel('Steps')
ax.set_ylabel('Avg. latency (ms)')
plt.legend()
plt.tight_layout()
plt.savefig(args.save + "/HighLoad_Q-learning_vs_SPF_jsac.pdf")
plt.show()
