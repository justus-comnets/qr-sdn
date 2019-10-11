import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
import os
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import plotting

plotting.setup()

linestyles = ["solid", "dashed", (0, (3, 1, 1, 1)), "-.", (0, (1, 1)), (0, (3, 10, 1, 10))]

parser = argparse.ArgumentParser(description='Plot average latency vs. load')

parser.add_argument('--folder', default="./data_avg-lat_vs_load", type=str, help='Specify folder of measurements')
parser.add_argument('--save', default=".", type=str, help='Specify save directory')
args = parser.parse_args()

folderpath = args.folder

dtype = {"step": "float", "reward": "float", "timestamp": "float"}
dtypenames = ["step", "reward", "timestamp"]

# check how many folders
dataDict = {}


def get_subdirs(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


algos = get_subdirs(folderpath)

for algo in algos:
    dataDict[algo] = {}
    iterations = get_subdirs(folderpath + "/" + algo)
    for it in iterations:
        # dataDict[algo][it] = {}
        load_levels = get_subdirs(folderpath + "/" + algo + '/' + it)
        for ll in load_levels:

            if not ll in dataDict[algo].keys():
                dataDict[algo][ll] = {}
            if not it in dataDict[algo][ll].keys():
                dataDict[algo][ll][it] = {"latency": [], "step": []}

            dataDict[algo][ll][it] = {"reward": [], "step": []}
            dirStr = folderpath + "/" + algo + '/' + it + '/' + ll + "/average_latency.csv"
            data = pd.read_csv(dirStr, delimiter=',', names=dtypenames, dtype=dtype, skiprows=1)

            if algo == "SPF":
                dataDict[algo][ll][it]["reward"] = data["reward"]
                dataDict[algo][ll][it]["step"] = data["step"]
            elif algo == "QR-SDN":
                dataDict[algo][ll][it]["reward"] = data["reward"][200:50000]
                dataDict[algo][ll][it]["step"] = data["step"][200:50000]

fig, ax = plt.subplots()

lower_bounds = [20, 20, 25.333, 25.333, 25.333, 25.333, 25.333, 25.333, 25.333, 25.333]


for c, algo in enumerate(dataDict.keys()):
    load_levels = sorted([int(i) for i in dataDict[algo].keys()])
    load_levels = [str(s) for s in load_levels]
    mean_latencies = []
    upper_percentiles = []
    lower_percentiles = []

    for ll in load_levels:
        latencies = []
        for it in dataDict[algo][ll].keys():
            latencies.extend(dataDict[algo][ll][it]["reward"])
        mean_latencies.append(np.mean(latencies))
        upper_percentiles.append(np.percentile(latencies, 95))
        lower_percentiles.append(np.percentile(latencies, 5))
    load_levels = [int(i) * 10 for i in load_levels]

    plt.errorbar(load_levels, mean_latencies,
                 yerr=[np.subtract(mean_latencies, lower_percentiles), np.subtract(upper_percentiles, mean_latencies)],
                 label="{}".format(algo), linestyle=linestyles[c])
    plt.plot(load_levels, lower_bounds, label="Lower Bound", color="b", linestyle="dotted")

ax.set_xlabel('Load level (\%)')
ax.set_ylabel('Avg. latency (ms)')
plt.legend()
plt.tight_layout()
plt.savefig(args.save + "/avg-lat_vs_load_jsac.pdf")
plt.show()
