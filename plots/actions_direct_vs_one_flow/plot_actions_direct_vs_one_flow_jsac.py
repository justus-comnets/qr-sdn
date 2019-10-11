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

bar_colors = ["lightgreen", "salmon", "lightblue", "lightgray", "m", "gold", "darkgrey", "darkorchid", "cornflowerblue",
              "turquoise"]
hatch_patterns = ("ooo", '\\\\', '---', '+++', 'xxx', 'o', '*', 'O', '.', "+")

parser = argparse.ArgumentParser(description='Plot average latency vs. steps for different approaches')

parser.add_argument('--folder', default="./data_actions_direct_vs_one_flow", type=str,
                    help='Specify folder of measurements')
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


algos = get_subdirs(folderpath)

for algo in algos:
    dataDict[algo] = {}
    iterations = get_subdirs(folderpath + "/" + algo)
    for it in iterations:
        dataDict[algo][it] = {"latency": [], "step": []}
        load_level = args.loadlevel

        dirStr = folderpath + "/" + algo + '/' + it + '/' + load_level + "/average_latency.csv"
        data = pd.read_csv(dirStr, delimiter=',', names=dtypenames, dtype=dtype, skiprows=1)

        dataDict[algo][it]["latency"] = data["latency"]
        dataDict[algo][it]["step"] = data["step"]

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
ax3 = ax2.twinx()

length = 430
plot_length = 430
N = 40
threshold = 0.4
width = 0.3

box_convsteps = []
box_overheads = []


for c, algo in enumerate(dataDict.keys()):
    latencies = np.array(
        [np.array(dataDict[algo][it]["latency"][:length]) for it in dataDict[algo].keys()])
    mean_latencies = np.mean(latencies, axis=0)[:length]
    upper_percentiles = np.percentile(latencies[:length], 95, axis=0)
    lower_percentiles = np.percentile(latencies[:length], 5, axis=0)

    conv_steps = []
    conv_flags = []
    for n, latency in enumerate(latencies):
        mv_avg = pd.Series(latency).rolling(window=N).mean().iloc[N - 1:].values
        diff = np.diff(mv_avg, n=1)
        conv_flags.append(False)
        conv_step = len(diff)
        for i, d in enumerate(diff):
            if abs(d) < threshold and not conv_flags[n]:
                conv_step = i
                conv_flags[n] = True
            elif abs(d) >= threshold:
                conv_flags[n] = False
                conv_step = len(diff)
        if conv_flags[n]:
            conv_steps.append(conv_step)
    box_convsteps.append(conv_steps)

    overheads = []
    i = 0
    for n, latency in enumerate(latencies):
        if conv_flags[n]:
            overheads.append(sum(latency[conv_steps[i]:]) / len(latency[conv_steps[i]:]))
            i += 1
        else:
            continue
    box_overheads.append(overheads)

    ax.plot(range(len(mean_latencies[:plot_length])), mean_latencies[:plot_length], label="{}".format(algo), linestyle=linestyles[c])
    ax.fill_between(range(len(mean_latencies[:plot_length])), upper_percentiles[:plot_length], lower_percentiles[:plot_length], alpha=.2)

bp_conv = ax2.boxplot(box_convsteps, positions=np.array(range(len(box_convsteps))) - width / 2, sym='', showfliers=True, patch_artist=True,
                      medianprops=dict(color="black", linewidth=2), whiskerprops=dict(linewidth=2),
                      boxprops=dict(facecolor="limegreen", hatch="//", linewidth=2))

bp_over = ax3.boxplot(box_overheads, positions=np.array(range(len(box_overheads))) + width / 2, sym='', showfliers=True, patch_artist=True,
                      medianprops=dict(color="black", linewidth=2), whiskerprops=dict(linewidth=2),
                      boxprops=dict(facecolor="salmon", hatch="ooo", linewidth=2))


ax2.legend([bp_conv['boxes'][0], bp_over['boxes'][0]], ["Convergence", "Avg. latency"], loc="lower center")
ax2.set_xticklabels(list(dataDict.keys()))
ax2.set_xlabel("Action type")
ax2.set_ylabel("Time till convergence (steps)")
ax3.set_ylabel("Avg. latency (ms)")

ax.set_xlabel('Steps')
ax.set_ylabel('Avg. latency (ms)')
fig.legend()

plt.tight_layout()
fig.savefig(args.save + "/actions_direct_vs_one_flow_jsac.pdf")
fig2.savefig(args.save + "/actions_direct_vs_one_flow_jsac_box.pdf")
plt.show()
