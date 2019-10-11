import matplotlib.pyplot as plt
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("..")
import plotting

plotting.setup()

linestyles = ["solid", "dashed", (0, (3, 1, 1, 1)), "-.", (0, (1, 1)), (0, (3, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1))]
bar_colors = ["lightgreen", "salmon", "lightblue", "lightgray", "m", "gold", "darkgrey", "darkorchid", "cornflowerblue",
              "turquoise"]
hatch_patterns = ('---', '\\\\', '+++', 'xxx', 'o', '*', 'O', '.', "ooo", "+")

parser = argparse.ArgumentParser(description='Plot average latency vs. steps for different approaches')

parser.add_argument('--folder', default="./data_scalability", type=str, help='Specify folder of measurements')
parser.add_argument('--save', default=".", type=str, help='Specify save directory')
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
    if algo == "SPF":
        continue
    dataDict[algo] = {}
    iterations = get_subdirs(folderpath + "/" + algo)
    for it in iterations:
        dataDict[algo][it] = {"latency": [], "step": []}

        dirStr = folderpath + "/" + algo + '/' + it + "/average_latency.csv"
        data = pd.read_csv(dirStr, delimiter=',', names=dtypenames, dtype=dtype, skiprows=1)

        dataDict[algo][it]["latency"] = data["latency"]
        dataDict[algo][it]["step"] = data["step"]

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
ax3 = ax2.twinx()


plot_start = 0
plot_stop = 5000000
threshold = 0.4
N = 40
width = 0.3

algo_N = {"2": 40, "3": 40, "4": 300}

box_overheads = []
box_convsteps = []
xticks_box = []

for num, algo in enumerate(sorted(dataDict.keys())):
    latencies = np.array(
        [np.array(dataDict[algo][it]["latency"]) for it in dataDict[algo].keys()])
    min_len = min([len(l) for l in latencies])

    latencies = [l[:min_len]for l in latencies]
    mean_latencies = np.mean(latencies[:min_len], axis=0)[plot_start: plot_stop]
    upper_percentiles = np.percentile(latencies[:min_len], 95, axis=0)[plot_start: plot_stop]
    lower_percentiles = np.percentile(latencies[:min_len], 5, axis=0)[plot_start: plot_stop]

    xticks_box.append("-".join(algo.split("_")))
    conv_steps = []
    conv_flags = []
    N = algo_N[algo]
    for n, latency in enumerate(latencies):
        mv_avg = pd.Series(latency).rolling(window=N).mean().iloc[N - 1:].values
        diff = np.diff(mv_avg, n=1)
        conv_flags.append(False)
        conv_step = len(diff)
        for c, d in enumerate(diff):
            if abs(d) < threshold and not conv_flags[n]:
                conv_step = c
                conv_flags[n] = True
            elif abs(d) >= threshold:
                conv_flags[n] = False
                conv_step = len(diff)
        if conv_flags[n]:
            conv_steps.append(conv_step)
    print(algo, conv_steps, np.mean(conv_steps))
    box_convsteps.append(conv_steps)

    overheads = []
    c = 0
    for n, latency in enumerate(latencies):
        if conv_flags[n]:
            overheads.append(sum(latency[conv_steps[c]:]) / len(latency[conv_steps[c]:]))
            c += 1
        else:
            continue
    box_overheads.append(overheads)

    ax.plot(range(len(mean_latencies)), mean_latencies, label="{}".format(algo), linestyle=linestyles[num])
    ax.fill_between(range(len(mean_latencies)), upper_percentiles, lower_percentiles, alpha=.2)

print(xticks_box)

bp_conv = ax2.boxplot(box_convsteps, positions=np.array(range(len(box_convsteps))) - width / 2, sym='', widths=width,
                      showfliers=True, patch_artist=True,
                      boxprops=dict(facecolor="lightgreen", hatch='///', linewidth=2),
                      medianprops=dict(color="black", linewidth=2), whiskerprops=dict(linewidth=2))
bp_over = ax3.boxplot(box_overheads, positions=np.array(range(len(box_overheads))) + width / 2, sym='', widths=width,
                      showfliers=True, patch_artist=True, boxprops=dict(facecolor="salmon", hatch='ooo', linewidth=2),
                      medianprops=dict(color="black", linewidth=2), whiskerprops=dict(linewidth=2))
ax2.legend([bp_conv['boxes'][0], bp_over['boxes'][0]], ["Convergence", "Avg. latency"], loc="lower right")  # ,
ax2.set_xticklabels(xticks_box, multialignment='center')


ax.set_xlabel('Steps')
ax.set_ylabel('Avg. latency (ms)')

ax2.set_xticks(np.array(range(len(box_convsteps))))
ax2.set_xlabel('N=M')
ax2.set_ylabel("Time till convergence (steps)")
ax3.set_ylabel('Avg. latency (ms)')


fig2.tight_layout()
fig2.savefig(args.save + "/scalability_jsac.pdf")
plt.show()
