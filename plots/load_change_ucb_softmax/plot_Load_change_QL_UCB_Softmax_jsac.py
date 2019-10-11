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

parser.add_argument('--folder', default="./data_Load_change_QL_UCB_Softmax", type=str, help='Specify folder of measurements')
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
    if algo == "QR-SDN" or algo == "SPF":
        continue
    dataDict[algo] = {}
    iterations = get_subdirs(folderpath + "/" + algo)
    for it in iterations:
        dataDict[algo][it] = {"latency": [], "step": []}

        dirStr = folderpath + "/" + algo + '/' + it + "/average_latency.csv"
        data = pd.read_csv(dirStr, delimiter=',', names=dtypenames, dtype=dtype, skiprows=1)

        dataDict[algo][it]["latency"] = data["latency"]
        dataDict[algo][it]["step"] = data["step"]
        dataDict[algo][it]["timestamp"] = data["timestamp"]

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
ax3 = ax2.twinx()


length = 770
plot_start = 370
plot_stop = 450

N = 40
threshold = 0.40
num = 0
lower_bound = 25.333
width = 0.3  # the width of the bars

xticks_bar = []
xticks_box = []
box_overheads = []
box_convsteps = []
box_convtimes = []


loadlevel_change = 0

for num, algo in enumerate(dataDict.keys()):
    latencies = np.array(
        [np.array(dataDict[algo][it]["latency"][:length]) for it in dataDict[algo].keys()])
    timestamps = np.array(
        [dataDict[algo][it]["timestamp"][:length] for it in dataDict[algo].keys()])
    mean_latencies = np.mean(latencies, axis=0)[:length][plot_start: plot_stop]
    upper_percentiles = np.percentile(latencies[:length], 95, axis=0)[plot_start: plot_stop]
    lower_percentiles = np.percentile(latencies[:length], 5, axis=0)[plot_start: plot_stop]

    conv_steps = []
    conv_times = []
    conv_flags = []
    for n, latency in enumerate(latencies[loadlevel_change:]):
        mv_avg = pd.Series(latency).rolling(window=N).mean().iloc[N - 1:].values
        mv_avg_time = pd.Series(timestamps[n][loadlevel_change:]).rolling(window=N).mean().iloc[N - 1:].values
        diff = np.diff(mv_avg, n=1)
        conv_flags.append(False)
        conv_step = len(diff)
        start_time = timestamps[n][0]
        conv_time = mv_avg_time[-1] - start_time
        for c, d in enumerate(diff):
            if abs(d) < threshold and not conv_flags[n]:
                conv_step = c
                conv_time = mv_avg_time[c] - start_time
                conv_flags[n] = True
            elif abs(d) >= threshold:
                conv_flags[n] = False
                conv_step = len(diff)
                conv_time = mv_avg_time[-1] - start_time
        if conv_flags[n]:
            conv_steps.append(conv_step)
            conv_times.append(conv_time)
    print(algo, conv_steps)
    box_convsteps.append(conv_steps)
    box_convtimes.append(conv_times)

    overheads = []
    c = 0
    for n, latency in enumerate(latencies):
        if conv_flags[n]:
            overheads.append(sum(latency[conv_steps[c]:]) / len(latency[conv_steps[c]:]))
            c += 1
        else:
            continue
    box_overheads.append(overheads)

    ax.plot(range(plot_start, plot_stop), mean_latencies, label="{}".format(algo), linestyle=linestyles[num])
    ax.fill_between(range(plot_start, plot_stop), upper_percentiles, lower_percentiles, alpha=.2)

bp_conv = ax2.boxplot(box_convsteps, positions=np.array(range(len(box_convsteps))) - width / 2, sym='', showfliers=True, patch_artist=True,
                      medianprops=dict(color="black", linewidth=2), whiskerprops=dict(linewidth=2),
                      boxprops=dict(facecolor="limegreen", hatch="///", linewidth=2))

bp_over = ax3.boxplot(box_overheads, positions=np.array(range(len(box_overheads))) + width / 2, sym='', showfliers=True, patch_artist=True,
                      medianprops=dict(color="black", linewidth=2), whiskerprops=dict(linewidth=2),
                      boxprops=dict(facecolor="salmon", hatch="ooo", linewidth=2))

ax2.legend([bp_conv['boxes'][0], bp_over['boxes'][0]], ["Convergence", "Avg. latency"],
           bbox_to_anchor=(0, 1.01, 1, 1.35), loc="lower left",
           mode="expand", borderaxespad=0, ncol=2)
ax2.set_xticklabels(list(dataDict.keys()))
ax2.set_xlabel("Exploration strategy")
ax2.set_ylabel("Time till convergence (steps)")
ax3.set_ylabel("Avg. latency (ms)")

ax.set_xlabel('Steps')
ax.set_ylabel('Avg. latency (ms)')
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(args.save + "/Load_change_QL_UCB_Softmax_steps_jsac.pdf")
fig2.tight_layout(rect=[0, -0.02, 1.02, 0.99])
fig2.savefig(args.save + "/Load_change_QL_UCB_Softmax_jsac.pdf")
plt.show()
