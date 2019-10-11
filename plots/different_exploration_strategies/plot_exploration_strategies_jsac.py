import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
import os
import numpy as np
import pandas as pd
import sys

sys.path.append("..")
import plotting

plotting.setup(span=True)

linestyles = ["solid", "dashed", (0, (3, 1, 1, 1)), "-.", (0, (1, 1)), (0, (3, 10, 1, 10))]
bar_colors = ["lightgreen", "salmon", "lightblue", "lightgray", "m", "gold", "darkgrey", "darkorchid", "cornflowerblue",
              "turquoise"]
hatch_patterns = ('---', '\\\\', '+++', 'xxx', 'o', '*', 'O', '.', "ooo", "+")

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

parser = argparse.ArgumentParser(description='Plot average latency vs. load')

parser.add_argument('--folder', default="./data_different_exploration_strategies", type=str,
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


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


algos = get_subdirs(folderpath)

for algo in algos:
    dataDict[algo] = {}
    params = get_subdirs(folderpath + "/" + algo)
    params = sorted(params, key=float)
    print(params)
    if algo == "Softmax":
        params = params[:4]
    elif algo == "Softmax-NOpt":
        params = params[:2]
    for param in params:
        dataDict[algo][param] = {}
        iterations = get_subdirs(folderpath + "/" + algo + "/" + param)
        for it in iterations:
            load_level = args.loadlevel
            dataDict[algo][param][it] = {"latency": [], "step": [], "timestamp": []}

            dirStr = folderpath + "/" + algo + '/' + param + '/' + it + '/' + load_level + "/average_latency.csv"
            data = pd.read_csv(dirStr, delimiter=',', names=dtypenames, dtype=dtype, skiprows=1)

            dataDict[algo][param][it]["latency"] = data["latency"]
            dataDict[algo][param][it]["step"] = data["step"]
            dataDict[algo][param][it]["timestamp"] = data["timestamp"]

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
ax5 = ax4.twinx()

fig6, ax6 = plt.subplots()
ax7 = ax6.twinx()

plotting.setup(span=False)
fig8, ax8 = plt.subplots()
ax9 = ax8.twinx()

length = 600
N = 40
threshold = 0.40
num = 0
lower_bound = 25.333
width = 0.3  # the width of the bars

# xticks_bar = 0
# for algo in dataDict.keys():
#     for param in dataDict[algo].keys():
#         xticks_bar += 1
xticks_bar = []
xticks_box = []
box_overheads = []
box_convsteps = []
box_convtimes = []

for algo in dataDict.keys():
    for param in sorted(dataDict[algo].keys(), key=float):
        latencies = np.array(
            [dataDict[algo][param][it]["latency"][:length] for it in dataDict[algo][param].keys()])
        timestamps = np.array(
            [dataDict[algo][param][it]["timestamp"][:length] for it in dataDict[algo][param].keys()])
        mean_latencies = np.mean(latencies, axis=0)[:length]
        # steps = dataDict[algo][param]["0"]["step"][:length]
        ax.plot(range(len(mean_latencies)), mean_latencies)

        # mov_avg = running_mean(mean_latencies, N)
        mov_avg = pd.Series(mean_latencies).rolling(window=N).mean().iloc[N - 1:].values
        ax2.plot(range(len(mov_avg)), mov_avg, label="{},{}".format(algo, param))

        diff_avg = np.diff(mov_avg, n=1)
        ax3.plot(range(len(diff_avg)), diff_avg)

        conv_steps = []
        conv_times = []
        conv_flags = []
        for n, latency in enumerate(latencies):
            mv_avg = pd.Series(latency).rolling(window=N).mean().iloc[N - 1:].values
            mv_avg_time = pd.Series(timestamps[n]).rolling(window=N).mean().iloc[N - 1:].values
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
        # mean_conv_step = np.mean(conv_steps)
        # upper_percentile_conv_step = np.percentile(conv_steps, 95)
        # lower_percentile_conv_step = np.percentile(conv_steps, 5)
        print(algo, param, conv_times)
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
        # mean_overhead = np.mean(overheads)
        # upper_percentile_overhead = np.percentile(overheads, 95)
        # lower_percentile_overhead = np.percentile(overheads, 5)
        box_overheads.append(overheads)

        conv_step = len(diff_avg)
        conv_flag = False
        for c, d in enumerate(diff_avg):
            # print(diff)
            if abs(d) < threshold and not conv_flag:
                conv_step = c
                conv_flag = True
            elif abs(d) >= threshold:
                conv_flag = False
                conv_step = len(diff_avg)
        print(algo, param, conv_step, conv_flag)

        # ax2.axvline(steps[int(N / 2):int(len(steps) - (N / 2))][conv_step], linestyle="--",
        #             label="{},{}".format(algo, param), color=colors[num])

        if conv_flag:
            xticks_bar.append("{},{}".format(algo, param))
            overhead = sum(mov_avg[conv_step:]) / len(mov_avg[conv_step:])
            ax4.bar(len(xticks_bar) - width / 2, conv_step, width, color=bar_colors[num], hatch=hatch_patterns[num],
                    edgecolor='black', linewidth='0.5', label="{},{}".format(algo, param))
            ax5.bar(len(xticks_bar) + width / 2, overhead, width, color=bar_colors[num], hatch=hatch_patterns[num],
                    edgecolor='black', linewidth='0.5')

        if True in conv_flags:
            xticks_box.append("{},\n{}".format(algo, param))

        num += 1

print(xticks_box)

ticks_softmax = []
conv_steps_softmax = []
overheads_softmax = []
conv_times_softmax = []

for c, tick in enumerate(xticks_box):
    if "Softmax" in tick and ("0.0001" in tick or "0.0005" in tick):
        conv_steps_softmax.append(box_convsteps[c])
        conv_times_softmax.append(box_convtimes[c])
        overheads_softmax.append(box_overheads[c])

        if "Inf" in tick:
            tick = "Softmax,\n" + "Inf," + tick.split("\n")[1]
        ticks_softmax.append(tick)

# NOTE: remove Softmax-Inf measurements
remove_list = [c for c, tick in enumerate(xticks_box) if "Inf" in tick]
xticks_box = [tick for tick in xticks_box if "Inf" not in tick]

for c in sorted(remove_list, reverse=True):
    len(box_convsteps)
    del box_convsteps[c]
    del box_convtimes[c]
    del box_overheads[c]

bp_conv = ax8.boxplot(conv_steps_softmax, positions=np.array(range(len(conv_steps_softmax))) - width / 2, sym='',
                      widths=width,
                      showfliers=True, patch_artist=True,
                      boxprops=dict(facecolor="lightgreen", hatch='///', linewidth=2),
                      medianprops=dict(color="black", linewidth=2), whiskerprops=dict(linewidth=2))

bp_over = ax9.boxplot(overheads_softmax, positions=np.array(range(len(overheads_softmax))) + width / 2, sym='',
                      widths=width,
                      showfliers=True, patch_artist=True, boxprops=dict(facecolor="salmon", hatch='ooo', linewidth=2),
                      medianprops=dict(color="black", linewidth=2), whiskerprops=dict(linewidth=2))

ax8.legend([bp_conv['boxes'][0], bp_over['boxes'][0]], ["Convergence", "Avg. latency"],
           bbox_to_anchor=(0, 1.01, 1, 1.3), loc="lower left",
           mode="expand", borderaxespad=0, ncol=2)
ax8.set_xticks(np.array(range(len(conv_steps_softmax))))
ax8.set_xticklabels(ticks_softmax)
ax8.set_xlabel("Algorithm, parameter")
ax8.set_ylabel("Time till convergence (steps)")
ax9.set_ylabel("Avg. latency (ms)")

bp_conv = ax6.boxplot(box_convsteps, positions=np.array(range(len(box_convsteps))) - width / 2, sym='', widths=width,
                      showfliers=True, patch_artist=True,
                      boxprops=dict(facecolor="lightgreen", hatch='///', linewidth=2),
                      medianprops=dict(color="black", linewidth=2), whiskerprops=dict(linewidth=2))

bp_over = ax7.boxplot(box_overheads, positions=np.array(range(len(box_overheads))) + width / 2, sym='', widths=width,
                      showfliers=True, patch_artist=True, boxprops=dict(facecolor="salmon", hatch='ooo', linewidth=2),
                      medianprops=dict(color="black", linewidth=2), whiskerprops=dict(linewidth=2))

ax6.legend([bp_conv['boxes'][0], bp_over['boxes'][0]], ["Convergence", "Avg. latency"], loc="upper center")

ax6.set_xticks(np.array(range(len(box_convsteps))))
ax6.set_xticklabels(xticks_box)
ax6.set_xlabel("Algorithm, parameter")
ax6.set_ylabel("Time till convergence (steps)")
ax7.set_ylabel("Avg. latency (ms)")

# ax2.plot(steps, [lower_bound] * len(steps), label="Lower Bound", color="b", linestyle="--")
ax4.set_xticks(range(1, len(xticks_bar) + 1))
ax4.set_xticklabels(xticks_bar)

ax4.set_xlabel("Algorithm, parameter")
ax4.set_ylabel("Time till convergence (steps)")
ax5.set_ylabel("Avg. latency since convergence (ms)")

fig.legend()
fig2.legend()
# fig3.legend()
# fig4.legend()
# fig6.legend()

ax.set_xlabel('Steps')
ax.set_ylabel('Avg. Latency in ms')

fig6.tight_layout()
fig6.savefig(args.save + "/exploration_strategies_jsac.pdf")

fig8.tight_layout(rect=[0, -0.02, 1.02, 0.99])
fig8.savefig(args.save + "/exploration_strategies__softmax_jsac.pdf")

plt.show()
