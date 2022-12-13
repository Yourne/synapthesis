import matplotlib.pyplot as plt
import pandas as pd
import os


INPUTDIR = "data"
FNAME = "procset.csv"
data = pd.read_csv(os.path.join(INPUTDIR, "procset.csv"))

labels = ["framework ass.", "direct ass.", "negotiated proc.", "open proc."]

# "amount","duration","pa_med_ann_expenditure","be_med_ann_revenue"
var = "amount"


def durationhist() -> None:
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    for i, (proc, label) in enumerate(zip([26, 23, 4, 1], labels)):
        X = data[data["id_award_procedure"] == proc]["duration"]
        ax[i//2, i % 2].hist(
            X, bins="fd", range=(0, 3000), label=label, density=True, log=True)
        ax[i//2, i % 2].set_title(label)
    fig.supxlabel("days")
    fig.supylabel("probability")
    plt.show()

# AMOUNT

# proc = "framework"
# X = data[data["id_award_procedure"] == 26]["amount"]
# maxx = 1e7
# plt.hist(
#     X, bins="fd", range=(0, maxx), density=True, log=True)
# ticks = list(x * 1e6 for x in range(0, 11, 2))
# plt.xticks(ticks, [f"{str(int(x/1e6))}M" for x in ticks])
# plt.xlabel("euro")
# plt.ylabel("probability")
# plt.savefig("histfigures/amount_" + proc +".png")

# proc = "direct"
# X = data[data["id_award_procedure"] == 23]["amount"]
# maxx = 3e5
# plt.hist(
#     X, bins="fd", range=(0, maxx), density=True, log=True)
# ticks = list(x * 1e3 for x in range(0, 301, 50))
# plt.xticks(ticks, [f"{str(int(x / 1e3))}K" for x in ticks])
# plt.xlabel("euro")
# plt.ylabel("probability")
# plt.savefig("histfigures/amount_" + proc +".png")

# proc = "negotiated"
# X = data[data["id_award_procedure"] == 4]["amount"]
# maxx = 4e6
# plt.hist(
#     X, bins="fd", range=(0, maxx), density=True, log=True)
# ticks = list(x * 1e6 for x in range(0, 5))
# plt.xticks(ticks, [f"{str(int(x/1e6))}M" for x in ticks])
# plt.xlabel("euro")
# plt.ylabel("probability")
# plt.savefig("histfigures/amount_" + proc +".png")
# plt.show()

# proc = "open"
# X = data[data["id_award_procedure"] == 1]["amount"]
# maxx = 1e7
# plt.hist(
#     X, bins="fd", range=(0, maxx), log=True, density=True)
# ticks = list(x * 1e6 for x in range(0, 11, 2))
# plt.xticks(ticks, [f"{str(x/1e6)}M" for x in ticks])
# plt.xlabel("euro")
# plt.ylabel("probability")
# plt.savefig("histfigures/amount_" + proc +".png")


# BE MEDIAN REV
# var = "be_med_ann_revenue"
# proc = "framework"
# X = data[data["id_award_procedure"] == 26][var]
# plt.hist(
#     X, bins="fd", density=True, log=True)
# ticks = list(x * 1e7 for x in range(0, 13, 2))
# plt.xticks(ticks, [f"{str(int(x/1e6))}M" for x in ticks])
# plt.xlabel("euro")
# plt.ylabel("probability")
# plt.savefig("histfigures/" + var + "_" + proc + ".png")
# plt.show()

var = "be_med_ann_revenue"
proc = "direct"
X = data[data["id_award_procedure"] == 23][var]
plt.hist(
    X, bins="fd",  range=(0, 4e7), density=True, log=True)
ticks = list(x * 1e6 for x in range(0, 41, 5))
plt.xticks(ticks, [f"{str(int(x/1e6))}M" for x in ticks])
plt.xlabel("euro")
plt.ylabel("probability")
plt.savefig("histfigures/" + var + "_" + proc + ".png")
plt.show()

