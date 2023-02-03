import matplotlib.pyplot as plt
import pandas as pd
import os


INPUTDIR = "../../data10"
FNAME = "contracts.csv"
OUTDIR = "../../images/histfigures/cpv/"
data = pd.read_csv(os.path.join(INPUTDIR, FNAME), index_col="index")

labels = ["framework ass.", "direct ass.", "negotiated proc.", "open proc."]


# "amount","duration","pa_med_ann_expenditure","be_med_ann_revenue"


# def durationhist() -> None:
#     fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
#     for i, (proc, label) in enumerate(zip([26, 23, 4, 1], labels)):
#         X = data[data["id_award_procedure"] == proc]["duration"]
#         ax[i//2, i % 2].hist(
#             X, bins="fd", range=(0, 3000), label=label, density=True, log=True)
#         ax[i//2, i % 2].set_title(label)
#     fig.supxlabel("days")
#     fig.supylabel("probability")
#     plt.show()

# DURATION
# var = "duration"
# for proc, proc_id in zip(["framework", "direct", "negotiated", "open"],
#                          [26, 23, 4, 1]):
#     plt.figure()
#     X = data[data["id_award_procedure"] == proc_id][var]
#     plt.hist(
#         X, bins="fd", range=(0, 4000), density=True, log=True)
#     # ticks = list(x * 1e8 for x in range(0, 7))
#     # plt.xticks(ticks, [f"{str(int(x/1e6))}M" for x in ticks])
#     plt.xlabel("duration days")
#     plt.ylabel("probability")
#     # plt.savefig(OUTDIR + var + "_" + proc + ".png")

# # AMOUNT
# proc = "framework"
# X = data[data["id_award_procedure"] == 26]["amount"]
# plt.figure()
# plt.hist(
#     X, bins="fd", range=(0, 1e7), density=True, log=True)
# ticks = list(x * 1e6 for x in range(0, 11, 2))
# plt.xticks(ticks, [f"{str(int(x/1e6))}M€" for x in ticks])
# plt.xlabel("lot amount")
# plt.ylabel("probability")
# plt.savefig("histfigures/amount_" + proc +".png")


# proc = "direct"
# X = data[data["id_award_procedure"] == 23]["amount"]
# # print((X > 3e5).sum()) # 130
# # print((X > 1e7).sum()) # 3
# # print((X > 1e6).sum())  # 34
# plt.figure()
# plt.hist(
#     X, bins="fd", range=(0, 3e5), density=True, log=True)
# ticks = list(x * 1e3 for x in range(0, 301, 50))
# plt.xticks(ticks, [f"{str(int(x / 1e3))}K€" for x in ticks])
# plt.xlabel("lot amount")
# plt.ylabel("probability")
# plt.savefig("histfigures/amount_" + proc + ".png")

# proc = "negotiated"
# X = data[data["id_award_procedure"] == 4]["amount"]
# maxx = 4e6
# plt.figure()
# plt.hist(
#     X, bins="fd", range=(0, maxx), density=True, log=True)
# ticks = list(x * 1e6 for x in range(0, 5))
# plt.xticks(ticks, [f"{str(int(x/1e6))}M€" for x in ticks])
# plt.xlabel("lot amount")
# plt.ylabel("probability")
# plt.savefig("histfigures/amount_" + proc +".png")


# proc = "open"
# X = data[data["id_award_procedure"] == 1]["amount"]
# maxx = 1e7
# plt.figure()
# plt.hist(
#     X, bins="fd", range=(0, maxx), log=True, density=True)
# ticks = list(x * 1e6 for x in range(0, 11, 2))
# plt.xticks(ticks, [f"{str(x/1e6)}M€" for x in ticks])
# plt.xlabel("lot amount")
# plt.ylabel("probability")
# plt.savefig("histfigures/amount_" + proc +".png")


# BE MEDIAN REV
# var = "be_med_ann_revenue"
# X = data[data["id_award_procedure"] == 23][var]
# print((X > 3e7).sum(), X.shape) # 1179, 280842
# for proc, proc_id in zip(["framework", "direct", "negotiated", "open"],
#                          [26, 23, 4, 1]):
#     X = data[data["id_award_procedure"] == proc_id][var]
#     plt.figure()
#     plt.hist(
#         X, bins="fd", density=True, log=True)
#     ticks = list(x * 1e7 for x in range(0, 13, 2))
#     plt.xticks(ticks, [f"{str(int(x/1e6))}M€" for x in ticks])
#     plt.xlabel("BE median annual revenue")
#     plt.ylabel("probability")
#     plt.savefig("histfigures/" + var + "_" + proc + ".png")

# PA MEDIAN ANNUAL EXPENDITURE
# var = "pa_med_ann_expenditure"
# for proc, proc_id in zip(["framework", "direct", "negotiated", "open"],
#                          [26, 23, 4, 1]):
#     plt.figure()
#     X = data[data["id_award_procedure"] == proc_id][var]
#     plt.hist(
#         X, bins="fd", density=True, log=True)
#     ticks = list(x * 1e8 for x in range(0, 7))
#     plt.xticks(ticks, [f"{str(int(x/1e6))}M€" for x in ticks])
#     plt.xlabel("PA median annual expenditure")
#     plt.ylabel("probability")
#     plt.savefig("histfigures/" + var + "_" + proc + ".png")


##### CPV #####
var = "duration"
for cpv in [33, 45, 65, 85]:
    plt.figure()
    X = data[data["cpv"] == cpv][var]
    plt.hist(X, bins="fd", range=(0, 4000), density=True, log=True)
    # ticks = list(x * 1e8 for x in range(0, 7))
    # plt.xticks(ticks, [f"{str(int(x/1e6))}M" for x in ticks])
    plt.xlabel("duration days")
    plt.ylabel("probability")
    plt.savefig(OUTDIR + var + "_" + str(cpv) + ".png")

var = "amount"
for cpv in [33, 45, 65, 85]:
    plt.figure()
    X = data[data["cpv"] == cpv][var]
    maxsize = 1e7
    plt.hist(X, bins="fd", range=(0, maxsize), density=True, log=True)
    ticks = list(x * 1e6 for x in range(0, 11))
    plt.xticks(ticks, [f"{str(int(x/1e6))}M€" for x in ticks])
    plt.xlabel("amount")
    plt.ylabel("probability")
    plt.savefig(OUTDIR + var + "_" + str(cpv) + ".png")

var = "be_med_ann_revenue"
for cpv in [33, 45, 65, 85]:
    plt.figure()
    X = data[data["cpv"] == cpv][var]
    if cpv == 33:
        plt.hist(X, bins="fd", density=True, log=True)
        ticks = list(x * 1e7 for x in range(0, 13, 2))
        plt.xticks(ticks, [f"{str(int(x/1e6))}M€" for x in ticks])
    else:
        plt.hist(X, bins="fd", range=(0, 40e6), density=True, log=True)
        ticks = list(x * 1e7 for x in range(0, 5))
        plt.xticks(ticks, [f"{str(int(x/1e6))}M€" for x in ticks])
    plt.xlabel("BE median annual revenue")
    plt.ylabel("probability")
    plt.savefig(OUTDIR + var + "_" + str(cpv) + ".png")

var = "pa_med_ann_expenditure"
for cpv in [33, 45, 65, 85]:
    plt.figure()
    X = data[data["cpv"] == cpv][var]
    plt.hist(X, bins="fd", density=True, log=True)
    ticks = list(x * 1e8 for x in range(0, 6))
    plt.xticks(ticks, [f"{str(int(x/1e6))}M€" for x in ticks])
    plt.xlabel("PA median annual expenditure")
    plt.ylabel("probability")
    plt.savefig(OUTDIR + var + "_" + str(cpv) + ".png")
