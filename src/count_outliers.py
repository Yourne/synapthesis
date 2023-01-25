#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 18:09:50 2023

@author: nepal
"""
from sklearn.metrics import confusion_matrix
import pandas as pd

input_path = "../data10/"
df = pd.read_csv(input_path + "contracts.csv", index_col="index")
# load outliers baseline model
rule1 = pd.read_csv(input_path + "rule1.csv",
                    index_col="index")  # amount feature
rule2 = pd.read_csv(input_path + "rule2.csv",
                    index_col="index")  # duration feature
rule3 = pd.read_csv(input_path + "rule3.csv",
                    index_col="index")  # amount feature
# load outliers extreme value model
be_amount_extreme = pd.read_csv(
    input_path + "be_amount_extreme.csv", index_col="index")
pa_amount_extreme = pd.read_csv(
    input_path + "pa_amount_extreme.csv", index_col="index")
be_duration_extreme = pd.read_csv(
    input_path + "be_duration_extreme.csv", index_col="index")
pa_duration_extreme = pd.read_csv(
    input_path + "pa_duration_extreme.csv", index_col="index")

df = pd.concat([df, rule1, rule2, rule3, be_amount_extreme,
               pa_amount_extreme, be_duration_extreme, pa_duration_extreme], axis=1)
# confusion_matrix(rule1, be_amount_extreme)  # C_ij = C_00, etc

aperta = df[df["id_award_procedure"] == 1]

rule_amount = df.rule1 | df.rule3
amount_extreme = df.be_amount_extreme | df.pa_amount_extreme
duration_extreme = df.be_duration_extreme | df.be_amount_extreme


# print(sum(rule_amount))
# print(sum(amount_extreme))
# print(sum(duration_extreme))

# print(confusion_matrix(rule_amount, amount_extreme))
# print(confusion_matrix(rule2, duration_extreme))

# print(sum(df.be_amount_extreme), sum(df.be_duration_extreme))
# print(sum(df.pa_amount_extreme), sum(df.pa_duration_extreme))


print(sum(aperta.rule1 | aperta.rule3))
print(sum(aperta.be_amount_extreme | aperta.pa_amount_extreme))
print(sum(aperta.rule2))
print(sum(aperta.be_duration_extreme | aperta.pa_duration_extreme))
# print(sum(aperta.be_amount_extreme |
#       aperta.pa_amount_extreme | aperta.rule1 | aperta.rule3))
print(sum(aperta.rule2 | aperta.be_duration_extreme | aperta.pa_duration_extreme))

rule_amount = rule_amount.replace({True: -1, False: 1}).rename("rule_amount")
rule_duration = rule2.rule2.replace(
    {True: -1, False: 1}).rename("rule_duration")
amount_extreme = amount_extreme.replace(
    {True: -1, False: 1}).rename("extreme_amount")
duration_extreme = duration_extreme.replace(
    {True: -1, False: 1}).rename("extreme_duration")

rule_amount.to_csv(input_path+"rule_amount.csv", index_label=False)
rule_duration.to_csv(input_path+"rule_duration.csv", index_label=False)
amount_extreme.to_csv(input_path+"extreme_amount.csv", index_label=False)
duration_extreme.to_csv(input_path+"extreme_duration.csv", index_label=False)
