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
duration_extrame = pd.read_csv(
    input_path + "duration_extreme.csv", index_col="index")

df = pd.concat([df, rule1, rule2, rule3, be_amount_extreme,
               pa_amount_extreme, duration_extrame], axis=1)
# confusion_matrix(rule1, be_amount_extreme)  # C_ij = C_00, etc

aperta = df[df["id_award_procedure"] == 1]

print("rule1 vs amount")
print(confusion_matrix(aperta.rule1, aperta.be_amount_extreme))
print(confusion_matrix(aperta.rule1, aperta.pa_amount_extreme))

print("rule3 vs amount")
print(confusion_matrix(aperta.rule3, aperta.be_amount_extreme))
print(confusion_matrix(aperta.rule3, aperta.pa_amount_extreme))

# both of them are all False values
# print("rule3 vs duration")
# print(confusion_matrix(aperta.rule2, aperta.duration_extreme))
