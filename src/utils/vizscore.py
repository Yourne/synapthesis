import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


fname = "output/171122-175326_aperta_oc-svm/aperta_oc-svm.csv"
df = pd.read_csv(fname)
data = df.sample(1000, random_state=42)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(projection='3d')


x = np.log10(data["amount"])
y = np.log10(data["be_med_ann_revenue"])
z = np.log10(data["pa_med_ann_expenditure"])
score = data["score"]
p = ax.scatter(x, y, z, c=score, alpha=.5)
cb = fig.colorbar(p)
cb.set_label("log-probabily")
ax.set_xlabel("log10(lot amount) €")
ax.set_zlabel("log10(pa median annnual expenditure) €")
ax.set_ylabel("lgo10(be median annual revenue) €")

plt.show()
