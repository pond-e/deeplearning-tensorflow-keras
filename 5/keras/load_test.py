import csv
import numpy as np

with open('./press_log_100Hz/press_logs_20220309-102401_2_checked.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]
print(len(l))
x = len(l)
f = [k[1:] for k in l[1:]]
for i in range(len(f)):
    for j in range(len(f[0])):
        f[i][j] = float(f[i][j])
#f = np.array(f)
print(f[:10])
