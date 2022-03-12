import csv

with open('./press_log_100Hz/press_logs_20220309-102401_2_checked.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]
print(len(l))
