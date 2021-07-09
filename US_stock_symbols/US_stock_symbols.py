import re
import csv

with open("美股.html", 'r') as f:
    pattern = re.compile(r"\(([A-Z]*?)\)<", re.S)
    result = re.findall(pattern, "".join(f.readlines()))
print(result)

with open("美股.csv", 'w', newline='') as f:
    for row in result:
        csv.writer(f).writerow([row])
