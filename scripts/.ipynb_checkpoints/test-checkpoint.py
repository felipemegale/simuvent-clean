import json
import sys

with open("collected_data/2022-06-22T11:31:07.430160.json", "r") as f1:
    d1 = f1.read()

with open("collected_data/2022-06-22T11:32:35.268752.json", "r") as f2:
    d2 = f2.read()

j1 = json.loads(d1)
j2 = json.loads(d2)

j1 = j1[0]["flowSegmentData"]
j2 = j2[0]["flowSegmentData"]

c1 = j1["coordinates"]["coordinate"]
c2 = j2["coordinates"]["coordinate"]

if len(c1) != len(c2):
    print("different lengths")
    sys.exit()

for i in range(len(c1)):
    if c1[i] != c2[i]:
        print("different coordinates")
        sys.exit()

print("coordinates are equal")
