import json, re
from collections import Counter

log = json.load(open("baseline_log.json"))
bad = [x for x in log if not x["ok"]]

def kind(x):
    if "Answer:" not in x["out"]:
        return "no 'Answer:' line"
    nums = re.findall(r"-?\d+", x["out"].split("Answer:")[-1])
    return "wrong number" if nums else "no number after Answer"

by_op = {"+": [0, 0], "-": [0, 0]}
for x in log:
    op = "+" if "+" in x["q"] else "-"
    by_op[op][1] += 1
    by_op[op][0] += x["ok"]

print("Failure types:", dict(Counter(kind(x) for x in bad)))
print("Correct/total by op:", by_op)
for x in bad[:8]:
    print("\n----", x["q"], "| truth:", x["truth"])
    print(x["out"][:300])