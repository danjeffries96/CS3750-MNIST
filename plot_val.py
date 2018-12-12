import re
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

def idx_to_indices(idx):
    return [i for i, x in enumerate(idx) if x]

if len(sys.argv) < 3:
    print("Usage: outfile file1 file2 ... ")
    exit()

save_fn = sys.argv[1]

files = sys.argv[2:]

names = ["with augmentation", "no augmentation"]
ni = 0

for fn in files:
    name = fn.split("/")[-1].split(".")[0]
    name = names[ni]
    ni += 1

    df = pd.read_csv(fn)

    plt.plot(df.Epoch, df.Validation, ".-", label=name)

def rand_color():
    r = hex(np.random.randint(0x333333, 0xbbbbbb))
    return r.replace("0x", "#")

plt.legend()

# plt.subplots_adjust(left=0.2, bottom=0.6)

# plt.xticks(ids)

plt.xlabel("Iterations")
plt.ylabel("Validation accuracy")
if save_fn:
    plt.savefig(save_fn)
