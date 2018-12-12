import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

def idx_to_indices(idx):
    return [i for i, x in enumerate(idx) if x]

if len(sys.argv) < 3:
    print("Usage: filename column_name ... ")

save_fn = ""
if len(sys.argv) > 3:
    save_fn = sys.argv[3]

fn = sys.argv[1]
col_name = "param_" + sys.argv[2]

df = pd.read_csv(fn)
df = df.sort_values("mean_test_score")

top = min(len(df), 5)
# df = df.iloc[-top:]

ids = list(range(len(df)))
cell_text = []
params = list(filter(lambda col: col.startswith("param_"), df.columns))

assert col_name in params, "Invalid column name:" + col_name

barlist = plt.bar(ids, df.mean_test_score)

color_groups = []
# color on augmentation
for col_val in df[col_name].unique():
    group_idx = df[col_name] == col_val
    color_groups.append((col_val, idx_to_indices(group_idx)))

def rand_color():
    r = hex(np.random.randint(0x333333, 0xbbbbbb))
    return r.replace("0x", "#")

colors = [rand_color() for _ in range(len(color_groups))]
leg_handles = []
ci = 0
for param_val, ind_list in color_groups:
    c = colors[ci]
    for idx in ind_list:
        barlist[idx].set_color(c)

    patch = mpatches.Patch(color=c, label=param_val)
    leg_handles.append(patch)
    ci += 1
plt.legend(handles=leg_handles, title=sys.argv[2])

# plt.subplots_adjust(left=0.2, bottom=0.6)

ts = df.mean_test_score
bot = ts.min() - ts.std() / 10
top = ts.max() + ts.std() / 10
plt.ylim(bot, top)
plt.xticks(ids)

plt.xlabel("Model")
plt.ylabel("Validation accuracy")

# plt.show()
if save_fn:
    plt.savefig(save_fn)
