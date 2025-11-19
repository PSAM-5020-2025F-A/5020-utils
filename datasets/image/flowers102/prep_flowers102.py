import json

from os import makedirs
from random import seed, sample, shuffle
from shutil import copy2

TRAIN_CNT = 4
MAX_CNT = 64

train_files = []
test_files = []

makedirs("./train", exist_ok=True)
makedirs("./test", exist_ok=True)

# Read label helper
with open("flowers102_labels.json", "r") as ifp:
  file2label = json.load(ifp)

# Group by label
flower_info = {}
for fname,label in file2label.items():
  if label not in flower_info: flower_info[label] = []
  flower_info[label].append({
    "file_name": fname,
    "label": label
  })

# Sample from groups
seed(1010)
for flist in flower_info.values():
  sampled = sample(flist, k=len(flist))
  train_files += sampled[:TRAIN_CNT]
  test_files += sampled[TRAIN_CNT:MAX_CNT]

shuffle(train_files)
shuffle(test_files)

# Copy files
for idx,finfo in enumerate(train_files):
  src_path = f"./jpg/{finfo['file_name']}"
  dst_idx = f"00000{idx}"[-5:]
  dst_path = f"./train/train_{dst_idx}_{finfo['label']}.jpg"
  copy2(src_path, dst_path)

for idx,finfo in enumerate(test_files):
  src_path = f"./jpg/{finfo['file_name']}"
  dst_idx = f"00000{idx}"[-5:]
  dst_path = f"./test/test_{dst_idx}_{finfo['label']}.jpg"
  copy2(src_path, dst_path)
