from os import listdir, makedirs, path, remove
from random import shuffle
from shutil import copy2, move, rmtree

# copy files from test/val to train
for s in ["test", "val"]:
  dirs = sorted([d for d in listdir(f"./birds/{s}") if path.isdir(f"./birds/{s}/{d}")])
  for d in dirs:
    fnames = sorted([f for f in listdir(f"./birds/{s}/{d}") if f.endswith("wav")])
    for f in fnames:
      fpath = f"./birds/{s}/{d}/{f}"
      copy2(fpath, f"./birds/train/{d}")

# rename directories
dirs = sorted([d for d in listdir(f"./birds/train") if path.isdir(f"./birds/train/{d}")])
for d in dirs:
  newd = f"./birds/{d.lower().split('_')[-1]}"
  if not path.isdir(newd):
    move(f"./birds/train/{d}", newd)

# remove unnecessary/temp files/dirs
for df in ["./birds/train", "./birds/test", "./birds/val", "./birds/background", "./birds/.git", "./birds/README.md", "./birds/split_metadata.csv", "./birds/test.zip"]:
  if path.isdir(df):
    rmtree(df)
  if path.isfile(df):
    remove(df)

# rename files
dirs = sorted([d for d in listdir(f"./birds") if path.isdir(f"./birds/{d}")])
for d in dirs:
  fnames = sorted([f for f in listdir(f"./birds/{d}") if f.endswith("wav")])
  for idx,f in list(enumerate(fnames)):
    str_idx = f"000{idx}"[-4:]
    fpath = f"./birds/{d}/{f}"
    move(fpath, f"./birds/{d}/{d}_{str_idx}.wav")

# prepare train/test for export
makedirs("./birds/train", exist_ok=True)
makedirs("./birds/test", exist_ok=True)

dirs = sorted([d for d in listdir(f"./birds") if path.isdir(f"./birds/{d}") and d not in ["train", "test"]])
for d in dirs:
  fnames = [f for f in listdir(f"./birds/{d}") if f.endswith("wav")]
  shuffle(fnames)
  for idx,f in enumerate(fnames):
    split = "train" if idx < (0.8 * len(fnames)) else "test"
    move(f"./birds/{d}/{f}", f"./birds/{split}/{f}")
  rmtree(f"./birds/{d}")
