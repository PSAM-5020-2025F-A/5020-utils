import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as PImage
import string
import urllib.request as request

from os import listdir, path
from random import seed, shuffle

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, root_mean_squared_error
from sklearn.metrics import silhouette_score as skl_silhouette_score, silhouette_samples as skl_silhouette_samples

from warnings import simplefilter


def object_from_json_url(url):
  with request.urlopen(url) as in_file:
    return json.load(in_file)


def toDataFrame(X):
  if isinstance(X, pd.core.frame.DataFrame) or isinstance(X, pd.core.series.Series):
    return X
  try:
    return pd.DataFrame(X)
  except:
    raise Exception("Wrong type. Please use arrays, numpy arrays, pandas DataFrames or pandas Series.")


def regression_error(labels, predicted):
  labels_np = toDataFrame(labels).values
  predicted_np = toDataFrame(predicted).values
  return root_mean_squared_error(labels_np, predicted_np)

def classification_error(labels, predicted):
  labels_np = toDataFrame(labels).values
  predicted_np = toDataFrame(predicted).values
  return 1.0 - accuracy_score(labels_np, predicted_np)

def accuracy_score_topk(labels, scores, k=1):
  labels_np = toDataFrame(labels).values
  sorted_idxs = scores.argsort(axis=1)
  pred_k = sorted_idxs[:, -k:]
  l_in_ps = [int(l in p) for l,p in zip(labels_np, pred_k)]
  return sum(l_in_ps) / len(l_in_ps)

def classification_error_topk(labels, scores, k=1):
  return 1.0 - accuracy_score_topk(labels, scores, k)


def distance_score(X, y):
  X_np = toDataFrame(X).values
  y_np = toDataFrame(y).values.reshape(-1)
  num_clusters = len(np.unique(y_np))
  cluster_centers = np.array([X_np[y_np == c].mean(axis=0) for c in range(num_clusters)])
  point_centers = [cluster_centers[i] for i in y_np]
  point_diffs = np.array([p - c for p,c in zip(X_np, point_centers)])
  cluster_L2 = [np.sqrt(np.square(point_diffs[y_np == c]).sum(axis=1)).mean() for c in range(num_clusters)]
  return sum(cluster_L2) / len(cluster_L2)

def balance_score(y):
  y_np = toDataFrame(y).values.reshape(-1)
  cluster_ids, counts = np.unique(y_np, return_counts=True)
  num_clusters = len(cluster_ids)
  sum_dists = np.abs(counts / len(y_np) - (1 / num_clusters)).sum()
  scale_factor = 0.5 * num_clusters / (num_clusters - 1)
  return 1.0 - (scale_factor * sum_dists)

def silhouette_score(X, y):
  X_np = toDataFrame(X).values
  y_np = toDataFrame(y).values.reshape(-1)
  return skl_silhouette_score(X_np, y_np)


def display_confusion_matrix(labels, predicted, display_labels):
  simplefilter(action='ignore', category=FutureWarning)
  ConfusionMatrixDisplay.from_predictions(labels, predicted, display_labels=display_labels, xticks_rotation="vertical")

def display_silhouette_plots(X, y):
  sample_silhouette_values = skl_silhouette_samples(X, y)
  silhouette_average = skl_silhouette_score(X, y)
  num_clusters = len(np.unique(y))
  maxx = round(sample_silhouette_values.max() / 0.2) * 0.2

  y_lower = 10
  for i in range(num_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[y == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / num_clusters)
    plt.fill_betweenx(
      np.arange(y_lower, y_upper),
      0,
      ith_cluster_silhouette_values,
      facecolor=color,
      edgecolor=color,
      alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    plt.text(-maxx / 10, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10

  plt.title("Silhouette Plot")
  plt.xlabel("Silhouette coefficient values")
  plt.ylabel("Cluster label")

  # The vertical line for average silhouette score of all the values
  plt.axvline(x=silhouette_average, color="red", linestyle="--")

  plt.yticks([])
  plt.xlim([min(-0.1, sample_silhouette_values.min()), sample_silhouette_values.max()])
  plt.xticks([-0.1] + list(np.arange(0, maxx+0.1, 0.2)))
  plt.show()


class LFWUtils:
  @classmethod
  def init(cls):
    IMAGE_DIRS = []
    if path.isdir(cls.DIR):
      IMAGE_DIRS = sorted([d for d in listdir(cls.DIR) if path.isdir(path.join(cls.DIR, d))])

    cls.LABELS = [d.split("-")[0] for d in IMAGE_DIRS if d[0] in string.ascii_letters]
    cls.L2I = {v:i for i,v in enumerate(cls.LABELS)}

    if len(IMAGE_DIRS) > 0:
      IMAGE_DIRS_PATH = path.join(cls.DIR, IMAGE_DIRS[0])
      _first_img = [f for f in listdir(IMAGE_DIRS_PATH) if f.endswith(".jpeg") or f.endswith(".jpg")][0]
      cls.IMAGE_SIZE = PImage.open(path.join(IMAGE_DIRS_PATH, _first_img)).size
      cls.WIDTH, cls.HEIGHT = cls.IMAGE_SIZE

  @classmethod
  def train_test_split(cls, dir, test_pct=0.5, random_state=101010):
    # cls.DIR = "./data/image/lfw/cropped"
    cls.DIR = dir
    cls.init()
    seed(random_state)
    dataset = { k : { "pixels": [], "labels": [], "files": [] } for k in ["test", "train"] }
    label_files = { k : [] for k in dataset.keys() }

    for label in cls.LABELS:
      label_path = path.join(cls.DIR, label)
      label_files_all = [f for f in listdir(label_path) if f.endswith(".jpeg") or f.endswith(".jpg")]
      shuffle(label_files_all)
      split_idx = int(test_pct * len(label_files_all))
      label_files["test"] = label_files_all[:split_idx]
      label_files["train"] = label_files_all[split_idx:]

      for split in dataset.keys():
        for f in label_files[split]:
          img = PImage.open(path.join(label_path, f))
          img.pixels = list(img.getdata())

          pixel = img.pixels[0]
          if (type(pixel) == list or type(pixel) == tuple) and len(pixel) > 2:
            img.pixels = [sum(l[:3]) / 3 for l in img.pixels]

          dataset[split]["pixels"].append(img.pixels)
          dataset[split]["labels"].append(cls.L2I[label])
          dataset[split]["files"].append(f)
          cls.IMAGE_SIZE = img.size
          cls.IMAGE_WIDTH = img.size[0]
          cls.IMAGE_HEIGHT = img.size[1]

    return dataset["train"], dataset["test"]

  @classmethod
  def top_precision(cls, labels, predicted, top=5):
    labels_np = np.array(cls.LABELS)
    cm = confusion_matrix(labels, predicted)
    precision_sum = np.sum(cm, axis=0)
    precision = [c/t if t != 0 else 0 for c,t in zip(np.diagonal(cm), precision_sum)]
    top_idx = np.argsort(precision)
    top_precision = list(reversed(labels_np[top_idx].tolist()))
    return top_precision[:top]

  @classmethod
  def top_recall(cls, labels, predicted, top=5):
    labels_np = np.array(cls.LABELS)
    cm = confusion_matrix(labels, predicted)
    recall_sum = np.sum(cm, axis=1)
    recall = [c/t if t != 0 else 0 for c,t in zip(np.diagonal(cm), recall_sum)]
    top_idx = np.argsort(recall)
    top_recall = list(reversed(labels_np[top_idx].tolist()))
    return top_recall[:top]
