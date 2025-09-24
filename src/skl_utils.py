import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans as SklKMeans, SpectralClustering as SklSpectralClustering
from sklearn.decomposition import PCA as SklPCA
from sklearn.ensemble import RandomForestClassifier as SklRandomForestClassifier
from sklearn.linear_model import LinearRegression as SklLinearRegression
from sklearn.linear_model import LogisticRegression as SklLogisticRegression
from sklearn.linear_model import SGDRegressor as SklSGDRegressor, SGDClassifier as SklSGDClassifier
from sklearn.manifold import TSNE as SklTSNE
from sklearn.metrics import silhouette_score as skl_silhouette_score, silhouette_samples as skl_silhouette_samples
from sklearn.mixture import GaussianMixture as SklGaussianMixture
from sklearn.preprocessing import MinMaxScaler as SklMinMaxScaler
from sklearn.preprocessing import StandardScaler as SklStandardScaler
from sklearn.preprocessing import PolynomialFeatures as SklPolynomialFeatures
from sklearn.svm import SVC as SklSVC, LinearSVC as SklLSVC


def toDataFrame(X):
  if isinstance(X, pd.core.frame.DataFrame) or isinstance(X, pd.core.series.Series):
    return X
  try:
    X = pd.DataFrame(X)
  except:
    raise Exception("Wrong type. Please use arrays, numpy arrays, pandas DataFrames or pandas Series.")


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


class PolynomialFeatures(SklPolynomialFeatures):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def fit_transform(self, X, *args, **kwargs):
    X = toDataFrame(X)

    self.columns = X.columns
    self.shape = X.shape
    X_t = super().fit_transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=self.get_feature_names_out())

  def transform(self, X, *args, **kwargs):
    X = toDataFrame(X)

    if list(self.columns) != list(X.columns) or self.shape[1] != X.shape[1]:
      raise Exception("Input has wrong shape.")

    X_t = super().transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=self.get_feature_names_out())


class Predictor():
  def __init__(self, type, **kwargs):
    if type == "linear":
      self.model = SklLinearRegression(**kwargs)
    elif type == "logistic":
      self.model = SklLogisticRegression(**kwargs)
    elif type == "sgdr":
      self.model = SklSGDRegressor(**kwargs)
    elif type == "sgdc":
      self.model = SklSGDClassifier(**kwargs)
    elif type == "forest":
      if "max_depth" not in kwargs:
        kwargs["max_depth"] = 16
      self.model = SklRandomForestClassifier(**kwargs)
    elif type == "svc":
      self.model = SklSVC(**kwargs)
    elif type == "lsvc":
      self.model = SklLSVC(**kwargs)

  def __getattr__(self, name):
    return getattr(self.model, name)

  def fit(self, X, y, *args, **kwargs):
    X = toDataFrame(X)
    y = toDataFrame(y)

    self.y_name = [y.name] if len(y.shape) == 1 else y.columns.values
    self.model.fit(X.values, y.values, *args, **kwargs)
    return self

  def predict(self, X, *args, **kwargs):
    X = toDataFrame(X)
    y_t = self.model.predict(X.values, *args, **kwargs)
    return pd.DataFrame(y_t, columns=self.y_name)


class Scaler():
  def __init__(self, type, **kwargs):
    if type == "minmax":
      self.scaler = SklMinMaxScaler(**kwargs)
    elif type == "std":
      self.scaler = SklStandardScaler(**kwargs)

  def __getattr__(self, name):
    return getattr(self.scaler, name)

  def fit_transform(self, X, *args, **kwargs):
    X = toDataFrame(X)

    self.columns = X.columns
    self.shape = X.shape
    X_t = self.scaler.fit_transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=X.columns)

  def transform(self, X, *args, **kwargs):
    X = toDataFrame(X)

    if list(self.columns) != list(X.columns) or self.shape[1] != X.shape[1]:
      raise Exception("Input has wrong shape.")

    X_t = self.scaler.transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=X.columns)

  def inverse_transform(self, X, *args, **kwargs):
    X = toDataFrame(X)

    col = ""
    col_vals = []

    if len(X.shape) == 1:
      col = X.name
      col_vals = X.values
    elif len(X.shape) == 2 and X.shape[1] == 1:
      col = X.columns[0]
      col_vals = X[col].values

    if col != "":
      X = pd.DataFrame(X.values, columns=[col])
      dummy_df = pd.DataFrame(np.zeros((len(col_vals), self.shape[1])), columns=self.columns)
      dummy_df[col] = col_vals
      X_t = self.scaler.inverse_transform(dummy_df.values, *args, **kwargs)
      return pd.DataFrame(X_t, columns=self.columns)[[col]]

    else:
      X_t = self.scaler.inverse_transform(X.values, *args, **kwargs)
      return pd.DataFrame(X_t, columns=X.columns)


class Clusterer():
  def __init__(self, type, **kwargs):
    self.num_clusters = 0
    kwargs["n_init"] = 10
    if type == "kmeans":
      self.model = SklKMeans(**kwargs)
    elif type == "gaussian":
      if "n_clusters" in kwargs:
        kwargs["n_components"] = kwargs["n_clusters"]
        del kwargs["n_clusters"]
      self.model = SklGaussianMixture(**kwargs)
    elif type == "spectral":
      if "affinity" not in kwargs:
        kwargs["affinity"] = 'nearest_neighbors'
      if "n_clusters" in kwargs:
        kwargs["n_clusters"] += 0
      self.model = SklSpectralClustering(**kwargs)

  def __getattr__(self, name):
    return getattr(self.model, name)

  def fit_predict(self, X, *args, **kwargs):
    X = toDataFrame(X)
    y = self.model.fit_predict(X.values, *args, **kwargs)
    self.X = X.values
    self.y = y
    self.num_clusters = len(np.unique(y))
    self.num_features = self.X.shape[1]
    self.cluster_centers_ = np.array([self.X[self.y == c].mean(axis=0) for c in range(self.num_clusters)]).tolist()
    return pd.DataFrame(y, columns=["clusters"])

  def distance_score(self):
    if self.num_clusters < 1:
      raise Exception("Error: need to run fit_predict() first")

    point_centers = [self.cluster_centers_[i] for i in self.y]
    point_diffs = np.array([p - c for p,c in zip(self.X, point_centers)])

    cluster_L2 = [np.sqrt(np.square(point_diffs[self.y == c]).sum(axis=1)).mean() for c in range(self.num_clusters)]

    return sum(cluster_L2) / len(cluster_L2)

  def balance_score(self):
    if self.num_clusters < 1:
      raise Exception("Error: need to run fit_predict() first")
    counts = np.unique(self.y, return_counts=True)[1]
    sum_dists = np.abs(counts / len(self.y) - (1 / self.num_clusters)).sum()
    scale_factor = 0.5 * self.num_clusters / (self.num_clusters - 1)
    return 1.0 - (scale_factor * sum_dists)

  def silhouette_score(self):
    if self.num_clusters < 1:
      raise Exception("Error: need to run fit_predict() first")
    return skl_silhouette_score(self.X, self.y)

  def plot_silhouette(self):
    display_silhouette_plots(self.X, self.y)


class Reducer():
  def __init__(self, type, **kwargs):
    self.t_labels = []
    self.o_labels = None
    if type == "pca":
      self.reducer = SklPCA(**kwargs)
      self.col_pre = "PC"
    elif type == "tsne":
      self.reducer = SklTSNE(**kwargs)
      self.col_pre = "TSNE"
    self.n_components = self.reducer.n_components

  def __getattr__(self, name):
    return getattr(self.reducer, name)

  def check_input(self, X):
    X = toDataFrame(X)
    self.o_labels = X.columns
    X = X.values.tolist()

    if not isinstance(X, list):
      raise Exception("Input has wrong type. Please use list of list of pixels")
    if not isinstance(X[0], list):
      raise Exception("Input has wrong type. Please use list of list of pixels")
    return X

  def has_fitted(self):
    return len(self.t_labels) == self.n_components or len(self.t_labels) == self.reducer.n_components_

  def fit(self, X, *args, **kwargs):
    X = self.check_input(X)
    X_np = np.array(X)
    self.reducer.fit(X_np, *args, **kwargs)

    self.n_components = self.n_components if type(self.n_components) == int else self.reducer.n_components_
    self.t_labels = [f"{self.col_pre}{i}" for i in range(self.n_components)]

  def transform(self, X, *args, **kwargs):
    if not self.has_fitted():
      raise Exception("Error: need to run fit() first")
    X = self.check_input(X)
    X_np = np.array(X)
    if hasattr(self.reducer, "transform"):
      X_t = self.reducer.transform(X_np, *args, **kwargs)
      self.components = self.reducer.components_
    else:
      X_t = self.reducer.fit_transform(X_np, *args, **kwargs)
    X_obj = [{f"{self.col_pre}{i}": v for i,v in enumerate(x)} for x in X_t]
    return pd.DataFrame.from_records(X_obj)

  def fit_transform(self, X, *args, **kwargs):
    self.fit(X, *args, **kwargs)
    return self.transform(X, *args, **kwargs)

  def inverse_transform(self, X_t, *args, **kwargs):
    if not self.has_fitted():
      raise Exception("Error: need to run fit() first")

    X_t = toDataFrame(X_t)
    X_t_np = X_t[self.t_labels].values

    if X_t_np.shape[1] != self.n_components:
      raise Exception("Input has wrong shape. Check number of features")

    X_i_np = self.reducer.inverse_transform(X_t_np, *args, **kwargs)
    return pd.DataFrame(X_i_np, columns=self.o_labels)

  def explained_variance(self):
    if not self.has_fitted():
      raise Exception("Error: need to run fit() first")
    return sum(self.reducer.explained_variance_ratio_)


class LinearRegression(Predictor):
  def __init__(self, **kwargs):
    super().__init__("linear", **kwargs)

class LogisticRegression(Predictor):
  def __init__(self, **kwargs):
    super().__init__("logistic", **kwargs)

class SGDRegressor(Predictor):
  def __init__(self, **kwargs):
    super().__init__("sgdr", **kwargs)

class SGDClassifier(Predictor):
  def __init__(self, **kwargs):
    super().__init__("sgdc", **kwargs)

class RandomForestClassifier(Predictor):
  def __init__(self, **kwargs):
    super().__init__("forest", **kwargs)

class SVC(Predictor):
  def __init__(self, **kwargs):
    super().__init__("svc", **kwargs)

class LinearSVC(Predictor):
  def __init__(self, **kwargs):
    super().__init__("lsvc", **kwargs)

class MinMaxScaler(Scaler):
  def __init__(self, **kwargs):
    super().__init__("minmax", **kwargs)

class StandardScaler(Scaler):
  def __init__(self, **kwargs):
    super().__init__("std", **kwargs)

class KMeansClustering(Clusterer):
  def __init__(self, **kwargs):
    super().__init__("kmeans", **kwargs)

class GaussianClustering(Clusterer):
  def __init__(self, **kwargs):
    super().__init__("gaussian", **kwargs)

class SpectralClustering(Clusterer):
  def __init__(self, **kwargs):
    super().__init__("spectral", **kwargs)

class PCA(Reducer):
  def __init__(self, **kwargs):
    super().__init__("pca", **kwargs)

class TSNE(Reducer):
  def __init__(self, **kwargs):
    super().__init__("tsne", **kwargs)
