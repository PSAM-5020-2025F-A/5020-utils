import numpy as np

def get_nonzero_scores(vectors, sorted_idxs, n_words):
  # iterate over vectors and list of indices
  # index each vector by sorted indexes
  # then slice to keep non-zero values
  # then slice to keep n_words
  nz_scores = [row[idx][:len(row.nonzero()[0])][:n_words] for row,idx in zip(vectors, sorted_idxs)]
  return np.array(nz_scores, dtype=object)

def get_top_words(vectors, vocab, n_words=None):
  is_row = vectors.shape[0] == 1

  if hasattr(vectors, "toarray"):
    vectors = vectors.toarray()

  sidxs = (-vectors).argsort()
  nz_scores = get_nonzero_scores(vectors, sidxs, n_words)
  # similar to scores... get sorted list of words by sorted indices
  # then slice up to length of corresponding score list
  nz_words = np.array([vocab[idx][:len(nzs)] for nzs,idx in zip(nz_scores, sidxs)], dtype=object)

  if is_row:
    nz_scores = nz_scores.squeeze()
    nz_words = nz_words.squeeze()

  return nz_words.tolist(), nz_scores.tolist()
