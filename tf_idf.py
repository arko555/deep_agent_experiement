# """TF-IDF from scratch (no sklearn) + window attention in PyTorch.

# Usage:
#     python tf_idf.py             # runs both demos
# or import:
#     from tf_idf import TfidfVectorizer, window_attention
# """

# import math
# import re
# from collections import Counter

# try:
#     import torch
# except ImportError:  # keep TF-IDF usable without PyTorch
#     torch = None


# class TfidfVectorizer:
#     """Vectorize a corpus of raw text documents into TF-IDF weight maps."""

#     def __init__(self, use_idf=True, norm="l2", sublinear_tf=False):
#         # norm: "l2" | "l1" | None
#         self.use_idf = use_idf
#         self.norm = norm
#         self.sublinear_tf = sublinear_tf
#         self.vocabulary_ = {}  # term -> column index
#         self.idf_ = {}         # term -> idf value

#     def fit(self, documents):
#         """documents: list of raw strings. Builds vocabulary + IDF."""
#         df = Counter()
#         for doc in documents:
#             for term in set(self._tokenize(doc)):
#                 df[term] += 1
#         n_docs = max(len(documents), 1)
#         self.vocabulary_ = {term: i for i, term in enumerate(sorted(df))}
#         # Smoothed IDF (sklearn convention): log((1 + N - df + 0.5) / (df + 0.5)) + 1
#         self.idf_ = {
#             term: math.log((1 + n_docs - df[term] + 0.5) / (df[term] + 0.5)) + 1.0
#             for term in df
#         }
#         return self

#     def transform(self, documents):
#         """Returns one {term: tfidf_weight} dict per document."""
#         vectors = []
#         for doc in documents:
#             counts = Counter(self._tokenize(doc))
#             total = sum(counts.values()) or 1
#             weights = {}
#             for term, count in counts.items():
#                 tf = math.log(1 + count) if self.sublinear_tf else count / total
#                 weights[term] = tf * self.idf_[term] if self.use_idf else tf
#             if self.norm:
#                 weights = self._normalize(weights)
#             vectors.append(weights)
#         return vectors

#     def fit_transform(self, documents):
#         return self.fit(documents).transform(documents)

#     @staticmethod
#     def _tokenize(text):
#         return re.findall(r"[a-z0-9]+", text.lower())

#     @classmethod
#     def _normalize(cls, weights, norm="l2"):
#         if not weights:
#             return weights
#         denom = (
#             sum(abs(w) for w in weights.values()) if norm == "l1"
#             else math.sqrt(sum(w * w for w in weights.values()))
#         )
#         return {t: w / denom for t, w in weights.items()}


# def window_attention(query, key, value, window_size=3):
#     """Local/sparse attention with a symmetric sliding window.

#     Each query position i attends only to keys j with |i - j| <= window_size,
#     clipped to the sequence bounds. Outside the window, positions are masked
#     to -inf before softmax.

#     Args:
#         query, key, value: tensors of shape (batch, seq_len, dim).
#         window_size: half-width w of the window. Global attention is
#             recovered with window_size >= seq_len.

#     Returns:
#         Tensor of shape (batch, seq_len, dim).
#     """
#     if torch is None:
#         raise ImportError("window_attention requires PyTorch (pip install torch)")
#     if window_size < 0:
#         raise ValueError(f"window_size must be >= 0, got {window_size}")

#     seq_len = query.size(1)
#     offsets = torch.arange(seq_len).to(query.device)
#     mask = (offsets[:, None] - offsets[None, :]).abs() <= window_size  # (L, L) bool

#     dim = query.size(-1)
#     scores = torch.bmm(query, key.transpose(1, 2)) * (dim ** -0.5)  # (B, L, L)
#     scores = scores.masked_fill(~mask[None], float("-inf"))

#     attn = torch.softmax(scores, dim=-1)
#     return torch.bmm(attn, value)


# if __name__ == "__main__":
#     corpus = [
#         "the cat sat on the mat",
#         "the dog sat on the log",
#         "cats and dogs make great pets",
#         "a cat is a wonderful pet",
#     ]

#     vec = TfidfVectorizer().fit_transform(corpus)
#     for doc, weights in zip(corpus, vec):
#         top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:4]
#         print(f"{doc!r}")
#         for term, w in top:
#             print(f"  {term}: {w:.4f}")
#     print()

#     # Cosine similarity between doc 0 and the others
#     def cosine(a, b):
#         shared = a.keys() & b.keys()
#         num = sum(a[t] * b[t] for t in shared)
#         den = math.sqrt(sum(v * v for v in a.values())) * \
#               math.sqrt(sum(v * v for v in b.values()))
#         return num / den if den else 0.0

#     for i in (1, 2, 3):
#         print(f"cosine(doc0, doc{i}) = {cosine(vec[0], vec[i]):.4f}")

#     # --- window attention demo (skipped if PyTorch not installed) ---
#     if torch is None:
#         print("\n(PyTorch not installed — skipping window attention demo)")
#     else:
#         torch.manual_seed(0)
#         batch, seq_len, dim = 2, 6, 4
#         q = torch.randn(batch, seq_len, dim)
#         k = torch.randn(batch, seq_len, dim)
#         v = torch.randn(batch, seq_len, dim)

#         out = window_attention(q, k, v, window_size=1)
#         print(f"\nwindow attention output shape: {tuple(out.shape)}")  # expect (2, 6, 4)

#         def full_attention(query, key, value):
#             dim = query.size(-1)
#             scores = torch.bmm(query, key.transpose(1, 2)) * (dim ** -0.5)
#             return torch.bmm(torch.softmax(scores, dim=-1), value)

#         # window_size >= seq_len must match global attention exactly
#         wide = window_attention(q, k, v, window_size=seq_len)
#         print(f"max |wide-window - full|: {(wide - full_attention(q, k, v)).abs().max().item():.2e}")

#         local0 = window_attention(q, k, v, window_size=0)
#         print(f"zero-window output finite: {torch.isfinite(local0).all().item()}")


import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B:ovhcloud",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in one sentence."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                    }
                }
            ]
        }
    ],
)

print(completion.choices[0].message)