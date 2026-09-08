"""Gene filtering strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd


def filter_by_variance(expression, top_k=2000):
    """Keep the top-K genes by variance. 0 or >= total genes keeps all."""
    return expression.loc[:, high_variance_genes(expression, top_k)]


def high_variance_genes(expression, top_k=2000):
    """Return gene names selected by variance, in descending variance order."""
    if top_k <= 0 or top_k >= expression.shape[1]:
        return expression.columns
    gene_var = expression.var(axis=0)
    return gene_var.nlargest(top_k).index


def filter_by_mean_expression(expression, min_mean=1.0):
    """Remove genes with mean expression below min_mean."""
    gene_means = expression.mean(axis=0)
    keep = gene_means >= min_mean
    return expression.loc[:, keep]


def filter_by_nonzero_fraction(expression, min_fraction=0.1):
    """Remove genes expressed in fewer than min_fraction of samples."""
    return expression.loc[:, nonzero_genes(expression, min_fraction)]


def nonzero_genes(expression, min_fraction=0.1):
    """Return genes expressed in at least the requested fraction of samples."""
    nonzero_frac = (expression > 0).mean(axis=0)
    return expression.columns[nonzero_frac >= min_fraction]
