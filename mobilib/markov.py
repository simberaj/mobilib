"""Markov chain modeling code, with focus on mean first passage times (MFPT)."""

from typing import Union, Optional

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg


def mfpt(trans: Union[np.ndarray, scipy.sparse.csr_matrix],
         stat: Optional[np.ndarray] = None,
         ) -> np.ndarray:
    """Compute Markov mean first passage times.

    :param trans: A Markov transition matrix.
    :param stat: A stationary distribution of the matrix. If not given or None,
        it is computed from the transition matrix, but providing it saves time.
    """
    if scipy.sparse.issparse(trans):
        trans = np.asarray(trans.todense())
    if not trans.size:
        return trans
    if stat is None:
        stat = stationary_distribution(trans)
    else:
        stat = stat.copy()
    stat[stat == 0] = 0.25 * stat[stat != 0].min()
    stat /= stat.sum()
    fund = fundamental_matrix(trans, stat)
    return (
        np.eye(stat.size)
        - fund
        + np.ones_like(fund).dot(np.diag(np.diag(fund)))
    ).dot(np.diag(1 / stat))


def stationary_distribution(p: np.ndarray) -> np.ndarray:
    """Compute a Markov chain stationary distribution from a transition matrix."""
    if scipy.sparse.issparse(p):
        eigval, eigvec = scipy.sparse.linalg.eigs(p.T, k=6, sigma=1)
    else:
        eigval, eigvec = np.linalg.eig(p.T)
    vec_stat = eigvec[:, np.argmin(abs(eigval - 1.0))].real.flatten()
    return vec_stat / vec_stat.sum()


def transition_matrix(inter: Union[np.ndarray, scipy.sparse.csr_matrix]) -> np.ndarray:
    """Create a Markov transition matrix from an interaction matrix by normalization."""
    if scipy.sparse.issparse(inter):
        return scipy.sparse.diags(1 / inter.sum(axis=1).A.ravel()).dot(inter)
    else:
        return inter / inter.sum(axis=1, keepdims=True)


def fundamental_matrix(trans: Union[np.ndarray, scipy.sparse.csr_matrix],
                       stat: Optional[np.ndarray] = None,
                       ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    """Compute the fundamental matrix of the given Markov transition matrix."""
    return np.linalg.inv(np.eye(stat.size) - trans + stat[np.newaxis, :])


def componental_mfpt(trans: np.ndarray, **kwargs) -> np.ndarray:
    """Compute Markov mean first passage times per connected component of the chain."""
    n_comps, comp_labels = scipy.sparse.csgraph.connected_components(
        trans, **kwargs
    )
    hier_trans = transition_matrix(trans)
    absorbing = np.isclose(np.diag(hier_trans), 1)
    if n_comps == 1 and not absorbing.any():
        print('shortcut')
        return mfpt(hier_trans)
    else:
        print('longrun')
        times = np.full_like(hier_trans, fill_value=np.inf)
        # for each autonomous subsystem
        for comp_i in range(n_comps):
            is_comp = (comp_labels == comp_i)
            absorbing_i = np.flatnonzero(absorbing & is_comp)
            nonabsorbing_i = np.flatnonzero(~absorbing & is_comp)
            times[nonabsorbing_i[:, None], nonabsorbing_i] = mfpt(
                hier_trans[nonabsorbing_i[:, None], nonabsorbing_i]
            )
            times[absorbing_i, absorbing_i] = 1
    return times
