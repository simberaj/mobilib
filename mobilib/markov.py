
import numpy as np
import scipy.sparse.csgraph


def mfpt(trans: np.ndarray, stat: np.ndarray = None) -> np.ndarray:
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
    eigval, eigvec = np.linalg.eig(p.T)
    vec_stat = eigvec[:, np.argmin(abs(eigval - 1.0))].real
    return vec_stat / vec_stat.sum()


def transition_matrix(inter: np.ndarray) -> np.ndarray:
    return inter / inter.sum(axis=1)[:, np.newaxis]


def fundamental_matrix(trans, stat):
    return np.linalg.inv(np.eye(stat.size) - trans + stat[np.newaxis, :])


def componental_mfpt(trans: np.ndarray, **kwargs) -> np.ndarray:
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
    