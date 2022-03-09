'''Differential privacy anonymization for mobile positioning data.

This module provides various anonymization methods based on differential privacy
methods from the diffprivlib library. They work by adding a small amount of
random noise to the anonymized values to prevent detecting whether any specific
user's data was used in compiling the values. For more detail and literature
references, see the diffprivlib documentation.

The overall interface is defined by :class:`DPInteractionAnonymizer`;
:class:`AtomicInteractionAnonymizer` is a simple implementation that works
primarily for one-dimensional data since it does not equalize interaction counts
and sums, while :class:`GravityInteractionAnonymizer` does so and is the
recommended default choice.

The functionality was primarily developed to anonymize spatial interaction
values, but can be used in a wide range of contexts.
'''

from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import diffprivlib.mechanisms


def match_input_type(output: pd.Series, input: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(input):
        output = output.round()
    return output[output > 0].astype(input.dtype)


class DPInteractionAnonymizer:
    def randomise(self, inters: pd.Series) -> pd.Series:
        raise NotImplementedError


class AtomicInteractionAnonymizer:
    '''Noisify each interaction individually.'''

    DEFAULT_MECHANISM_CLASS = diffprivlib.mechanisms.Geometric

    def __init__(self,
                 mechanism: Optional[diffprivlib.mechanisms.DPMechanism] = None,
                 epsilon: float = 0.5,
                 seed: Optional[int] = None,
                 ):
        if mechanism is None:
            mechanism = self.DEFAULT_MECHANISM_CLASS(epsilon=epsilon)
        self.mechanism = mechanism
        self.seed = seed
    
    def randomise(self, inters: pd.Series) -> pd.Series:
        if self.seed is not None:
            diffprivlib.utils.global_seed(self.seed)
        return match_input_type(inters.apply(self.mechanism.randomise), inters)


class GravityInteractionAnonymizer:
    '''Noisify interactions, preserving some of their invariants.

    This anonymizer preserves (to the available precision) the sums and counts
    of values of outgoing interactions for each unit. Wherever some interactions
    of the unit drop to zero by the noisification, new (previously zero-valued)
    interactions with other units are created and noisified. These interactions
    are primarily created with units with high potential interaction predicted
    by the gravity model on unit weights (such as populations) and distances
    between their centroids (points).
    '''

    DEFAULT_INNER_ANONYMIZER = AtomicInteractionAnonymizer()

    def __init__(self,
                 unit_weights: pd.Series,
                 unit_pts: gpd.GeoSeries,
                 inner: Optional[DPInteractionAnonymizer] = None,
                 supplement_k: int = 3,
                 seed: Optional[int] = None,
                 ):
        if inner is None:
            inner = self.DEFAULT_INNER_ANONYMIZER
        self.inner = inner
        self.unit_weights = unit_weights
        self.unit_pts = unit_pts
        self._gravity_matrix = None
        self.unit_ids = self.unit_pts.index.values
        self.supplement_k = supplement_k
        self.seed = seed

    @property
    def gravity_matrix(self) -> np.ndarray:
        if self._gravity_matrix is None:
            self._compute_gravity_matrix()
        return self._gravity_matrix

    def _compute_gravity_matrix(self):
        weights = (
            self.unit_weights
            .reindex(self.unit_pts.index)
            .fillna(0)
            .astype(self.unit_weights.dtype)
        )
        xs = self.unit_pts.x
        ys = self.unit_pts.y
        massprod = weights.values * weights.values.reshape(-1, 1)
        distsq = (
            (xs.values - xs.values.reshape(-1, 1)) ** 2
            + (ys.values - ys.values.reshape(-1, 1)) ** 2
        )
        distsq[np.isclose(distsq, 0)] = np.inf
        self._gravity_matrix = massprod / distsq

    def randomise(self, inters: pd.Series) -> pd.Series:
        '''Noisify the interactions.
        
        :param inters: A series with a two-level index; the first level should
            be identifiers of interaction source units, the second level should
            contain targets.
        '''
        if self.seed is not None:
            diffprivlib.utils.global_seed(self.seed)
            np.random.seed(self.seed)
        randomised = self.inner.randomise(inters)
        induced_zero_counts = (
            inters.groupby(level=0).count() - randomised.groupby(level=0).count()
        ).fillna(0).astype(int)
        gravity_zeros = self._get_gravity_supplement_candidates(
            randomised, induced_zero_counts * self.supplement_k
        )
        randomised_zeros = self.inner.randomise(gravity_zeros)
        all_randomised = pd.concat((
            randomised,
            self._filter_supplements(randomised_zeros, induced_zero_counts)
        )).sort_index()
        equalized = self._equalize_outsums(all_randomised, inters)
        return match_input_type(equalized, inters)

    @staticmethod
    def _equalize_outsums(randomised: pd.Series, inters: pd.Series) -> pd.Series:
        rand_outsums = randomised.groupby(level=0).sum()
        orig_outsums = inters.groupby(level=0).sum()
        corr_coef = (orig_outsums / rand_outsums).replace([np.inf, -np.inf], np.nan).fillna(1)
        return (
            pd.DataFrame({'inters': randomised})
            .join(corr_coef.rename('corr_coef'), on=randomised.index.names[0])
            .fillna({'inters': 0})
            .eval('inters * corr_coef')
            .rename(randomised.name)
        )

    def _filter_supplements(self, suppl: pd.Series, counts: pd.Series) -> pd.Series:
        # return randomly selected counts from suppl
        return (
            pd.DataFrame({'value': suppl})
            .assign(randval=np.random.rand(len(suppl)))
            .assign(randrank=lambda df: df.groupby(suppl.index.names[0])['randval'].rank(method='first'))
            .join(counts.rename('count'), on=suppl.index.names[0], how='inner')
            .query('randrank <= count')
            ['value']
            .rename(suppl.name)
        )

    def _get_gravity_supplement_candidates(self, orig: pd.Series, needed: pd.Series) -> pd.Series:
        n_conns = (
            orig.groupby(level=0).count() + needed
        ).reindex(self.unit_ids).fillna(0).astype(int)
        n = self.gravity_matrix.shape[0]
        threshold = (
            np.sort(self.gravity_matrix, axis=1)
            [np.arange(n), n - n_conns.values - 1]
            .reshape(-1, 1)
        )
        is_expected = self.gravity_matrix > threshold
        from_is, to_is = np.nonzero(is_expected)
        return pd.Series(
            0,
            index=pd.MultiIndex.from_arrays(
                arrays=(self.unit_ids[from_is], self.unit_ids[to_is]),
                names=orig.index.names
            )
        ).drop(orig.index, errors='ignore')
