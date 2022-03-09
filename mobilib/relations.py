"""Generate interactions from user anchor points and manage them."""

from typing import List, Dict, Optional, Iterable

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry

import mobilib.core

DEFAULT_HOME_CODE = 'k'
DEFAULT_WORK_CODE = 't'
DEFAULT_MULTIFX_CODE = 'm'


class Aimer:
    """Determine anchor weights for anchor interaction from their types."""
    def aim(self, types: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class UntypedAimer(Aimer):
    def aim(self, types: np.ndarray) -> np.ndarray:
        return np.ones(types.shape)

    def __repr__(self):
        return '<UntypedAimer>'


class TypedAimer(Aimer):
    def __init__(self, type_fractions: Dict[str, float]):
        self.type_fractions = type_fractions

    def aim(self, types: np.ndarray) -> np.ndarray:
        return np.stack([
            (types == code) * weight
            for code, weight in self.type_fractions.items()
        ]).sum(axis=0)

    def __repr__(self):
        return '<Aimer{}>'.format(self.type_fractions)


class AimedRelationGenerator:
    def __init__(self,
                 name: Optional[str] = None,
                 source_aimer: Aimer = UntypedAimer(),
                 target_aimer: Aimer = UntypedAimer(),
                 selfinter: bool = True,
                 ):
        self.name = name
        self.source_aimer = source_aimer
        self.target_aimer = target_aimer
        self.selfinter = selfinter

    def eligibility(self,
                    n: int,
                    ids: Optional[np.ndarray] = None
                    ) -> np.ndarray:
        elig = np.ones((n,n), dtype=bool)
        if not self.selfinter:
            if ids is None:
                np.fill_diagonal(elig, False)
            else:
                elig &= (ids.reshape(-1,1) != ids.reshape(1,-1))
        return elig

    def __repr__(self) -> str:
        return  (
            f'<{self.__class__.__name__}({self.name},'
            f'{self.source_aimer}-{self.target_aimer},'
            f'selfinter={self.selfinter})>'
        )


class ProportionalRelationGenerator(AimedRelationGenerator):
    """Generate many fractional interactions between all pairs of eligible anchors."""
    def relate(self,
               importances: np.ndarray,
               types: np.ndarray,
               ids: Optional[np.ndarray] = None,
               ) -> np.ndarray:
        n = len(importances)
        elig = self.eligibility(n, ids).astype(float)
        sources = importances * self.source_aimer.aim(types) * elig.max(axis=1)
        source_sum = sources.sum()
        if source_sum == 0:
            return None
        targets = importances * self.target_aimer.aim(types) * elig.max(axis=0)
        target_sum = targets.sum()
        if target_sum == 0:
            if self.selfinter:
                return np.diag(sources / source_sum)
            else:
                return None
        result = mobilib.core.ipf(
            elig,
            sources / source_sum,
            targets / target_sum
        )
        return result


class MaximumRelationGenerator(AimedRelationGenerator):
    """Generate a single interaction between the two most important anchors."""
    def relate(self,
               importances: np.ndarray,
               types: np.ndarray,
               ids: Optional[np.ndarray] = None,
               ) -> Optional[np.ndarray]:
        n = len(importances)
        elig = self.eligibility(n, ids).astype(float)
        sources = importances * self.source_aimer.aim(types) * elig.max(axis=1)
        source_sum = sources.sum()
        if source_sum == 0:
            return None
        has_unique_ids = (ids is None or len(np.unique(ids)) == n)
        # get largest eligible source
        if has_unique_ids:
            src_i = sources.argmax()
        else:
            src_i = np.flatnonzero(
                ids == pd.Series(sources).groupby(ids).sum().idxmax()
            )[0]
        targets = importances * self.target_aimer.aim(types) * elig.max(axis=0)
        target_sum = targets.sum()
        if target_sum <= targets[src_i]:   # only the source is a valid target
            if self.selfinter:
                tgt_i = src_i
            else:
                return None
        else:
            targets[src_i] = 0
            # get largest (still) eligible target
            if has_unique_ids:
                tgt_i = targets.argmax()
            else:
                tgt_i = np.flatnonzero(
                    ids == pd.Series(targets).groupby(ids).sum().idxmax()
                )[0]
        rels = np.zeros((n,n))
        rels[src_i,tgt_i] = 1
        return rels


def default_generators(home_code: str = DEFAULT_HOME_CODE,
                       work_code: str = DEFAULT_WORK_CODE,
                       multi_code: str = DEFAULT_MULTIFX_CODE,
                       multi_home_frac: float = .5,
                       selfinter: bool = True
                       ) -> List[AimedRelationGenerator]:
    home_aimer = TypedAimer({
        home_code: 1,
        multi_code: multi_home_frac,
    })
    work_aimer = TypedAimer({
        work_code: 1,
        multi_code: (1 - multi_home_frac),
    })
    return [
        ProportionalRelationGenerator('general',
            selfinter=selfinter,
        ),
        ProportionalRelationGenerator('homebased',
            source_aimer=home_aimer,
            selfinter=selfinter,
        ),
        ProportionalRelationGenerator('homework',
            source_aimer=home_aimer,
            target_aimer=work_aimer,
            selfinter=selfinter,
        ),
        MaximumRelationGenerator('genfirst',
            source_aimer=home_aimer,
            selfinter=selfinter,
        ),
        MaximumRelationGenerator('homefirst',
            source_aimer=home_aimer,
            selfinter=selfinter,
        ),
        MaximumRelationGenerator('homeworkfirst',
            source_aimer=home_aimer,
            target_aimer=work_aimer,
            selfinter=selfinter,
        ),
    ]


def to_lines(rel_df: pd.DataFrame,
             pts: gpd.GeoDataFrame,
             ) -> gpd.GeoDataFrame:
    """Create a line geodataframe from an interaction dataframe.

    Joins the point geodataframe to the interactions and creates straight lines
    connecting the interacting pairs of points.
    """
    from_id, to_id = rel_df.columns[:2]
    all_df = (
        rel_df
        .join(pts.rename(columns=lambda col: 'from_' + str(col)), on=from_id)
        .join(pts.rename(columns=lambda col: 'to_' + str(col)), on=to_id)
    )
    all_df['geometry'] = [
        shapely.geometry.LineString([pt1, pt2]).wkt
        for pt1, pt2 in zip(all_df['from_geometry'], all_df['to_geometry'])
    ]
    return all_df.drop(['from_geometry', 'to_geometry'], axis=1)


def from_anchors(df: pd.DataFrame,
                 generators: Iterable[AimedRelationGenerator],
                 site_id_col: str,
                 user_id_col: str,
                 importance_col: str,
                 anchor_type_col: str,
                 ) -> pd.DataFrame:
    """Generate interactions from user anchor points using given generators.

    :param df: A dataframe containing site_id_col, user_id_col (key columns),
        importance_col and anchor_type_col.
    :param generators: Generators producing interactions.
    :param site_id_col: Column with ID of site or spatial unit defining the
        user anchor.
    :param user_id_col: Column with ID of user. Used only to identify anchors
        belonging to the same person, does not appear in the output.
    :param importance_col: Anchor point importance measure column. May be
        arbitrarily scaled.
    :param anchor_type_col: Anchor point type (home, work, etc.) column. Used
        by some generators to select eligible anchors, etc.
    :return: A dataframe with two site ID columns (from_- and to_-prefixed
        site ID column name) and interaction magnitude columns, one per
        provided generator and using their names for column names.
        The interaction magnitude is measured in number of users and may be
        fractional depending on the generator.
    """
    sites = numpy.sort(df[site_id_col].unique())
    n_sites = len(sites)
    series = []
    for gener in generators:
        matrix = numpy.zeros((n_sites, n_sites))
        for uid, subdf in df.groupby(user_id_col):
            ids = subdf[site_id_col].values
            rels = gener.relate(
                subdf[importance_col].values,
                subdf[anchor_type_col].values,
                ids=ids,
            )
            site_indexes = numpy.searchsorted(sites, ids)
            if rels is not None:
                for src_i, tgt_i in enumerate(site_indexes):
                    matrix[tgt_i,site_indexes] += rels[src_i]
        rowis, colis = numpy.nonzero(matrix)
        series.append(pd.Series(
            matrix[rowis,colis],
            index=[sites[rowis],sites[colis]],
            name=gener.name,
        ))
    out_df = series[0]
    for ser in series[1:]:
        out_df = pd.merge(out_df, ser, left_index=True, right_index=True, how='outer')
    out_df.index.names = ['from_' + site_id_col, 'to_' + site_id_col]
    return out_df.fillna(0).reset_index()


if __name__ == '__main__':
    imps = np.array([4,2,2,1,1,1])
    codes = np.array([
        DEFAULT_HOME_CODE,
        DEFAULT_WORK_CODE,
        DEFAULT_MULTIFX_CODE,
        DEFAULT_WORK_CODE,
        DEFAULT_HOME_CODE,
        '',
    ])
    ids = np.array([1,2,3,4,4,5])
    for selfinter in (True, False):
        for gen in default_generators(selfinter=selfinter):
            print(gen)
            rels = gen.relate(imps, codes, ids)
            if rels is not None:
                print((rels * 1000).astype(int))
            else:
                print(rels)
    print()
