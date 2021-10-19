from typing import Dict, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry

DEFAULT_HOME_CODE = 'k'
DEFAULT_WORK_CODE = 't'
DEFAULT_MULTIFX_CODE = 'm'


class Aimer:
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


def ipf(values: np.ndarray,
        rowsums: np.ndarray,
        colsums: np.ndarray,
        tol: float = 1e-9,
        max_iter: int = 100,
        ) -> np.ndarray:
    # print((values * 1000).astype(int))
    # print(rowsums, colsums)
    rowsums = rowsums.reshape(-1, 1)
    colsums = colsums.reshape(1, -1)
    prevalues = values.copy()
    for i in range(max_iter):
        prevalues[:] = values[:]
        prerowsums = values.sum(axis=1)
        values *= rowsums / np.where(prerowsums == 0, 1, prerowsums).reshape(-1, 1)
        # print('R')
        # print((values * 1000).astype(int))
        precolsums = values.sum(axis=0)
        values *= colsums / np.where(precolsums == 0, 1, precolsums).reshape(1, -1)
        # print('C')
        # print((values * 1000).astype(int))
        diff = abs(prevalues - values).sum()
        if diff <= tol:
            prerowsums = values.sum(axis=1)
            values *= rowsums / np.where(prerowsums == 0, 1, prerowsums).reshape(-1, 1)
            # print(i)
            break
    # print((values * 1000).astype(int))
    # print((values.sum(axis=1) * 1000).astype(int))
    # print((values.sum(axis=0) * 1000).astype(int))
    return values


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
        return  '<{0.__class__.__name__}({0.name},{0.source_aimer}-{0.target_aimer},selfinter={0.selfinter})>'.format(self)


class ProportionalRelationGenerator(AimedRelationGenerator):
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
        result = ipf(elig, sources / source_sum, targets / target_sum)
        return result


class MaximumRelationGenerator(AimedRelationGenerator):
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
                       ):
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
