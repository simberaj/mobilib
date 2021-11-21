
from __future__ import annotations

import random
import logging
import warnings
from typing import List, Dict, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd
import scipy.sparse
import shapely.geometry

import mobilib.markov
import mobilib.vector

logger = logging.getLogger(__name__)


def components_by_targets(targets: np.ndarray,
                          strong: bool = True,
                          return_count: bool = False,
                          ) -> Optional[np.ndarray]:
    n = len(targets)
    target_neigh = scipy.sparse.csr_matrix(
        (
            np.ones(n, dtype=bool),
            (np.arange(n), targets),
        ),
        shape=(n, n)
    )
    n_strong, labels = scipy.sparse.csgraph.connected_components(
        target_neigh,
        directed=True,
        connection=('strong' if strong else 'weak'),
        return_labels=True
    )
    out_labels = None if n_strong == n else labels
    if return_count:
        return n_strong, out_labels
    else:
        return out_labels


def random_weighted(cum_weights: np.ndarray) -> int:
    # argmax gets first occurrence of True (the maximum)
    return (cum_weights >= random.random()).argmax()


class Relations:
    def __init__(self, matrix: np.ndarray, weights: Optional[np.ndarray] = None):
        self.matrix = self._correct_matrix(matrix)
        assert len(set(self.matrix.shape)) == 1    # matrix must be square
        self.n = self.matrix.shape[0]
        self.outsums = self.matrix.sum(axis=1)
        self.insums = self.matrix.sum(axis=0)
        self.weights = weights if weights is not None else self.outsums
        self.totweight = self.weights.sum()
        self.unit_weights = self.weights / self.totweight
        self.cum_weights = self.unit_weights.cumsum()
        self.selfrels = np.diag(self.matrix)
        self.outsums_noself = self.outsums - self.selfrels
        self.transition_probs = self.matrix / self.outsums[:, np.newaxis]
        self.selfprobs = np.diag(self.transition_probs)
        rels_w = (self.matrix - np.diag(self.selfrels))
        rels_w_sums = rels_w.sum(axis=1)
        self.weighting = rels_w * (
            self.outsums / np.where(rels_w_sums, rels_w_sums, 1)
        )[:, np.newaxis] / self.matrix.sum()

    def _correct_matrix(self, matrix: np.ndarray) -> np.ndarray:
        # add ones on the diagonal for absorbing states
        no_outflows = (matrix.sum(axis=1) == 0)
        return matrix + np.diag(no_outflows)

    def main_component(self) -> np.ndarray:
        n_comps, comp_labels = scipy.sparse.csgraph.connected_components(
            self.matrix,
            directed=True,
            connection='weak',
            return_labels=True
        )
        if n_comps == 1:
            return np.ones(len(comp_labels), dtype=bool)
        else:
            maxcomp = np.bincount(comp_labels).argmax()
            return comp_labels == maxcomp

    def weighted_sum(self, items: np.ndarray) -> float:
        return (items * self.unit_weights).sum()

    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame,
                       from_id_col: Optional[str] = None,
                       to_id_col: Optional[str] = None,
                       strength_col: Optional[str] = None,
                       ) -> Tuple[Relations, np.ndarray]:
        if from_id_col is None:
            from_id_col = df.columns[0]
        if to_id_col is None:
            to_id_col = df.columns[1]
        if strength_col is None:
            strength_col = df.columns[-1]
        all_ids = np.array(list(sorted(set(
            list(df[from_id_col].unique())
            + list(df[to_id_col].unique())
        ))))
        n = len(all_ids)
        matrix = np.zeros((n, n), dtype=df[strength_col].dtype)
        from_ids = np.searchsorted(all_ids, df[from_id_col])
        to_ids = np.searchsorted(all_ids, df[to_id_col])
        matrix[from_ids, to_ids] = df[strength_col].values
        return cls(matrix), all_ids


class Model:
    PARENT_COL = 'parent'
    STAGE_COL = 'stage'
    INDEX_COL = 'id'
    ORG_COL = 'organic'
    AUTONOMOUS_COL = 'autonomous'
    HIER_COL = 'hier'
    BASE_COLS = [PARENT_COL, STAGE_COL]

    STAGE_AUTONOM = 'A'
    STAGE_NODAL = 'N'
    STAGE_ORG = 'O'
    STAGES = [STAGE_AUTONOM, STAGE_NODAL, STAGE_ORG]

    def __init__(self, df: pd.DataFrame, more_columns: bool = 'error'):
        if more_columns == 'ignore':
            df = df[self.BASE_COLS].copy()
        if set(df.columns) != set(self.BASE_COLS):
            raise ValueError(f'superfluous columns in HSSM input: {df.columns}')
        self.df = df

    def reset_cache(self):
        self.df.drop(
            [col for col in self.df.columns if col not in self.BASE_COLS],
            axis=1, inplace=True
        )

    @property
    def n(self):
        return len(self.df)

    @property
    def index(self):
        return self.df.index

    @property
    def parents(self):
        return self.df[self.PARENT_COL]

    def parent_is(self, selector: Optional[pd.Series] = None) -> np.ndarray:
        parents = self.df[self.PARENT_COL]
        if selector is not None:
            parents = parents[selector]
        return self.df.index.get_indexer(parents)

    def hier_is(self):
        return np.flatnonzero(self.hier)

    def index_of(self, id) -> int:
        return self.df.index.get_loc(id)

    def indices_of(self, ids: np.ndarray) -> np.ndarray:
        return self.df.index.get_indexer(ids)

    @property
    def organics(self):
        if self.ORG_COL not in self.df.columns:
            self.df[self.ORG_COL] = (self.df[self.STAGE_COL] == self.STAGE_ORG)
        return self.df[self.ORG_COL]

    @property
    def autonomous(self):
        if self.AUTONOMOUS_COL not in self.df.columns:
            self.df[self.AUTONOMOUS_COL] = (self.df[self.STAGE_COL] == self.STAGE_AUTONOM)
        return self.df[self.AUTONOMOUS_COL]

    @property
    def hier(self):
        if self.HIER_COL not in self.df.columns:
            self.df[self.HIER_COL] = ~self.autonomous
        return self.df[self.HIER_COL]
    
    def org_nodes(self):
        org_subs = self.parents[self.organics]
        org_parents = org_subs.unique()
        return pd.concat((org_subs, pd.Series(org_parents, index=org_parents)))

    @property
    def tree(self):
        return self.df.loc[self.hier,:]

    def root_id(self):
        return self.df[(self.parents == self.index) & self.hier].index[0]

    def root_i(self):
        return np.flatnonzero((self.parents == self.index) & self.hier)[0]

    def root_ids(self):
        root_id = self.root_id()
        root_org_subs = (self.parents == root_id) & self.organics
        return np.hstack(([root_id], root_org_subs[root_org_subs].index))

    def root_is(self):
        return self.indices_of(self.root_ids())

    def is_root(self, key: Any) -> bool:
        row = self.df.loc[key, :]
        return row[self.PARENT_COL] == key and row[self.STAGE_COL] == self.STAGE_NODAL

    def is_valid(self) -> bool:
        try:
            self.check()
            return True
        except AssertionError:
            return False

    def check(self) -> None:
        # conditions:
        # all parent ids must be in the index
        assert set(self.df[self.PARENT_COL]).issubset(set(self.index))
        # only valid stages
        assert set(self.df[self.STAGE_COL].unique()).issubset(set(self.STAGES))
        # no cycles in the hierarchy tree
        assert components_by_targets(self.parent_is()) is None, 'cyclic parental relationship'
        # targets of bindings must be nodal
        assert (self.df[self.STAGE_COL][self.tree[self.PARENT_COL]] == self.STAGE_NODAL).all(), \
            'dependence on autonomous unit or non-head unit of organic subsystem'
        # the tree must be a single component
        n_tree_comps, comp_labels = components_by_targets(
            self.parent_is(self.hier), strong=False, return_count=True
        )
        assert n_tree_comps == 1, 'non-unified hierarchy tree (actually a forest)'
        # there must be a single root
        assert len(self.df[(self.parents == self.index) & self.hier]) == 1, 'multi-root hierarchy tree'


    @classmethod
    def from_arrays(cls,
                    parents: np.ndarray,
                    stages: np.ndarray,
                    index: Optional[np.ndarray] = None,
                    ) -> StagedModel:
        return cls(pd.DataFrame({
            cls.PARENT_COL: parents,
            cls.STAGE_COL: stages,
        }, index=index))

    @classmethod
    def from_flag_arrays(cls,
                         parents: np.ndarray,
                         autonomous: np.ndarray,
                         organics: np.ndarray,
                         index: Optional[np.ndarray] = None,
                         ) -> StagedModel:
        stages = np.where(
            autonomous, self.STAGE_AUTONOM,
            np.where(
                organics, self.STAGE_ORG,
                self.STAGE_NODAL
            )
        )
        return cls.from_arrays(parents, stages, index=index)

    def connection_df(self) -> pd.DataFrame:
        self.hier, self.organics    # to have secondary columns ready
        conn_df = (
            self.df
            .rename_axis(index=self.INDEX_COL)
            .reset_index()
            .query(f'{self.INDEX_COL} != {self.PARENT_COL} and {self.HIER_COL}')
            [[self.INDEX_COL, self.PARENT_COL, self.ORG_COL]]
        )
        if conn_df[self.ORG_COL].any():
            return self._add_organic_connections(conn_df)
        else:
            return conn_df

    def _add_organic_connections(self, conn_df: pd.DataFrame) -> pd.DataFrame:
        org_conn_df = conn_df.query(self.ORG_COL)
        # For units bound to organic subsystems, add connections to their non-head units.
        conn_add_simples_df = self._join_on_parent(
            conn_df.query(f'not {self.ORG_COL}'), org_conn_df
        )
        # For non-head units in organic subsystems, add connections to other non-head units.
        conn_add_org_head_df = self._join_on_parent(org_conn_df, org_conn_df).query(
            f'{self.INDEX_COL} > {self.PARENT_COL}'
        )
        # For non-head units in organic subsystems, add connections to their head's parent.
        conn_add_head_tgt_df = org_conn_df.merge(
            pd.concat([conn_df, conn_add_simples_df]),
            left_on=self.PARENT_COL,
            right_on=self.INDEX_COL,
            suffixes=('_src', '')
        )[[f'{self.INDEX_COL}_src', self.PARENT_COL, self.ORG_COL]].rename(
            columns={f'{self.INDEX_COL}_src': self.INDEX_COL}
        )
        return pd.concat([
            conn_df,
            conn_add_simples_df.assign(**{self.ORG_COL: False}),
            conn_add_org_head_df.assign(**{self.ORG_COL: True}),
            conn_add_head_tgt_df.assign(**{self.ORG_COL: False}),
        ], ignore_index=True)

    def _join_on_parent(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        indexcol_key_tgt = f'{self.INDEX_COL}_tgt'
        return df1.merge(
            df2, on=self.PARENT_COL, suffixes=('', '_tgt')
        )[[self.INDEX_COL, indexcol_key_tgt, self.ORG_COL]].rename(
            columns={indexcol_key_tgt: self.PARENT_COL}
        )

    def tree_keys(self, head_key: Any) -> List[Any]:
        parents = self.parents.reset_index(drop=True)
        prev_n_keys = 0
        tree_keys = {head_key: None}
        while len(tree_keys) > prev_n_keys:
            prev_n_keys = len(tree_keys)
            tree_keys.update({unit_key: None for unit_key in self.parents[self.parents.isin(tree_keys.keys())].index})
        return list(tree_keys.keys())

    def descendant_keys(self, head_key: Any) -> List[Any]:
        return self.tree_keys(head_key)[1:]

    def copy(self) -> Model:
        return Model(self.df[self.BASE_COLS].copy())

    def to_lines(self, points: 'gpd.GeoSeries') -> 'gpd.GeoSeries':
        org_node_df = (
            pd.DataFrame({'node': self.org_nodes()})
            .join(points.rename('from_pt'))
        )
        org_center_pts = org_node_df.groupby('node')['from_pt'].agg(mobilib.vector.centroid)
        org_pt_df = (
            org_node_df
            .join(org_center_pts.rename('to_pt'), on='node')
            [['from_pt', 'to_pt']]
        )
        if len(org_pt_df) == 0:
            org_lines = type(points)()
        else:
            org_lines = org_pt_df.apply(lambda row: mobilib.vector.straight_line(
                row['from_pt'], row['to_pt']
            ), axis=1)
        org_lines.index = pd.MultiIndex.from_arrays((
            org_lines.index,
            pd.Series(self.STAGE_ORG, index=org_lines.index)
        ))
        is_nodal = self.df[self.STAGE_COL] == self.STAGE_NODAL
        pts_nodal = points[is_nodal]
        pts_nodal[org_center_pts.index] = org_center_pts
        nodal_lines = (
            self.df.loc[is_nodal,[self.PARENT_COL]]
            .drop(self.root_id())
            .join(pts_nodal.rename('from_pt'))
            .join(pts_nodal.rename('to_pt'), on=self.PARENT_COL)
            [['from_pt', 'to_pt']]
            .apply(lambda row: mobilib.vector.straight_line(row['from_pt'], row['to_pt']), axis=1)
        )
        nodal_lines.index = pd.MultiIndex.from_arrays((
            nodal_lines.index,
            pd.Series(self.STAGE_NODAL, index=nodal_lines.index)
        ))
        auto_lines = (
            points[self.df[self.STAGE_COL] == self.STAGE_AUTONOM]
            .apply(lambda pt: mobilib.vector.straight_line(pt, pt))
        )
        auto_lines.index = pd.MultiIndex.from_arrays((
            auto_lines.index,
            pd.Series(self.STAGE_AUTONOM, index=auto_lines.index)
        ))
        return (
            type(points)(pd.concat([nodal_lines, org_lines, auto_lines]))
            .rename_axis(index=[self.INDEX_COL, self.STAGE_COL + '_line'])
        )


class FitnessCriterion:
    def evaluate(self, model: Model, rels: Relations) -> float:
        raise NotImplementedError

    def evaluate_nodes(self, model: Model, rels: Relations) -> pd.Series:
        raise NotImplementedError


def fitness_criterion(name: str, **kwargs) -> FitnessCriterion:
    return FITNESS_CRITERIA[name](**kwargs)


class MFPTCriterion(FitnessCriterion):
    def binding_matrix(self, model: Model, rels: Relations) -> np.ndarray:
        raise NotImplementedError
        # binding matrix would have to re-include self-interactions

    def evaluate(self, model: Model, rels: Relations) -> float:
        return self.evaluate_nodes(model, rels).sum()

    def evaluate_nodes(self, model: Model, rels: Relations) -> pd.Series:
        bmatrix = self.binding_matrix(model, rels)
        hier_is = model.hier_is()
        sel_bmatrix = bmatrix[hier_is][:,hier_is].tocsr()
        transition_matrix = mobilib.markov.transition_matrix(sel_bmatrix)
        tree_mfpt = mobilib.markov.mfpt(transition_matrix)
        return pd.Series(
            (rels.weighting[hier_is][:,hier_is] / tree_mfpt).sum(axis=1),
            index=hier_is
        )


class DijkstraCriterion(FitnessCriterion):
    def evaluate(self, model: Model, rels: Relations) -> float:
        return self.evaluate_nodes(model, rels).sum()

    def evaluate_nodes(self, model: Model, rels: Relations) -> pd.Series:
        bmatrix = self.binding_matrix(model, rels)
        hier_is = model.hier_is()
        sel_bmatrix = bmatrix[hier_is][:,hier_is].tocsr()
        path_costs = (
            scipy.sparse.csgraph.dijkstra(edge_costs)
            + np.diag(np.ones(edge_costs.shape[0]))
        )
        return pd.Series(
            (rels.weighting[hier_is][:,hier_is] / path_costs).sum(axis=1),
            index=hier_is
        )

    def binding_matrix(self, model: Model, rels: Relations) -> np.ndarray:
        diag = np.diag(rels.matrix)
        conn_df = model.connection_df()

        from_is = model.indices_of(conn_df[model.INDEX_COL])
        to_is = model.indices_of(conn_df[model.PARENT_COL])
        conn_values_along = rels.matrix[from_is, to_is]
        conn_values_cnter = rels.matrix[to_is, from_is]
        conn_values = np.where(
            conn_df[model.ORG_COL],
            (
                np.minimum(diag[from_is], diag[to_is])
                / np.minimum(conn_values_along, conn_values_cnter)
            ),
            diag[from_is] / conn_values_along
        )
        out_shape = (len(diag), len(diag))
        return (
            scipy.sparse.coo_matrix((conn_values, (from_is, to_is)), shape=out_shape)
            + scipy.sparse.coo_matrix((conn_values, (to_is, from_is)), shape=out_shape)
        )


class StageStateCriterion:
    def evaluate(self, model: Model, rels: Relations) -> float:
        return rels.weighted_sum(self.evaluate_nodes(model, rels)[model.index])

    def evaluate_nodes(self, model: Model, rels: Relations) -> pd.Series:
        maxflow_k = self.maxflow_k(rels)
        conn_df = model.connection_df()
        edges = self.edge_matrix(model, conn_df)
        return (
            pd.Series(self.edge_fits(edges, rels), index=model.index).where(model.hier, 1)
            * self.stage_coefs(model, rels, maxflow_k, conn_df.query('organic'))
        )

    def maxflow_k(self, rels: Relations) -> float:
        return self._lstsq1(rels.outsums, (rels.matrix - np.diag(rels.selfrels)).max(axis=1))

    def tree_fits(self, model: Model, rels: Relations) -> pd.Series:
        edges = self.edge_matrix(model)
        return pd.Series(self.edge_fits(edges, rels), index=model.index).where(model.hier, 1)

    def edge_fits(self, edges: scipy.sparse.coo_matrix, rels: Relations) -> pd.Series:
        # add directed=False if it helps with performance here
        path_lengths = scipy.sparse.csgraph.dijkstra(edges, unweighted=True)
        all_ids = np.arange(rels.n)
        # set diagonal to one to prevent nans
        path_lengths[all_ids,all_ids] += 1
        tot_outprobs = (1 - rels.selfprobs)[:,np.newaxis]
        relative_rels = rels.transition_probs / np.where(tot_outprobs == 0, 1, tot_outprobs)
        relative_rels[all_ids,all_ids] = 0
        return (relative_rels / path_lengths).sum(axis=1)
    
    def path_lengths(self, model: Model) -> np.ndarray:
        return scipy.sparse.csgraph.dijkstra(self.edge_matrix(model), unweighted=True)

    def edge_matrix(self, model: Model, conn_df: Optional[pd.DataFrame] = None) -> scipy.sparse.coo_matrix:
        if conn_df is None:
            conn_df = model.connection_df()
        from_is = model.indices_of(conn_df[model.INDEX_COL])
        to_is = model.indices_of(conn_df[model.PARENT_COL])
        out_shape = (model.n, ) * 2
        ones = np.ones(len(from_is), dtype=bool)
        return (
            scipy.sparse.coo_matrix((ones, (from_is, to_is)), shape=out_shape)
            + scipy.sparse.coo_matrix((ones, (to_is, from_is)), shape=out_shape)
        )

    def stage_coefs(self,
                    model: Model,
                    rels: Relations,
                    maxflow_k: Optional[float] = None,
                    org_conn_df: Optional[pd.DataFrame] = None,
                    ) -> pd.Series:
        if maxflow_k is None: maxflow_k = self.maxflow_k(rels)
        if org_conn_df is None: org_conn_df = model.connection_df().query('organic')
        nodality = np.minimum(self.nodality(model, rels, maxflow_k), 1)
        organicity = self.organicity(model, rels, maxflow_k, org_conn_df)
        return (
            nodality
            .where(model.hier, 1 - nodality)
            .multiply(organicity, fill_value=1)
        )

    def nodality(self,
                 model: Model,
                 rels: Relations,
                 maxflow_k: Optional[float] = None,
                 ) -> pd.Series:
        if maxflow_k is None: maxflow_k = self.maxflow_k(rels)
        org_df = model.df[model.organics]
        from_org_ids = model.indices_of(org_df.index)
        to_org_ids = model.indices_of(org_df[model.PARENT_COL])
        # from_org_ids, to_org_ids
        nodality_matrix = rels.matrix - np.diag(rels.selfrels)
        nodality_matrix[:,to_org_ids] += nodality_matrix[:,from_org_ids]
        maxflows = np.maximum(nodality_matrix.max(axis=1), nodality_matrix.max(axis=0))
        return pd.Series(maxflows / (maxflow_k * rels.outsums), index=model.index)

    def organicity(self,
                   model: Model,
                   rels: Relations,
                   maxflow_k: Optional[float] = None,
                   org_conn_df: Optional[pd.DataFrame] = None,
                   ) -> pd.Series:
        return (
            np.minimum(self.symcoh(model, rels, maxflow_k, org_conn_df), 1)
            # * self.outsim(model, rels)
        )
    
    def symcoh(self,
               model: Model,
               rels: Relations,
               maxflow_k: Optional[float] = None,
               org_conn_df: Optional[pd.DataFrame] = None,
               ) -> pd.Series:
        if maxflow_k is None: maxflow_k = self.maxflow_k(rels)
        if org_conn_df is None: org_conn_df = model.connection_df().query('organic')
        org_link_df = pd.concat([
            org_conn_df,
            org_conn_df.rename(columns={
                model.INDEX_COL: model.PARENT_COL,
                model.PARENT_COL: model.INDEX_COL
            })
        ], ignore_index=True)
        from_is = model.indices_of(org_link_df[model.INDEX_COL])
        to_is = model.indices_of(org_link_df[model.PARENT_COL])
        strengths = np.minimum(rels.matrix[from_is, to_is], rels.matrix[to_is, from_is])
        # weights = np.sqrt(rels.outsums[from_is] ** 2 + rels.outsums[to_is] ** 2)
        weights = np.maximum(rels.outsums[from_is], rels.outsums[to_is])
        org_link_df['organicity'] = strengths / (maxflow_k * weights)
        return org_link_df.groupby(model.INDEX_COL)['organicity'].min()
    
    def outsim(self,
               model: Model,
               rels: Relations,
               ) -> pd.Series:
        wt_ser = pd.Series(rels.weights, index=model.df.index)
        org_nodes = model.org_nodes()
        org_weights = wt_ser[org_nodes.index]
        org_node_df = (
            pd.DataFrame({'node': org_nodes, 'weight': org_weights})
            .join(org_weights.groupby(org_nodes).sum().rename('node_weight'), on='node')
            .assign(node_frac=lambda df: df.eval('weight / node_weight'))
            .drop(['weight', 'node_weight'], axis=1)
        )
        outsims = pd.Series(1., index=org_node_df.index)
        for node_id, subdf in org_node_df.groupby('node'):
            indexer = model.indices_of(subdf.index)
            sel_outs = rels.matrix[indexer]
            node_fracs = subdf['node_frac'].values
            expect_outs = sel_outs.sum(axis=0) * node_fracs[:,np.newaxis]
            sel_ins = rels.matrix[:,indexer]
            expect_ins = sel_ins.sum(axis=1)[:,np.newaxis] * node_fracs
            min_outs = np.minimum(sel_outs, expect_outs)
            max_outs = np.maximum(sel_outs, expect_outs)
            min_ins = np.minimum(sel_ins, expect_ins)
            max_ins = np.maximum(sel_ins, expect_ins)
            min_outs[:,indexer] = 0
            max_outs[:,indexer] = 0
            min_ins[indexer] = 0
            max_ins[indexer] = 0
            tot_mins = min_outs.sum(axis=1).flatten() + min_ins.sum(axis=0)
            tot_maxs = max_outs.sum(axis=1).flatten() + max_ins.sum(axis=0)
            outsims[subdf.index] = tot_mins / tot_maxs
        return outsims

    @staticmethod
    def _lstsq1(x: np.ndarray, y: np.ndarray) -> float:
        return x.dot(y) / x.transpose().dot(x)


class FlowCoverageCriterion:
    def evaluate(self, model: Model, rels: Relations) -> float:
        conn_df = model.connection_df()
        coverage = scipy.sparse.coo_matrix(self.edge_matrix(model, conn_df))
        covered = rels.matrix[coverage.row, coverage.col].sum()
        total = rels.matrix.sum() - rels.selfrels.sum()
        n_units = model.n
        n_links = n_units + conn_df[model.ORG_COL].sum()
        return covered / total * n_units / n_links
        
    def edge_matrix(self, model: Model, conn_df: Optional[pd.DataFrame] = None) -> scipy.sparse.coo_matrix:
        if conn_df is None:
            conn_df = model.connection_df()
        from_is = model.indices_of(conn_df[model.INDEX_COL])
        to_is = model.indices_of(conn_df[model.PARENT_COL])
        out_shape = (model.n, ) * 2
        ones = np.ones(len(from_is), dtype=bool)
        fwd = scipy.sparse.coo_matrix((ones, (from_is, to_is)), shape=out_shape)
        back = scipy.sparse.coo_matrix((
            ones[:conn_df[model.ORG_COL].sum()],
            (to_is[conn_df[model.ORG_COL]], from_is[conn_df[model.ORG_COL]])
        ), shape=out_shape)
        return fwd + back
    

FITNESS_CRITERIA: Dict[str, type] = {
    'mfpt': MFPTCriterion,
    'dijkstra': DijkstraCriterion,
    'stagestate': StageStateCriterion,
}


class ModelBuilder:
    deterministic: bool = True

    def build(self, rels: Relations, index: Optional[np.ndarray] = None) -> StagedModel:
        raise NotImplementedError


def model_builder(name: str, **kwargs) -> ModelBuilder:
    return MODEL_BUILDERS[name](**kwargs)


class IterativeDecyclingBuilder(ModelBuilder):
    def __init__(self, criterion: Optional[FitnessCriterion] = None, verbose: bool = False):
        self.verbose = verbose

    def build(self, rels: Relations, index: Optional[np.ndarray] = None) -> Model:
        relmatrix = rels.matrix - np.diag(rels.selfrels)
        maincomp = rels.main_component()
        parents = np.empty(rels.n, dtype=np.int64)
        stages = np.full(rels.n, 'N')
        if not maincomp.all():
            # we have some autonomous units
            autonomous = ~maincomp
            auto_ind = np.flatnonzero(autonomous)
            self._check_autonom_interrels(relmatrix, autonomous)
            parents[autonomous] = auto_ind
            stages[autonomous] = Model.STAGE_AUTONOM
            if self.verbose:
                logger.debug('setting units %s to autonomous', ', '.join(str(x) for x in index[auto_ind].tolist()))
        # now, fire up the main loop
        mainrels = relmatrix[maincomp, :][:, maincomp].copy()
        mainweights = rels.weights[maincomp].copy()
        mainorgs = np.zeros_like(mainweights, dtype=bool)
        main_is = np.flatnonzero(maincomp)
        selfrels = scipy.sparse.coo_matrix((
            rels.selfrels[maincomp],
            (np.arange(len(main_is)), np.arange(len(main_is)))
        )).tocsr()
        while True:
            root_i = self.select_root(mainrels + selfrels)
            if self.verbose: logger.debug('root found at %s', index[main_is[root_i]])
            targets = self.select_parents(mainrels)
            targets[root_i] = root_i
            # find strongly connected components in the directed hierarchy graph,
            # these will be cycles
            cycle_labels = components_by_targets(targets)
            if cycle_labels is None:
                if self.verbose: logger.debug('single component, terminating')
                # no connected components = no cycles, model is complete
                break
            head_i, other_is = self._select_to_organify(cycle_labels, mainweights)
            if self.verbose: 
                logger.debug('organifying cycle %s <- %s',
                             index[main_is[head_i]], index[main_is[other_is]])
            mainorgs[other_is] = True
            # contract weights and relations
            self._contract_weights(mainweights, head_i, other_is)
            self._contract_rels(mainrels, selfrels, head_i, other_is)
        parents[maincomp] = main_is[targets]
        stages[main_is[mainorgs]] = 'O'
        if index is not None:
            parents = index[parents]
        model = Model.from_arrays(parents, stages, index=index)
        # with pd.option_context('display.max_rows', 100):
        #     print(model.df.reset_index())
        return model

    @staticmethod
    def _check_autonom_interrels(rels_matrix: np.ndarray, autonom: np.ndarray) -> None:
        # check if there are any internal interactions and warn if so
        autonom_rels = rels_matrix[autonom, :][:, autonom]
        if autonom_rels.any():
            warnings.warn('interactions within autonomous set')

    @staticmethod
    def _select_to_organify(cycle_labels: np.ndarray,
                            weights: np.ndarray,
                            ) -> Tuple[int, np.ndarray]:
        # component IDs that mark cycles
        cyclecomps = np.flatnonzero(np.bincount(cycle_labels) > 1)
        # which cycle which unit is in
        cycle_mem = np.where(np.isin(cycle_labels, cyclecomps), cycle_labels, -1)
        # select cycle with smallest weight
        sel_cycle_i = min(cyclecomps,
            key=lambda comp: weights[cycle_mem == comp].sum()
        )
        sel_cycle_mem = (cycle_mem == sel_cycle_i)
        # contract the system to the unit with maximum weight
        cycle_is = np.flatnonzero(sel_cycle_mem)
        head_i = cycle_is[weights[cycle_is].argmax()]
        other_is = np.array([i for i in cycle_is if i != head_i])
        return head_i, other_is

    @staticmethod
    def _contract_rels(rels_matrix: np.ndarray, selfrels: np.ndarray, head_i: int, other_is: np.ndarray) -> None:
        selfrels[head_i, head_i] += (
            selfrels[other_is].sum()
            + rels_matrix[other_is, head_i].sum()
            + rels_matrix[head_i, other_is].sum()
        )
        selfrels[other_is, other_is] = 0
        rels_matrix[:, head_i] += rels_matrix[:, other_is].sum(axis=1)
        rels_matrix[head_i, :] += rels_matrix[other_is, :].sum(axis=0)
        rels_matrix[:, other_is] = 0
        rels_matrix[other_is, :] = 0
        rels_matrix[other_is, head_i] = 1  # to ensure stable binding into the system
        rels_matrix[head_i, head_i] = 0

    @staticmethod
    def _contract_weights(weights: np.ndarray, head_i: int, other_is: np.ndarray) -> None:
        weights[head_i] += weights[other_is].sum()
        weights[other_is] = 0


class MaxflowBuilder(IterativeDecyclingBuilder):
    deterministic = True

    def select_root(self, rels_matrix: np.ndarray) -> int:
        return self._argmax(rels_matrix.sum(axis=1))

    def select_parents(self, rels_matrix: np.ndarray) -> np.ndarray:
        return self._argmax(rels_matrix, axis=1)

    @staticmethod
    def _argmax(array, axis=None):
        return array.argmax(axis=axis)


class StochasticMaxflowBuilder(MaxflowBuilder):
    deterministic = False

    @staticmethod
    def _argmax(array, axis=None):
        if len(array.shape) == 1:
            cum = array.cumsum()
            return (np.random.rand() <= (cum / cum[-1])).argmax()
        elif len(array.shape) == 2:
            cum = array.cumsum(axis=axis)
            guesses = np.random.rand(array.shape[1-axis])
            if axis == 1:
                guesses = guesses[:, np.newaxis]
            # if (cum[:, [-1]] == 0).any():
                # print(np.flatnonzero(cum[:, [-1]] == 0))
                # import scipy.sparse
                # print(scipy.sparse.coo_matrix(array))
                # raise NotImplementedError
            return (guesses <= (cum / cum[:, [-1]])).argmax(axis=axis)
        else:
            raise ValueError


class ImprovementBuilder:
    deterministic = True

    def __init__(self, criterion, builder_type=None, verbose: bool = False, **kwargs):
        self.criterion = criterion # fitness_criterion(criterion, **kwargs)
        self.verbose = verbose
        if builder_type is None:
            builder_type = MaxflowBuilder
        self.inner = builder_type(self.criterion, verbose=verbose, **kwargs)

    def build(self, rels: Relations, index: Optional[np.ndarray] = None) -> Model:
        logger.debug('maxflow k: %g', self.criterion.maxflow_k(rels))
        model = self.inner.build(rels, index=index)
        self.autonomize_weak_leaves(model, rels)
        model, crit = self.add_root_organic(model, rels)
        model = self.revisit_organic(model, rels, crit=crit)
        if self.verbose:
            logger.debug('final nodality vals:\n%s', self.criterion.nodality(model, rels))
            logger.debug('final organicity vals:\n%s', self.criterion.symcoh(model, rels))
            logger.debug('final stage indicator vals:\n%s', self.criterion.stage_coefs(model, rels))
            logger.debug('final path lengths:\n%s', self.criterion.path_lengths(model))
            logger.debug('final path indicator vals:\n%s', self.criterion.tree_fits(model, rels))
        return model

    def autonomize_weak_leaves(self, model: Model, rels: Relations) -> None:
        nodalities = self.criterion.nodality(model, rels)
        crit_vals = self.criterion.evaluate_nodes(model, rels)
        leaves = model.df.query('not organic').drop(model.df.parent).index
        is_weak_leaf = (nodalities[leaves] < .5) & (crit_vals[leaves] < .5)
        weak_leaves = is_weak_leaf[is_weak_leaf].index
        n_weak_leaves = len(weak_leaves)
        if n_weak_leaves:
            logger.debug('setting %d leaves to autonomous', n_weak_leaves)
            model.df.loc[weak_leaves, model.STAGE_COL] = model.STAGE_AUTONOM
            model.df.loc[weak_leaves, model.PARENT_COL] = weak_leaves
            model.reset_cache()
        else:
            logger.debug('no weak leaves to autonomize (lowest nodality %g at %s)',
                          nodalities[leaves].min(), nodalities[leaves].idxmin())

    def add_root_organic(self, model: Model, rels: Relations) -> Tuple[Model, float]:
        logger.debug('organifying root')
        root_id = model.root_id()
        root_ids = [root_id]
        logger.debug('evaluating initial solution')
        crit = self.criterion.evaluate(model, rels)
        logger.debug('initial fitness: %g', crit)
        while True:
            logger.debug('computing organic candidates')
            max_desc_id = self.best_organic_candidate_id(model, rels, root_ids)
            logger.debug('testing organification of %s', max_desc_id)
            # try adding this maximum descendant as organic, including its organic subs
            orged_model = model.copy()
            orged_model.df.loc[max_desc_id, model.STAGE_COL] = model.STAGE_ORG
            # rewire descendants of organified to root
            orged_model.df.loc[orged_model.df[model.PARENT_COL] == max_desc_id, model.PARENT_COL] = root_id
            orged_crit = self.criterion.evaluate(orged_model, rels)
            if orged_crit > crit:
                logger.debug('accepting %s as organic sub of root (new fitness: %g)', max_desc_id, orged_crit)
                model = orged_model
                crit = orged_crit
                root_ids.append(max_desc_id)
            else:
                logger.debug('%s not accepted (fitness %g), terminating root organification', max_desc_id, orged_crit)
                break
        return model, crit

    def best_organic_candidate_id(self, model: Model, rels: Relations, to_ids: List[Any]) -> Any:
        # get total inflows and outflows of root incl. organics
        to_is = model.indices_of(to_ids)
        tgt_inflows = rels.matrix[:,to_is].sum(axis=1)
        tgt_outflows = rels.matrix[to_is,:].sum(axis=0)
        tot_weights = rels.outsums.copy()
        org_sub_is = model.indices_of(model.index[model.organics])
        org_head_is = model.indices_of(model.parents.iloc[org_sub_is])
        tgt_inflows[org_head_is] += tgt_inflows[org_sub_is]
        tgt_outflows[org_head_is] += tgt_outflows[org_sub_is]
        tot_weights[org_head_is] += tot_weights[org_sub_is]
        # all direct nodal descendants of root
        # use min(fij, fji) / sqrt(Fi, Fj) - linearly dependent on organicity measure
        dir_desc_rels = (
            (model.parents.isin(to_ids) & ~model.organics)
            * np.minimum(tgt_inflows, tgt_outflows)
        ) / np.sqrt(tot_weights[to_is].sum() * tot_weights)
        dir_desc_rels[to_ids] = 0
        return dir_desc_rels.idxmax()

    def revisit_organic(self, model: Model, rels: Relations, crit: Optional[float] = None) -> Model:
        weights = pd.Series(rels.weights, index=model.df.index, name='weight')
        stable = False
        while not stable:
            logging.debug('entering deorganify round')
            stable = True
            to_revisit = model.df.join(weights)[model.organics].sort_values('weight').index.tolist()
            if crit is None:
                logging.debug('evaluating initial solution')
                crit = self.criterion.evaluate(model, rels)
            logging.debug('initial fitness: %g, %d organics to revisit', crit, len(to_revisit))
            for sub_id in to_revisit:
                logging.debug('checking organic %s', sub_id)
                nodalized = self.nodalize(model, sub_id, rels)
                new_crit = self.criterion.evaluate(nodalized, rels)
                if new_crit > crit:
                    logger.debug('deorganifying %s: %g -> %g', sub_id, crit, new_crit)
                    model = nodalized
                    crit = new_crit
                    stable = False
                else:
                    logger.debug('%s passed: %g > %g', sub_id, crit, new_crit)
            if stable:
                logging.debug('all organics passed, terminating')
        return model

    def nodalize(self, model: Model, sub_id: Any, rels: Relations) -> Model:
        # find out which units are dependent on sub_id's parent
        parent_id = model.df.loc[sub_id, model.PARENT_COL]
        all_deps = model.parents == parent_id
        dep_ids = model.df[all_deps & ~model.organics].index
        if len(dep_ids):
            other_ids = model.df[all_deps & model.organics].index.tolist()
            other_ids.remove(sub_id)
            other_ids.append(parent_id)
            dep_is = model.indices_of(dep_ids)
            # move those that have larger links to sub_id than to the delinked parent
            # (and possibly, rest of its orgsys)
            dep_flow = rels.matrix[dep_is, :]
            flow_to_sub = dep_flow[:, model.index_of(sub_id)]
            flow_to_rest = dep_flow[:, model.indices_of(other_ids)]
            if flow_to_rest.ndim == 2:
                flow_to_rest = flow_to_rest.sum(axis=1)
            move_to_sub = dep_ids[flow_to_sub > flow_to_rest]
        else:
            move_to_sub = []
        # nodalize the sub_id's relationship
        nodalized = model.copy()
        nodalized.df.loc[sub_id, nodalized.STAGE_COL] = nodalized.STAGE_NODAL
        # give sub_id's its share of dependents
        nodalized.df.loc[move_to_sub, nodalized.PARENT_COL] = sub_id
        return nodalized


MODEL_BUILDERS: Dict[str, type] = {
    'maxflow': MaxflowBuilder,
    'improvement': ImprovementBuilder,
    'stochastic': StochasticMaxflowBuilder,
}


def create(interaction_df: pd.DataFrame,
           save_weights: bool = True,
           save_evaluations: bool = True,
           criterion: str = 'stagestate',
           builder: str = 'maxflow',
           seed: Optional[int] = None,
           **kwargs) -> pd.DataFrame:
    rels, ids = Relations.from_dataframe(
        interaction_df, *interaction_df.columns[:3]
    )
    criterion = fitness_criterion(criterion)
    builder = model_builder(builder, criterion=criterion, **kwargs)
    if seed is not None:
        np.random.seed(seed)
    model = builder.build(rels, index=ids)
    out_df = model.df.copy().rename_axis(index='id').reset_index()
    if save_weights:
        out_df['weight'] = rels.weights
    if save_evaluations:
        node_eval = criterion.evaluate_nodes(model, rels)
        out_df['criterion'] = node_eval
    return out_df


def regional_succession(model: Model,
                        weights: Optional[pd.Series] = None,
                        ) -> Tuple[pd.Series, pd.Series]: # regions, region weights
    if weights is None:
        weights = pd.Series(1, index=model.index)
    nonorg = model.df[~model.organics].index
    assigs = pd.Series(nonorg, index=nonorg)
    root_id = model.root_id()
    stop_weights = weights.loc[nonorg].copy()
    active = model.df[model.hier][[model.PARENT_COL]].drop(root_id).join(weights.rename('weight'))
    while len(active):
        assigs = assigs.append(active[model.PARENT_COL])
        add_weights = active.groupby(model.PARENT_COL)['weight'].sum()
        stop_weights = stop_weights.add(add_weights, fill_value=0)
        active = (
            active
            .query(f'{model.PARENT_COL} != {root_id!r}')
            .rename(columns={model.PARENT_COL: f'old_{model.PARENT_COL}'})
            .join(model.parents, on=f'old_{model.PARENT_COL}')
            .drop(f'old_{model.PARENT_COL}', axis=1)
        )
    return assigs, stop_weights.astype(weights.dtype).rename(weights.name)
