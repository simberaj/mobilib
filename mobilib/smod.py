
from __future__ import annotations

import random
from typing import List, Dict, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd
import scipy.sparse

import mobilib.markov


def components_by_targets(targets: np.ndarray) -> Optional[np.ndarray]:
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
        connection='strong',
        return_labels=True
    )
    return None if n_strong == n else labels


def autonomous_subsystems(matrix: np.ndarray) -> Tuple[int, np.ndarray]:
    return scipy.sparse.csgraph.connected_components(
        matrix,
        directed=True,
        connection='weak',
        return_labels=True
    )


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
        rels_w = (self.matrix - np.diag(np.diag(self.matrix)))
        rels_w_sums = rels_w.sum(axis=1)
        self.weighting = rels_w * (
            self.outsums / np.where(rels_w_sums, rels_w_sums, 1)
        )[:, np.newaxis] / self.matrix.sum()

    def _correct_matrix(self, matrix: np.ndarray) -> np.ndarray:
        # add ones on the diagonal for absorbing states
        no_outflows = (matrix.sum(axis=1) == 0)
        return matrix + np.diag(no_outflows)

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


class StagedModel:
    PARENT_KEY = 'parent'
    ORG_KEY = 'organic'
    INDEXCOL_KEY = 'id'

    def __init__(self, df: pd.DataFrame):
        assert self.PARENT_KEY in df.columns
        assert self.ORG_KEY in df.columns
        self.df = df
    
    @property
    def n(self):
        return len(self.df)

    @property
    def index(self):
        return self.df.index

    @property
    def parents(self):
        return self.df[self.PARENT_KEY]

    def parent_is(self):
        return self.df.index.get_indexer(self.df[self.PARENT_KEY])

    @property
    def organics(self):
        return self.df[self.ORG_KEY]

    @property
    def n_roots(self) -> int:
        return (self.df[self.PARENT_KEY] == self.df.index).sum()

    def is_root(self, key: Any) -> bool:
        return self.df.loc[key, self.PARENT_KEY] == key

    def is_valid(self) -> bool:
        try:
            self.check()
            return True
        except AssertionError:
            return False

    def check(self) -> None:
        # conditions: no cycles, targets of organic bindings must not have an organic binding
        assert components_by_targets(self.parent_is()) is None, 'cyclic parental relationship'
        assert ~self.organics[self.parents].all(), 'dependence on non-primary unit of organic subsystem'

    @classmethod
    def from_arrays(cls,
                    parents: np.ndarray,
                    organics: np.ndarray,
                    index: Optional[np.ndarray] = None,
                    ) -> StagedModel:
        return cls(pd.DataFrame({
            cls.PARENT_KEY: parents,
            cls.ORG_KEY: organics,
        }, index=index))

    def binding_matrix(self, rels: Relations) -> np.ndarray:
        bmatrix = np.diag(np.diag(rels.matrix))
        conn_df = self.connection_df()
        from_is = self.index.get_indexer(conn_df[self.INDEXCOL_KEY])
        to_is = self.index.get_indexer(conn_df[self.PARENT_KEY])
        conn_values_along = rels.matrix[from_is, to_is]
        conn_values_cnter = rels.matrix[to_is, from_is]
        conn_df['value'] = np.where(
            conn_df[self.ORG_KEY],
            np.minimum(conn_values_along, conn_values_cnter),
            conn_values_along
        )
        # # Equalize values of outflows from organic systems so that they are proportional
        # # to the total outflow size of the constituent units.
        # from_org_conns = conn_df[conn_df['from_organic']].copy()
        # from_org_conns['out_sum'] = rels.outsums[from_is[from_org_conns.index]]
        # conn_df.loc[from_org_conns.index,'value'] = from_org_conns.join(
            # from_org_conns.groupby(self.PARENT_KEY)[['value', 'out_sum']].sum(),
            # on=self.PARENT_KEY, rsuffix='_tot'
        # ).eval('value_tot * out_sum / out_sum_tot')
        # # Equalize values of inflows to organic systems so that they are proportional
        # # to the total inflow size of the constituent units.
        # to_org_conns = conn_df[conn_df['to_organic'] & ~conn_df[self.ORG_KEY]].copy()
        # to_org_conns['in_sum'] = rels.insums[self.index.get_indexer(to_org_conns[self.PARENT_KEY])]
        # conn_df.loc[to_org_conns.index,'value'] = to_org_conns.join(
            # to_org_conns.groupby(self.INDEXCOL_KEY)[['value', 'in_sum']].sum(),
            # on=self.INDEXCOL_KEY, rsuffix='_tot'
        # ).eval('value_tot * in_sum / in_sum_tot')
        # # We might have produced some floats even if original was integer, thus need to cast.
        # bmatrix = bmatrix.astype(conn_df['value'].dtype)
        # Paste the values symetrically.
        bmatrix[from_is, to_is] = conn_df['value']
        bmatrix[to_is, from_is] = conn_df['value']
        return bmatrix
    
    def connection_df(self) -> pd.DataFrame:
        conn_df = self.df.rename_axis(index=self.INDEXCOL_KEY).reset_index().query(
            f'{self.INDEXCOL_KEY} != {self.PARENT_KEY}'
        )
        print(conn_df)
        if self.organics.any():
            return self._add_organic_connections(conn_df)
        else:
            return conn_df.assign(from_organic=False, to_organic=False)

    def _add_organic_connections(self, conn_df: pd.DataFrame) -> pd.DataFrame:
        org_conn_df = conn_df.query(self.ORG_KEY)
        # For units bound to organic subsystems, add connections to their nonprimary units.
        conn_add_simples_df = self._join_on_parent(
            conn_df.query(f'not {self.ORG_KEY}'), org_conn_df
        )
        # For nonprimary units in organic subsystems, add connections to other nonprimary units.
        conn_add_org_head_df = self._join_on_parent(org_conn_df, org_conn_df).query(
            f'{self.INDEXCOL_KEY} != {self.PARENT_KEY}'
        )
        # For nonprimary units in organic subsystems, add connections to their head's parent.
        conn_add_head_tgt_df = org_conn_df.merge(
            pd.concat([conn_df, conn_add_simples_df]),
            left_on=self.PARENT_KEY,
            right_on=self.INDEXCOL_KEY,
            suffixes=('_src', '')
        )[[f'{self.INDEXCOL_KEY}_src', self.PARENT_KEY, self.ORG_KEY]].rename(
            columns={f'{self.INDEXCOL_KEY}_src': self.INDEXCOL_KEY}
        )
        # from_organic: outgoing hierarchical connection from an organic system
        all_conn_df = pd.concat([
            # conn_df.assign(from_organic=False, to_organic=False),
            # conn_add_simples_df.assign(from_organic=False, to_organic=True),
            # conn_add_org_head_df.assign(from_organic=False, to_organic=False, **{self.ORG_KEY: True}),
            # conn_add_head_tgt_df.assign(from_organic=True, to_organic=False),
            conn_df,
            conn_add_simples_df,
            conn_add_org_head_df.assign(**{self.ORG_KEY: True}),
            conn_add_head_tgt_df,
        ], ignore_index=True)
        # org_heads = org_conn_df[self.PARENT_KEY].unique()
        # all_conn_df.loc[all_conn_df[self.INDEXCOL_KEY].isin(org_heads),'from_organic'] = True
        # all_conn_df.loc[all_conn_df[self.PARENT_KEY].isin(org_heads),'to_organic'] = True
        return all_conn_df

    def _join_on_parent(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        indexcol_key_tgt = f'{self.INDEXCOL_KEY}_tgt'
        return df1.merge(
            df2, on=self.PARENT_KEY, suffixes=('', '_tgt')
        )[[self.INDEXCOL_KEY, indexcol_key_tgt, self.ORG_KEY]].rename(
            columns={indexcol_key_tgt: self.PARENT_KEY}
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

    def copy(self) -> StagedModel:
        return StagedModel(self.df.copy())


class FitnessCriterion:
    def evaluate(self, model: StagedModel, rels: Relations) -> float:
        raise NotImplementedError

    def evaluate_nodes(self, model: StagedModel, rels: Relations) -> pd.Series:
        raise NotImplementedError


def fitness_criterion(name: str, **kwargs) -> FitnessCriterion:
    return FITNESS_CRITERIA[name](**kwargs)


class MFPTCriterion:
    def evaluate(self, model: StagedModel, rels: Relations) -> float:
        return self.evaluate_nodes(model, rels).sum()

    def evaluate_nodes(self, model: StagedModel, rels: Relations) -> pd.Series:
        model_edge_mfpt = mobilib.markov.componental_mfpt(
            model.binding_matrix(rels), directed=True, connection='strong'
        )
        return pd.Series(
            (rels.weighting / model_edge_mfpt).sum(axis=1),
            index=model.index
        )

    @staticmethod
    def _times(bind_matrix: np.ndarray) -> np.ndarray:
        n_comps, comp_labels = scipy.sparse.csgraph.connected_components(
            bind_matrix,
            directed=True,
            connection='strong'
        )
        hier_trans = mobilib.markov.transition_matrix(bind_matrix)
        times = np.full_like(hier_trans, fill_value=np.inf)
        absorbing = np.isclose(np.diag(hier_trans), 1)
        # for each autonomous subsystem
        for comp_i in range(n_comps):
            is_comp = (comp_labels == comp_i)
            absorbing_i = np.flatnonzero(absorbing & is_comp)
            nonabsorbing_i = np.flatnonzero(~absorbing & is_comp)
            times[nonabsorbing_i[:, None], nonabsorbing_i] = mobilib.markov.mfpt(
                hier_trans[nonabsorbing_i[:, None], nonabsorbing_i]
            )
            times[absorbing_i, absorbing_i] = 1
        return times


FITNESS_CRITERIA: Dict[str, type] = {
    'mfpt': MFPTCriterion,
}


class StagedModelBuilder:
    deterministic: bool = True

    def build(self, rels: Relations, index: Optional[np.ndarray] = None) -> StagedModel:
        raise NotImplementedError


def model_builder(name: str, **kwargs) -> StagedModelBuilder:
    return MODEL_BUILDERS[name](**kwargs)


class IterativeDecyclingBuilder(StagedModelBuilder):
    def __init__(self, criterion: Optional[FitnessCriterion] = None):
        pass

    def build(self, rels: Relations, index: Optional[np.ndarray] = None) -> StagedModel:
        ncomps, compmem = autonomous_subsystems(rels.matrix)
        relmatrix = rels.matrix - np.diag(rels.selfrels)
        parents = np.empty_like(rels.weights, dtype=np.int64)
        organicity = np.zeros_like(parents, dtype=bool)
        for comp_i in range(ncomps):
            # for each autonomous subsystem
            in_comp = compmem == comp_i
            comprels = relmatrix[in_comp, :][:, in_comp].copy()
            if comprels.size == 1:
                in_comp_i = np.flatnonzero(in_comp)[0]
                parents[in_comp_i] = in_comp_i
                continue
            compweights = rels.weights[in_comp].copy()
            comporgs = np.zeros_like(compweights, dtype=bool)
            while True:
                root_i = self.select_root(comprels)
                # print('root at', index[in_comp][root_i])
                targets = self.select_parents(comprels)
                targets[root_i] = root_i
                # find strongly connected components in the directed hierarchy graph,
                # these will be cycles
                cycle_labels = components_by_targets(targets)
                if cycle_labels is None:
                    # no connected components = no cycles, model is complete
                    break
                # component IDs that mark cycles
                cyclecomps = np.flatnonzero(np.bincount(cycle_labels) > 1)
                # which cycle which unit is in
                cycle_mem = np.where(np.isin(cycle_labels, cyclecomps), cycle_labels, -1)
                # select cycle with smallest weight
                sel_cycle_i = min(cyclecomps,
                    key=lambda comp: compweights[cycle_mem == comp].sum()
                )
                sel_cycle_mem = (cycle_mem == sel_cycle_i)
                # contract the system to the unit with maximum weight
                cycle_is = np.flatnonzero(sel_cycle_mem)
                main_i = cycle_is[compweights[cycle_is].argmax()]
                other_is = np.array([i for i in cycle_is if i != main_i])
                # print('contracting cycle', index[in_comp][sel_cycle_mem], index[main_i], index[other_is])
                comporgs[other_is] = True
                # contract weights and relations
                self._contract_weights(compweights, main_i, other_is)
                self._contract_rels(comprels, main_i, other_is)
            in_comp_is = np.flatnonzero(in_comp)
            parents[in_comp_is] = in_comp_is[targets]
            organicity[in_comp_is[comporgs]] = True
            organicity[in_comp_is[root_i]] = False
        if index is not None:
            parents = index[parents]
        model = StagedModel.from_arrays(parents, organicity, index=index)
        # with pd.option_context('display.max_rows', 100):
        #     print(model.df.reset_index())
        return model
    
    @staticmethod
    def _contract_rels(rels_matrix: np.ndarray, main_i: int, other_is: np.ndarray) -> None:
        rels_matrix[:, main_i] += rels_matrix[:, other_is].sum(axis=1)
        rels_matrix[main_i, :] += rels_matrix[other_is, :].sum(axis=0)
        rels_matrix[:, other_is] = 0
        rels_matrix[other_is, :] = 0
        rels_matrix[other_is, main_i] = 1  # to ensure stable binding into the system
        rels_matrix[main_i, main_i] = 0

    @staticmethod
    def _contract_weights(weights: np.ndarray, main_i: int, other_is: np.ndarray) -> None:
        weights[main_i] += weights[other_is].sum()
        weights[other_is] = 0


class HeuristicBuilder(IterativeDecyclingBuilder):
    deterministic = True

    def select_root(self, rels_matrix: np.ndarray) -> int:
        mfpt = mobilib.markov.mfpt(mobilib.markov.transition_matrix(rels_matrix))
        out_sums = rels_matrix.sum(axis=1)
        return (out_sums[:,np.newaxis] / mfpt).sum(axis=0).argmax()

    def select_parents(self, rels_matrix: np.ndarray) -> np.ndarray:
        src_sums = rels_matrix.sum(axis=1)[np.newaxis,:]
        tgt_sums = rels_matrix.sum(axis=0)[:,np.newaxis]
        crit = (
            rels_matrix / np.where(src_sums == 0, 1, src_sums)
            + rels_matrix / np.where(tgt_sums == 0, 1, tgt_sums)
        )
        return crit.argmax(axis=1)


class MaxflowBuilder(IterativeDecyclingBuilder):
    deterministic = True

    def select_root(self, rels_matrix: np.ndarray) -> int:
        return self._argmax(comprels.sum(axis=0))

    def select_parents(self, rels_matrix: np.ndarray) -> np.ndarray:
        return self._argmax(comprels, axis=1)

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


class ModelOperator:
    def __init__(self, tries: int = 20):
        self.tries = tries

    @staticmethod
    def random_heavy(rels: Relations, index: pd.Index) -> Any:
        return index[random_weighted(rels.cum_weights)]


class ModelCrossover(ModelOperator):
    def crossover(self,
                  model1: StagedModel,
                  model2: StagedModel,
                  rels: Relations) -> StagedModel:
        """Crossover two staged settlement system models.

        Works by selecting a suitably large (*takeover*) node from model2 and transposing the settings of all its
        descendant nodes into model1.

        Validity of the output model is ensured the following way:

        -   Parental cycles are prevented by discarding takeover nodes whose parent in model1 is among the descendants
            of the takeover node in model2.
        -   Nested organic systems would only occur if the takeover node is a non-primary node of an organic system
            in model1; therefore, if that is the case, all direct descendants of the takeover node are reassigned to
            the primary node of that system before implanting.
        """
        assert model1.n == model2.n
        takeover_key = model1.index[0]
        implant_keys = [model1.parents[takeover_key]]
        for i in range(self.tries):
            if implant_keys and model1.parents[takeover_key] not in implant_keys:
                break
            takeover_key = self._select_crossover_point(model2, rels)
            implant_keys = model2.descendant_keys(takeover_key)
        else:
            return model1.copy()
        implants = model2.df.loc[implant_keys, :]
        if model1.organics[takeover_key]:
            # If the takeover head node is part of organic system, we need to redirect its direct descendants to
            # point to the main node of that system.
            implants = implants.copy()
            implants.loc[implants[model2.PARENT_KEY] == takeover_key, model2.PARENT_KEY] = model1.parents[takeover_key]
        crossed = self._fix_cycles(model1, takeover_key, implant_keys)
        crossed.df.loc[implant_keys, :] = implants
        # try:
        #     crossed.check()
        # except AssertionError:
        #     print(model1.df)
        #     print(model1.is_valid())
        #     print(model2.df)
        #     print(model2.is_valid())
        #     print(takeover_i, implant_inds)
        #     print(crossed.df)
        #     raise
        return crossed

    def _select_crossover_point(self, model: StagedModel, rels: Relations) -> int:
        while True:
            crosspoint = self.random_heavy(rels, model.index)
            if not model.organics[crosspoint] and not (model.is_root(crosspoint) and model.n_roots == 1):
                break
        return crosspoint

    @staticmethod
    def _fix_cycles(model: StagedModel, takeover_key: Any, implant_keys: List[Any]) -> StagedModel:
        model = model.copy()
        current_key = takeover_key
        parent_key = model.parents[current_key]
        while parent_key != current_key:    # until we reach the root
            if parent_key in implant_keys:
                # Break the cyclical relationship leading from one of takeover_i's predecessors to its descendants.
                model.df.loc[current_key, model.PARENT_KEY] = current_key
                break
            current_key = parent_key
            parent_key = model.parents[current_key]
        return model


class ModelModifier(ModelOperator):
    def modify(self, model: StagedModel, rels: Relations) -> StagedModel:
        raise NotImplementedError

    def random_simple(self, model: StagedModel) -> Optional[Any]:
        for i in range(self.tries):
            unit = model.index[random.randrange(0, model.n)]
            if not (model.organics[unit] or model.is_root(unit)):
                return unit
        return None


class RootMover(ModelModifier):
    """Randomly select a large node and make it the root of its autonomous subsystem.

    Inverts the hierarchical relationships on the pathway from the new root to the old one.
    """
    def modify(self, model: StagedModel, rels: Relations) -> StagedModel:
        new_model = model.copy()
        new_root = self._select_new_root(model, rels)
        # If no suitable root candidate was found, return the model unchanged.
        if new_root is None:
            return new_model
        # Invert the pathway from the old root to the new one.
        prev_node = new_root
        cur_node = new_model.parents[new_root]
        changes = [new_root]
        new_parents = [new_root]
        i = 0
        while not new_model.is_root(cur_node):
            next_node = new_model.parents[cur_node]
            changes.append(cur_node)
            new_parents.append(prev_node)
            prev_node = cur_node
            cur_node = next_node
            i += 1
            if i > 100:
                print(model.df)
                raise RuntimeError
        changes.append(cur_node)
        new_parents.append(prev_node)
        new_model.df.loc[changes, new_model.PARENT_KEY] = new_parents
        return new_model

    def _select_new_root(self, model: StagedModel, rels: Relations) -> Optional[Any]:
        for i in range(self.tries):
            # Select a random important node.
            new_root = self.random_heavy(rels, model.index)
            # If the unit is actually a non-primary in an organic system, select the primary of that system.
            if model.organics[new_root]:
                new_root = model.parents[new_root]
            # Repeat until we have a non-root node.
            if not model.is_root(new_root):
                return new_root
        return None


class ParentShifter(ModelModifier):
    """Randomly change the parent of a random node that is neither a root nor a non-primary unit of an organic system.

    The random selection of the parent is weighted by interactions of the node.
    """
    def modify(self, model: StagedModel, rels: Relations) -> StagedModel:
        new_model = model.copy()
        # Select a random node that is neither a root nor a non-primary in an organic system.
        change_node = self.random_simple(model)
        if change_node is None:
            return new_model
        change_descs = model.descendant_keys(change_node)
        current_parent = model.parents[change_node]
        # Randomly select a new parent (weighting by interaction strength).
        # Avoid selecting self, the current parent or a descendant of self.
        cum_weights = rels.transition_probs[model.index.get_loc(change_node)].cumsum()
        parent = change_node
        disallowed_parents = change_descs + [change_node, current_parent]
        for i in range(self.tries):
            if parent not in disallowed_parents:
                break
            parent = model.index[random_weighted(cum_weights)]
            # If the selected parent is a non-primary in an organic subsystem, select its subsystem head instead.
            if model.organics[parent]:
                parent = model.parents[parent]
        else:
            return new_model    # if we did not manage to find a suitable new parent
        new_model.df.loc[change_node, new_model.PARENT_KEY] = parent
        return new_model


class Organifier(ModelModifier):
    """Randomly make one non-organic relationship organic."""
    def modify(self, model: StagedModel, rels: Relations) -> StagedModel:
        new_model = model.copy()
        # Select a random node that is neither a root nor a non-primary in an organic system.
        unit = self.random_simple(model)
        if unit is None:
            return new_model
        parent = model.parents[unit]
        new_model.df.loc[unit, new_model.ORG_KEY] = True
        # Reassign children of the unit to the organic primary.
        new_model.df.loc[new_model.parents == unit, new_model.PARENT_KEY] = parent
        return new_model


class Deorganifier(ModelModifier):
    """Randomly make one organic relationship non-organic.

    Distributes the children of the former system randomly, weighting by their relation to both resulting parts.
    """
    def modify(self, model: StagedModel, rels: Relations) -> StagedModel:
        new_model = model.copy()
        # Select a random organic non-primary to deorganify.
        org_keys = model.index[model.organics]
        if not len(org_keys):
            return new_model
        tgt_key = random.choice(org_keys)
        # Get other organically (all_other_nonprim) and non-organically (all_subord) bound elements to the primary.
        parent = model.parents[tgt_key]
        subord_flags = (model.parents == parent)
        subord_flags[parent] = False
        all_other_nonprim = list(model.index[subord_flags & model.organics].tolist())
        all_other_nonprim.remove(tgt_key)
        # print(model.index, parent, all_other_nonprim)
        all_other_is = model.index.get_indexer([parent] + all_other_nonprim)
        all_subord_is = np.flatnonzero(subord_flags & ~model.organics)
        if len(all_subord_is):
            # Reassign some non-organically bound subordinates randomly to the target instead of the organic subsystem.
            weights_other = rels.matrix[all_subord_is][:, all_other_is].sum(axis=1)
            weights_tgt = rels.matrix[all_subord_is, model.index.get_loc(tgt_key)].flatten()
            weights_tgt = weights_tgt / (weights_tgt + weights_other)
            assign_to_tgt = model.index[all_subord_is[np.random.rand(len(weights_tgt)) < weights_tgt]]
            new_model.df.loc[assign_to_tgt, new_model.PARENT_KEY] = tgt_key
        # Unlink the target from the organic subsystem.
        new_model.df.loc[tgt_key, new_model.ORG_KEY] = False
        return new_model


MODEL_MODIFIERS: List[type] = [
    RootMover,
    ParentShifter,
    Organifier,
    Deorganifier,
]


class GeneticBuilder:
    deterministic = False

    population: int = 50
    crossover_rate: float = 1.
    mutation_rate: float = 1.
    elitism_rate: float = 0.05
    stability_termination: int = 20
    max_generations: int = 100
    constructer: StagedModelBuilder = StochasticMaxflowBuilder()
    crosser: ModelCrossover = ModelCrossover()
    criterion: FitnessCriterion = MFPTCriterion()
    mutators: List[ModelModifier] = [mut() for mut in MODEL_MODIFIERS]
    mutator_weights: List[float] = [1, 10, 2, 2]

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        msum = sum(self.mutator_weights)
        self.mutator_probs = [w / msum for w in self.mutator_weights]

    def build(self, rels: Relations, *args, **kwargs) -> StagedModel:
        solutions = [
            self.constructer.build(rels, *args, **kwargs)
            for i in range(self.population)
        ]
        fitnesses = self.evaluate(solutions, rels)
        maxfit = fitnesses.max()
        stable_generations = 0
        for gen_i in range(self.max_generations):
            print(gen_i, maxfit)
            solutions.extend(self.crossover(solutions, rels))
            solutions.extend(self.mutate(solutions, rels))
            fitnesses = self.evaluate(solutions, rels, fitnesses)
            solutions, fitnesses = self.select(solutions, fitnesses)
            prev_maxfit = maxfit
            maxfit = fitnesses.max()
            if maxfit == prev_maxfit:
                stable_generations += 1
                if stable_generations >= self.stability_termination:
                    break
            else:
                stable_generations = 0
        return solutions[fitnesses.argmax()]

    def evaluate(self,
                 solutions: List[StagedModel],
                 rels: Relations,
                 fitnesses: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
        if fitnesses is None:
            fitnesses = []
        for sol in solutions:
            try:
                sol.check()
            except AssertionError:
                print(sol.df)
                raise
        return np.concatenate((fitnesses, np.array([
            self.criterion.evaluate(sol, rels)
            for sol in solutions[len(fitnesses):]
        ])))

    def crossover(self, source: List[StagedModel], rels: Relations) -> Iterable[StagedModel]:
        for i in range(int(self.crossover_rate * self.population)):
            sol1, sol2 = np.random.choice(source, 2, replace=False)
            yield self.crosser.crossover(sol1, sol2, rels)

    def mutate(self, source: List[StagedModel], rels: Relations) -> Iterable[StagedModel]:
        for i in range(int(self.mutation_rate * self.population)):
            oper = np.random.choice(self.mutators, 1, p=self.mutator_probs)[0]
            chosen = np.random.choice(source, 1)[0]
            yield oper.modify(chosen, rels)

    def select(self, solutions: List[StagedModel], fitnesses: np.ndarray) -> Tuple[List[StagedModel], np.ndarray]:
        n_elit = int(self.elitism_rate * self.population)
        elitist_is = np.argsort(fitnesses)[-n_elit:]
        others = np.ones(len(solutions), dtype=bool)
        others[elitist_is] = False
        scores = np.random.rand(len(solutions)) * fitnesses * others
        all_is = (
            list(elitist_is)
            + list(np.argsort(scores)[-(self.population - len(elitist_is)):])
        )
        return [solutions[i] for i in all_is], fitnesses[all_is]


MODEL_BUILDERS: Dict[str, type] = {
    'maxflow': MaxflowBuilder,
    'stochastic': StochasticMaxflowBuilder,
    'genetic': GeneticBuilder,
    'heuristic': HeuristicBuilder,
}
