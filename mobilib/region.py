
import numpy

from . import graph

class Relations:
    def __init__(self, matrix, weights=None):
        self.matrix = matrix
        assert len(set(self.matrix.shape)) == 1
        self.n = self.matrix.shape[0]
        self.outsums = matrix.sum(axis=1)
        self.insums = matrix.sum(axis=0)
        self.weights = weights if weights is not None else self.outsums
        self.totweight = self.weights.sum()
        self.unit_weights = self.weights / self.totweight
        self.selfrels = numpy.diag(matrix)
        self.outsums_noself = self.outsums - self.selfrels
        self.transition_probs = self.matrix / self.outsums[:,numpy.newaxis]
        self.selfprobs = numpy.diag(self.transition_probs)

    def weighted_sum(self, items):
        return (items * self.unit_weights).sum()

    @classmethod
    def from_dataframe(cls, df, from_id_col, to_id_col, strength_col):
        all_ids = numpy.array(list(sorted(set(
            list(df[from_id_col].unique())
            + list(df[to_id_col].unique())
        ))))
        n = len(all_ids)
        matrix = numpy.zeros((n,n), dtype=df[strength_col].dtype)
        from_ids = numpy.searchsorted(all_ids, df[from_id_col])
        to_ids = numpy.searchsorted(all_ids, df[to_id_col])
        matrix[from_ids,to_ids] = df[strength_col].values
        return cls(matrix), all_ids


class Hierarchy:
    def __init__(self, root, n, complete=True):
        self.root = root
        self.n = n
        self.flat = [root]
        self.flat.extend(root.descendants())
        self.organics = [item for item in self.flat if item.is_organic]
        self.elements_by_id = self._register_by_id()
        self.complete = complete
        if self.complete:
            self.organic_membership = self._compute_organics()
        else:
            self.organic_membership = numpy.empty(self.n, dtype=bool)
        self.change_made()

    def completed(self):
        self.complete = True
        self.organic_membership = self._compute_organics()

    def _compute_organics(self):
        return numpy.array([
            el.is_organic for el in self.elements_by_id
        ])

    def copy(self):
        return type(self)(self.root.copy(), self.n)

    def set_root(self, new_root):
        self.root = new_root
        self.root.subdue_parents()
        self.root.parent = None
        self.change_made()

    def register(self, element):
        self.flat.append(element)
        if element.is_organic:
            self.organics.append(element)
            self.organic_membership[element.id] = True
        self.elements_by_id[element.id] = element
        self.change_made()

    def deregister(self, element):
        self.flat.remove(element)
        if element.is_organic:
            self.organics.remove(element)
            self.organic_membership[element.id] = False
        self.elements_by_id[element.id] = None
        if element is self.root:
            self.root = None
        self.change_made()

    def refresh(self, element):
        former_positions = (self.elements_by_id == element)
        if element.is_organic:
            self.organic_membership[former_positions] = False
            self.organic_membership[element.id] = True
        self.elements_by_id[former_positions] = None
        self.elements_by_id[element.id] = element
        self.change_made()

    def has_valid_root(self):
        return (
            self.root is not None and
            set(self.elements_by_id[list(self.root.ids())]) != {None}
        )

    def change_made(self):
        self._organic_edges = None
        self._binding_matrix = None

    def structure_string(self):
        return '\n'.join(self.root.structure_lines(indent=0))

    def binding_matrix(self, rels):
        if self._binding_matrix is None:
            self._binding_matrix = numpy.full((self.n,self.n), numpy.inf)
            self.root.weigh_bindings(rels)
            self.root.bind_paths(self._binding_matrix)
        return self._binding_matrix

    @property
    def organic_edges(self):
        if self._organic_edges is None:
            e = numpy.zeros((self.n,self.n), dtype=bool)
            for orgsys in self.organics:
                for mem in orgsys.id:
                    e[mem,orgsys.id] = True
            self._organic_edges = e
        return self._organic_edges

    def _register_by_id(self):
        mem = numpy.empty(self.n, dtype=object)
        for el in self.flat:
            mem[el.id] = el
        return mem

    @classmethod
    def create(cls, parents, organics=None, ids=None, root=None):
        n = parents.size
        indices = numpy.arange(n)
        if ids is None: ids = indices
        if organics is None: organics = numpy.zeros(n, dtype=bool)
        if root is None: root = numpy.flatnonzero(indices == parents)[0]
        nodes = [HierarchyNode(i, ids[i]) for i in indices]
        for i in indices:
            if organics[i]:
                node = nodes[i]
                partner_i = parents[i]
                partner = nodes[partner_i]
                subsystem = OrganicSubsystem.create(partner, node)
                nodes[i] = subsystem
                nodes[partner_i] = subsystem
                partner_parent = nodes[parents[partner_i]]
                partner_parent.discard_child(partner)
                partner_parent.add_child(subsystem)
            elif i != root:
                nodes[parents[i]].add_child(nodes[i])
        hier = cls(nodes[root], len(nodes))
        return hier


class HierarchyElement:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.binding_weight = None

    def weight(self, rels):
        return rels.unit_weights[self.id].sum()

    def dissolve(self):
        print('dissolving', self)
        parent = self.parent
        children = self.children[:]
        for child in children:
            self.discard_child(child)
        if parent:
            if children:
                print('reassigning', children, 'to', parent)
            for child in children:
                parent.add_child(child)
            parent.discard_child(self)

    def tree_weight(self, rels):
        return self.weight(rels) + sum(
            child.tree_weight(rels) for child in self.children
        )

    def tree_ids(self):
        yield from self.ids()
        for child in self.children:
            yield from child.tree_ids()

    def subdue_parents(self):
        if self.parent is not None:
            prev_parent = self.parent
            prev_parent.subdue_parents()
            self.add_child(prev_parent)
            prev_parent.discard_child(self)

    def has_predecessor(self, element):
        predec = self.parent
        while predec is not None:
            if predec is element:
                return True
            predec = predec.parent
        return False

    def weigh_bindings(self, rels):
        self.weigh_binding(rels)
        for child in self.children:
            child.weigh_bindings(rels)

    def bind_paths(self, matrix):
        if self.parent is not None:
            parent_binding = matrix[self.parent.id,:]
            if len(parent_binding.shape) == 2:
                parent_binding = parent_binding[0]
            if self.binding_weight == 0:
                binding_distance = parent_binding * numpy.inf
            else:
                binding_distance = parent_binding / self.binding_weight
            self._record_bind_paths(matrix, binding_distance)
        self._record_self_bind(matrix)
        for child in self.children:
            child.bind_paths(matrix)

    def structure_lines(self, indent):
        yield ' ' * indent + self.name # + (
            # (' -> ' + str(self.binding_weight))
                # if self.binding_weight is not None else ''
        # )
        for child in self.children:
            yield from child.structure_lines(indent+2)

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        if child not in self.children:
            self.children.append(child)
            child.set_parent(self)

    def discard_child(self, child):
        if child in self.children:
            self.children.remove(child)
            child.set_parent(None)

    def weigh_binding(self, rels):
        if self.parent is not None:
            self.binding_weight = self._extract_binding_weight(rels)

    def predecessors(self):
        if self.parent:
            yield self.parent
            yield from self.parent.predecessors()

    def descendants(self):
        for child in self.children:
            yield child
            yield from child.descendants()

    def _with_copied_children(self, element):
        for child in self.children:
            element.add_child(child.copy())
        return element

    def _prune_children(self, ids):
        children = self.children[:]
        for child in children:
            child.prune(ids)


class HierarchyNode(HierarchyElement):
    is_organic = False

    def __init__(self, id, name, parent=None, subsystem=None):
        super().__init__(parent)
        self.id = id
        self.name = str(name)
        self.subsystem = subsystem

    def copy(self):
        return self._with_copied_children(
            type(self)(id=self.id, name=self.name)
        )

    def ids(self):
        yield self.id

    def add_child(self, child):
        if self.subsystem is None:
            super().add_child(child)
        else:
            self.subsystem.add_child(child)

    def set_subsystem(self, subsystem):
        self.subsystem = subsystem
        children = self.children
        self.children = []
        if self.parent:
            self.parent.discard_child(self)
        return children

    def unset_subsystem(self):
        self.subsystem = None

    def prune(self, ids):
        print('pruning', self)
        self._prune_children(ids)
        if self.id in ids:
            self.dissolve()

    def outward_transition_probabilities(self, rels):
        selfprob = rels.selfprobs[self.id]
        if selfprob == 1:
            return numpy.zeros(rels.n)
        else:
            probs = rels.transition_probs[self.id].copy()
            probs[self.id] = 0
            return probs / (1 - selfprob)

    def inflow_sum(self, rels):
        return rels.insums[self.id]

    def _extract_binding_weight(self, rels):
        selfprob = rels.selfprobs[self.id]
        if selfprob == 1:
            return 0
        else:
            return rels.transition_probs[self.id,self.parent.id].sum() / (1 - selfprob)

    def _record_bind_paths(self, matrix, binding):
        matrix[self.id,:] = matrix[:,self.id] = binding

    def _record_self_bind(self, matrix):
        matrix[self.id,self.id] = 1

    def __repr__(self):
        return '<Node({})>'.format(self.name)


class OrganicSubsystem(HierarchyElement):
    is_organic = True

    def __init__(self, members, parent=None):
        super().__init__(parent)
        self.members = []
        for member in members:
            self.add_member(member)

    def ids(self):
        yield from self.id

    def copy(self):
        return self._with_copied_children(
            type(self)([member.copy() for member in self.members])
        )

    def inflow_sum(self, rels):
        return rels.insums[self.id].sum()

    @property
    def id(self):
        return [member.id for member in self.members]

    @property
    def size(self):
        return len(self.members)

    def add_member(self, member):
        self.members.append(member)
        member_children = member.set_subsystem(self)
        for child in member_children:
            self.add_child(child)

    def remove_member(self, member):
        self.members.remove(member)
        member.unset_subsystem()

    def prune(self, ids):
        print('pruning', self)
        self._prune_children(ids)
        remove_ids = []
        remain_ids = []
        for id in self.id:
            (remove_ids if id in ids else remain_ids).append(id)
        print('ids:', remove_ids, remain_ids)
        if not remain_ids:
            self.dissolve()
        elif len(remain_ids) == 1:
            old_parent = self.parent
            remainer = self.member_by_id(remain_ids[0])
            print('replacing', self, 'with', remainer)
            self.remove_member(remainer)
            old_parent.add_child(remainer)
            old_parent.discard_child(self)
            remainer.add_child(self)
            self.dissolve()
        elif remove_ids:
            print('removing', remove_ids, 'from', self)
            for id in remove_ids:
                self.remove_member(self.member_by_id(id))

    def dissolve(self):
        for member in self.members:
            self.remove_member(member)
        return super().dissolve()

    def member_by_id(self, id):
        return self.members[self.id.index(id)]

    def merge(self, other):
        for mem in other.members:
            self.add_member(mem)
        for child in other.children:
            self.add_child(child)
        if other.parent:
            other.parent.discard_child(other)

    def outward_transition_probabilities(self, rels):
        weights = rels.weights[self.id]
        probs = (rels.transition_probs[self.id] * weights[:,numpy.newaxis]).sum(axis=0)
        probs[self.id] = 0
        probsum = probs.sum()
        if probsum == 0:
            return probs
        else:
            return probs / probs.sum()

    def _extract_binding_weight(self, rels):
        weights = rels.weights[self.id]
        parent_probs = rels.transition_probs[self.id][:,self.parent.id]
        if len(parent_probs.shape) == 2:
            parent_probs = parent_probs.sum(axis=1)
        for i, id in enumerate(self.id):
            selfprob = rels.transition_probs[id,self.id].sum()
            if not numpy.isclose(selfprob, 1):
                parent_probs[i] /= (1 - selfprob)
        return (parent_probs * weights).sum() / weights.sum()

    def _record_bind_paths(self, matrix, binding):
        matrix[self.id,:] = binding[numpy.newaxis,:]
        matrix[:,self.id] = binding[:,numpy.newaxis]

    def _record_self_bind(self, matrix):
        for mem in self.id:
            matrix[mem,self.id] = 1

    @property
    def name(self):
        return ' <> '.join(member.name for member in self.members)

    @classmethod
    def create(cls, item1, item2):
        if isinstance(item1, cls):
            item1.add_member(item2)
            return item1
        else:
            return cls([item1, item2])

    def __repr__(self):
        return '<OrgSubsys({})>'.format(self.name)



class Criterion:
    def evaluate(self, rels, hierarchy):
        raise NotImplementedError


class TransitionCriterion(Criterion):
    def __init__(self, organic_tolerance=1):
        self.organic_tolerance = organic_tolerance
        self._expon = 1 / (2 * self.organic_tolerance)

    def evaluate(self, rels, hierarchy):
        if hierarchy.organics:
            # cohesion: how well the organic subsystems are integrated
            weighted_orgedges = (
                hierarchy.organic_edges
                & ~numpy.diag(numpy.diag(hierarchy.organic_edges))
            ) * rels.weights[numpy.newaxis,:]
            print(hierarchy.organic_membership)
            print(weighted_orgedges.sum(axis=1) * (1 - rels.selfprobs))
            cohesion = (
                (rels.transition_probs * weighted_orgedges).sum(axis=1)
                / numpy.where(
                    hierarchy.organic_membership,
                    weighted_orgedges.sum(axis=1) * (1 - rels.selfprobs),
                    1
                )
            )
            cohesion[~hierarchy.organic_membership] = 1
            # print(cohesion)
            # reciprocity: how equal the organic subsystem units are
            orgflows = rels.matrix * hierarchy.organic_edges
            org_inflows = orgflows.sum(axis=0)
            org_outflows = orgflows.sum(axis=1)
            reciprocity = 1 - (
                numpy.abs(org_outflows - org_inflows)
                / (org_inflows + org_outflows - 2 * rels.selfrels)
            )
            # print(reciprocity)
            organic_crit = (cohesion * reciprocity) ** self._expon
            # print(organic_crit)
        else:
            organic_crit = 1
        hierarchy_crit = (rels.transition_probs / hierarchy.binding_matrix(rels)).sum(axis=1)
        # print(hierarchy_crit)
        # print(numpy.sqrt(hierarchy_crit * organic_crit))
        return rels.weighted_sum(numpy.sqrt(hierarchy_crit * organic_crit))


class HierarchyBuilder:
    def build(self, rels):
        raise NotImplementedError


class MaxflowHierarchyBuilder(HierarchyBuilder):
    DECYCLE_TRIES = 10
    deterministic = True

    def build(self, rels, **kwargs):
        probs = (rels.matrix - numpy.diag(rels.selfrels)) / rels.outsums_noself[:,numpy.newaxis]
        parents = numpy.apply_along_axis(self._argmax, 1, probs).ravel()
        # choose hierarchy root proportionally to number of incoming hierarchy edges
        # parent_counts = numpy.bincount(parents, minlength=rels.n)
        sqinflows = rels.insums ** 2
        rootpos = self._argmax(sqinflows / sqinflows.sum())
        parents[rootpos] = rootpos
        # find all cycles, declare them organic and add their binding higher
        organicity = numpy.zeros(rels.n, dtype=bool)
        cycles = list(graph.parental_cycles(parents))
        organic_heads = []
        for i_try in range(self.DECYCLE_TRIES):
            print('decycling')
            if len(cycles) == 1:
                break
            for cycle in cycles:
                if len(cycle) == 1:
                    continue # root cycle (no other unit may point to itself)
                # declare organic by finding the largest (deterministically)
                orghead = cycle[rels.outsums[cycle].argmax()]
                # assign the others to it organically
                parents[cycle] = orghead
                organicity[cycle] = True
                # if the items in the cycle were already parts of organic
                # subsystems, merge their tails into this one
                for i in cycle:
                    if i in organic_heads:
                        organic_heads.remove(i)
                        parents[parents == i] = orghead
                organic_heads.append(orghead)
                # assign the head somewhere else
                probs[orghead,cycle] = 0
                probs[orghead] /= probs[orghead].sum()
                parents[orghead] = self._argmax(probs[orghead])
                organicity[orghead] = False
            cycles = list(graph.parental_cycles(parents))
        else:
            # failed to decycle, rinse and repeat
            if self.deterministic:
                print(parents)
                print(organicity)
                print(organic_heads)
                print(cycles)
                raise ValueError('cannot decycle')
            else:
                return self.build(rels, **kwargs)
        return Hierarchy.create(parents, organicity, **kwargs)

    def _argmax(self, weights):
        return weights.argmax()

class MaxflowStochasticHierarchyBuilder(MaxflowHierarchyBuilder):
    deterministic = False

    def _argmax(self, weights):
        return numpy.random.choice(len(weights), 1, p=weights)


class HierarchyOperator:
    def _select_random_element(self, source, exclude=[]):
        if isinstance(source, Hierarchy):
            source = source.flat
        assert len(exclude) < len(source)
        sel = numpy.random.choice(source, 1)[0]
        while sel in exclude:
            sel = numpy.random.choice(source, 1)[0]
        return sel


class HierarchyModifier(HierarchyOperator):
    def modify(self, hierarchy):
        raise NotImplementedError

    def _select_bound_to(self, hierarchy, rels, binder, exclude=[]):
        assert len(exclude) < len(hierarchy.flat)
        probs = binder.outward_transition_probabilities(rels)
        for element in exclude:
            probs[element.id] = 0
        cumprobs = probs.cumsum()
        probsum = cumprobs[-1]
        if probsum == 0:
            return None
        else:
            return hierarchy.elements_by_id[
                numpy.searchsorted(cumprobs, numpy.random.rand() * probsum)
            ]

    def _select_bound_to_limited(self, source, rels, binder, exclude=[]):
        if isinstance(source, Hierarchy):
            source = source.flat
        probs = binder.outward_transition_probabilities(rels)
        weights = numpy.array([
            probs[element.id].sum() if element not in exclude else 0
            for element in source
        ])
        cumweights = weights.cumsum()
        wsum = cumweights[-1]
        if wsum == 0:
            return None
        else:
            return source[
                numpy.searchsorted(cumweights, numpy.random.rand() * wsum)
            ]

    def _select_high_inflow(self, source, rels, exclude=[]):
        if isinstance(source, Hierarchy):
            source = source.flat
        insums = numpy.array([
            el.inflow_sum(rels) if el not in exclude else 0
            for el in source
        ])
        cuminsums = insums.cumsum()
        insumsum = cuminsums[-1]
        if insumsum == 0:
            return None
        else:
            return source[
                numpy.searchsorted(cuminsums, numpy.random.rand() * insumsum)
            ]

    def _select_random_organic(self, hierarchy):
        if hierarchy.organics:
            return numpy.random.choice(hierarchy.organics, 1)[0]
        else:
            return None


class RootMover(HierarchyModifier):
    def modify(self, hierarchy, rels):
        # change root to randomly selected node with high inflows
        new_root = self._select_high_inflow(hierarchy, rels, exclude=[hierarchy.root])
        if new_root:
            print('setting', new_root, 'as new root')
            hierarchy.set_root(new_root)
            return True

    # def modify(self, hierarchy, rels):
        # # change root to randomly selected node with high inflows
        # cumweights = numpy.array([
            # el.inflow_sum(rels) if el is not hierarchy.root else 0
            # for el in hierarchy.flat
        # ]).cumsum()
        # hierarchy.set_root(hierarchy.flat[numpy.searchsorted(
            # cumweights, numpy.random.rand() * cumweights[-1]
        # )])


class ElementMover(HierarchyModifier):
    def modify(self, hierarchy, rels):
        if len(hierarchy.flat) > 1:
            # if there is only one element (root), there is nothing to move
            # do not move root (we have root mover for that)
            new_child = self._select_random_element(hierarchy, exclude=[hierarchy.root])
            # select a parent with a strong binding
            new_parent = self._select_bound_to(hierarchy, rels, new_child, exclude=[new_child.parent])
            if new_parent:
                print('moving', new_child, 'under', new_parent)
                # if new_child is predecessor of new_parent, we will have to assign
                # new_parent to new_child.parent to maintain hierarchy
                if new_parent.has_predecessor(new_child):
                    new_grandparent = new_child.parent
                else:
                    new_grandparent = None
                new_child.parent.discard_child(new_child)
                new_parent.add_child(new_child)
                if new_grandparent:
                    new_parent.parent.discard_child(new_parent)
                    new_grandparent.add_child(new_parent)
                hierarchy.change_made()
                return True


class OrganicSubsystemCreator(HierarchyModifier):
    def modify(self, hierarchy, rels):
        # TODO we are losing elements here!
        if hierarchy.organic_membership.sum() < (hierarchy.n - 1):
            # when everybody is in organic system, cannot do anything
            # create a new organic subsystem from a random non-organic node
            source = self._select_random_element(hierarchy, exclude=hierarchy.organics)
            # and select a strongly bound other non-organic node
            companion = self._select_bound_to(hierarchy, rels, source, exclude=hierarchy.organics)
            if companion:
                print('creating organic subsystem from', source, 'and', companion)
                # put newly created subsystem at the root if source or companion were root
                if not source.parent:
                    new_parent = None
                    companion.parent.discard_child(companion)
                elif not companion.parent:
                    new_parent = None
                    source.parent.discard_child(source)
                else:
                    # the parent is taken from the source primarily
                    if companion in source.predecessors():
                        new_parent = companion.parent
                    else:
                        new_parent = source.parent
                    source.parent.discard_child(source)
                    companion.parent.discard_child(companion)
                # create an organic system out of source and companion
                hierarchy.deregister(source)
                hierarchy.deregister(companion)
                orgsys = OrganicSubsystem.create(source, companion)
                hierarchy.register(orgsys)
                if new_parent:
                    new_parent.add_child(orgsys)
                else:
                    hierarchy.set_root(orgsys)
                return True


class OrganicSubsystemDestructor(HierarchyModifier):
    def modify(self, hierarchy, rels):
        # select an organic subsystem and destroy it
        # TIP select subsystem with low fitness?
        orgsys = self._select_random_organic(hierarchy)
        if orgsys:
            print('destroying', orgsys)
            hierarchy.deregister(orgsys)
            nodes = orgsys.members.copy()
            children = orgsys.children.copy()
            # retain former subsystem parent for all nodes
            parent = orgsys.parent
            for node in nodes:
                orgsys.remove_member(node)
                hierarchy.register(node)
            if not parent:
                # system was at root, choose parent among nodes (stochastically by inflow sum)
                parent = self._select_high_inflow(nodes, rels)
                hierarchy.set_root(parent)
            else:
                parent.discard_child(orgsys)
            # the nodes within will be assigned to parent
            for node in nodes:
                if node is not parent:
                    parent.add_child(node)
            for child in children:
                orgsys.discard_child(child)
                redir = self._select_bound_to_limited(nodes, rels, child)
                redir.add_child(child)
            return True


class OrganicSubsystemAdder(HierarchyModifier):
    def modify(self, hierarchy, rels):
        # select a random organic system
        # TIP select subsystem with low fitness?
        # TODO makes two merged organic systems disappear!
        orgsys = self._select_random_organic(hierarchy)
        if orgsys:
            # and add an element with a high bond to it
            joiner = self._select_bound_to(hierarchy, rels, orgsys)
            if joiner:
                print('joining', joiner, 'to', orgsys)
                joiner_parent = joiner.parent
                prev = orgsys
                for predec in orgsys.predecessors():
                    if predec is joiner:
                        joiner.discard_child(prev)
                        if joiner_parent:
                            joiner_parent.add_child(prev)
                        break
                    prev = predec
                if not joiner_parent:
                    # the joiner is root, set the joined system as root
                    hierarchy.set_root(orgsys)
                hierarchy.deregister(joiner)
                if joiner.is_organic:
                    # merge two organic subsystems
                    orgsys.merge(joiner)
                else:
                    # merge a node to the subsystem
                    orgsys.add_member(joiner)
                hierarchy.refresh(orgsys)
                return True


class OrganicSubsystemSubtractor(HierarchyModifier):
    def modify(self, hierarchy, rels):
        if hierarchy.organics:
            # select a random node within an organic system with at least three
            # members
            elig_orgs = []
            elig_weights = []
            for orgsys in hierarchy.organics:
                if orgsys.size > 2:
                    elig_orgs.append(orgsys)
                    elig_weights.append(orgsys.size)
            if not elig_orgs:
                return
            elig_probs = numpy.array(elig_weights)
            # select the system
            selsys = numpy.random.choice(
                elig_orgs, 1,
                p=(elig_probs / elig_probs.sum())
            )[0]
            # TIP select node with low fitness contribution?
            node = numpy.random.choice(selsys.members, 1)[0]
            print('removing', node, 'from', selsys, 'and setting it as a child')
            # remove it from the system and add it as a child
            orgsys = node.subsystem
            orgsys.remove_member(node)
            orgsys.add_child(node)
            hierarchy.refresh(orgsys)
            # TIP try reassigning some of the orgsys children?
            return True


class HierarchyCrossover(HierarchyOperator):
    def crossover(self, hier1, hier2, rels):
        # select random element from second hierarchy
        sel_el = self._select_crossover_element(hier2, rels)
        sel_predec_ids = [id
            for el in sel_el.predecessors()
                for id in sorted(el.ids(), key=(lambda id: -rels.weights[id]))
        ]
        crossel = sel_el.copy()
        crossids = list(crossel.tree_ids())
        print(crossel, '(', crossids, ')', crossel.parent)
        # copy the first hierarchy - this will be the crossover result
        root = hier1.root.copy()
        # remove all ids contained in the 2nd hierarchy element from result
        orphans = self._prune(root, crossids)
        # TODO reassign some of the nodes whose parents have been pruned?
        # probably not... they just were not in the crossovered part
        if orphans:
            return self._hierarchy_with_orphans(crossel, orphans, hier1.n)
        else:
            return self._joined_hierarchy(root, crossel, sel_predec_ids, hier1.n)

    def _hierarchy_with_orphans(self, crossel, orphans, n):
        hierarchy = Hierarchy(crossel, n, complete=False)
        for orphan in orphans:
            crossel.add_child(orphan)
            hierarchy.register(orphan)
            for el in orphan.descendants():
                hierarchy.register(el)
        hierarchy.completed()
        return hierarchy

    def _joined_hierarchy(self, root, crossel, predec_ids, n):
        hierarchy = Hierarchy(root, n, complete=False)
        # go along the predecessor ids until we get to root or find something
        # in the hierarchy
        print('joining', predec_ids)
        for id in predec_ids:
            parent = hierarchy.elements_by_id[id]
            if parent:
                parent.add_child(crossel)
                replace_root = False
                break
        else:
            replace_root = True
        hierarchy.register(crossel)
        for el in crossel.descendants():
            hierarchy.register(el)
        if replace_root:
            # no parent found, replace root
            crossel.add_child(root)
            hierarchy.set_root(crossel)
        hierarchy.completed()
        return hierarchy

    def _prune(self, root, ids):
        accum = HierarchyElement()
        accum.add_child(root)
        root.prune(ids)
        if accum.children == [root]:
            accum.discard_child(root)
            return []
        else:
            orphans = accum.children[:]
            accum.dissolve()
            return orphans

    def _join(self, hierarchy, crossel, predec_ids):
        # go along the predecessor ids until we get to root or find something
        # in the hierarchy
        hierarchy.register(crossel)
        for el in crossel.descendants():
            hierarchy.register(el)
        for id in predec_ids:
            parent = hierarchy.elements_by_id[id]
            if parent:
                parent.add_child(crossel)
                break
        else:
            # no parent found, replace root
            crossel.add_child(hierarchy.root)

    def _select_crossover_element(self, hier, rels):
        cweight = 0
        while not abs(cweight - .5) * 4 < numpy.random.rand():
            crossel = self._select_random_element(hier)
            cweight = crossel.tree_weight(rels)
        return crossel


HIERARCHY_MODIFIERS = [
    RootMover,
    ElementMover,
    OrganicSubsystemCreator,
    OrganicSubsystemDestructor,
    OrganicSubsystemAdder,
    OrganicSubsystemSubtractor,
]