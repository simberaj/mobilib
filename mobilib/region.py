
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
        self.transition_probs = self.matrix / self.weights[:,numpy.newaxis]
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
    def __init__(self, root, n):
        self.root = root
        self.n = n
        self.flat = list(root.descendants())
        self.organics = [item for item in self.flat if item.is_organic]
        self.has_organic = bool(self.organics)
        self.organic_membership = self._compute_organic_membership()
        self._organic_edges = None
        self._binding_matrix = None
        
    def structure_string(self):
        return '\n'.join(self.root.structure_lines(indent=0))
    
    @property
    def binding_matrix(self):
        if self._binding_matrix is None:
            self._binding_matrix = numpy.full((self.n,self.n), numpy.inf)
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
        
    def _compute_organic_membership(self):
        mem = numpy.zeros(self.n, dtype=bool)
        for orgsys in self.organics:
            mem[orgsys.id] = True
        return mem
       
    @classmethod
    def create(cls, parents, organics=None, ids=None, root=None, rels=None):
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
        for item in nodes:
            item.weigh_binding(rels)
        return cls(nodes[root], len(nodes))

        
class HierarchyElement:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.binding_weight = None
    
    def bind_paths(self, matrix):
        if self.parent is not None:
            parent_binding = matrix[self.parent.id,:]
            if len(parent_binding.shape) == 2:
                parent_binding = parent_binding[0]
            self._record_bind_paths(matrix, parent_binding / self.binding_weight)
        self._record_self_bind(matrix)
        for child in self.children:
            child.bind_paths(matrix)

    def structure_lines(self, indent):
        yield ' ' * indent + self.name + (
            (' -> ' + str(self.binding_weight))
                if self.binding_weight is not None else ''
        )
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
    
    def descendants(self):
        yield self
        for child in self.children:
            yield from child.descendants()
        
        
class HierarchyNode(HierarchyElement):
    is_organic = False

    def __init__(self, id, name, parent=None, subsystem=None):
        super().__init__(parent)
        self.id = id
        self.name = str(name)
        self.subsystem = subsystem
    
    def add_child(self, child):
        if self.subsystem is None:
            super().add_child(child)
        else:
            self.subsystem.add_child(child)
    
    def set_subsystem(self, subsystem):
        self.subsystem = subsystem
        children = self.children
        self.children = []
        self.parent = None
        return children
    
    def _extract_binding_weight(self, rels):
        return rels.transition_probs[self.id,self.parent.id].sum() / (1 - rels.selfprobs[self.id])
        
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
    
    @property
    def id(self):
        return [member.id for member in self.members]
    
    def add_member(self, member):
        self.members.append(member)
        member_children = member.set_subsystem(self)
        for child in member_children:
            self.add_child(child)
    
    def _extract_binding_weight(self, rels):
        weights = rels.weights[self.id]
        parent_probs = rels.transition_probs[self.id][:,self.parent.id]
        if len(parent_probs.shape) == 2:
            parent_probs = parent_probs.sum(axis=1)
        for i, id in enumerate(self.id):
            parent_probs[i] /= (1 - rels.transition_probs[id,self.id].sum())
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
        if hierarchy.has_organic:
            # cohesion: how well the organic subsystems are integrated
            weighted_orgedges = (
                hierarchy.organic_edges
                & ~numpy.diag(numpy.diag(hierarchy.organic_edges))
            ) * rels.weights[numpy.newaxis,:]
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
        hierarchy_crit = (rels.transition_probs / hierarchy.binding_matrix).sum(axis=1)
        # print(hierarchy_crit)
        # print(numpy.sqrt(hierarchy_crit * organic_crit))
        return rels.weighted_sum(numpy.sqrt(hierarchy_crit * organic_crit))
                
        
class HierarchyBuilder:
    def build(self, rels):
        raise NotImplementedError
        
        
class MaxflowStochasticHierarchyBuilder(HierarchyBuilder):
    DECYCLE_TRIES = 10

    def build(self, rels, **kwargs):
        cumrels = (rels.matrix - numpy.diag(rels.selfrels)).cumsum(axis=1)
        parents = numpy.diag(numpy.apply_along_axis(numpy.searchsorted, 1,
            cumrels,
            numpy.random.rand(rels.n) * rels.outsums_noself
        )).copy()
        # choose hierarchy root proportionally to number of incoming hierarchy edges
        # parent_counts = numpy.bincount(parents, minlength=rels.n)
        rootpos = self._stochastic_argmax((rels.insums ** 2).cumsum())
        parents[rootpos] = rootpos
        # find all cycles, declare them organic and add their binding higher
        organicity = numpy.zeros(rels.n, dtype=bool)
        cycles = list(graph.parental_cycles(parents))
        organic_heads = []
        for i_try in range(self.DECYCLE_TRIES):
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
                parents[orghead] = self._stochastic_argmax(cumrels[orghead])
                organicity[orghead] = False
            cycles = list(graph.parental_cycles(parents))
        else:
            # failed to decycle, rinse and repeat
            return self.build(rels, **kwargs)
        return Hierarchy.create(parents, organicity, rels=rels, **kwargs)
        
    
    def _stochastic_argmax(self, cumwts):
        return numpy.searchsorted(
            cumwts,
            numpy.random.rand() * cumwts[-1]
        )
