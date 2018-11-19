# coding: utf8

import numpy
import pandas as pd
import scipy.sparse.csgraph
# import matplotlib.pyplot as plt
# import sklearn.manifold

# outfile = open('markov.txt', 'w', encoding='utf8')

def mfpt(trans, stat=None):
    if stat is None:
        stat = stationary_distribution(p)
    else:
        stat = stat.copy()
    stat[stat==0] = 0.25 * stat[stat!=0].min()
    stat /= stat.sum()
    fund = fundamental_matrix(trans, stat)
    return (
        numpy.eye(stat.size)
        - fund
        + numpy.ones_like(fund).dot(numpy.diag(numpy.diag(fund)))
    ).dot(numpy.diag(1 / stat))
    
    
def stationary_distribution(p):
    eigval, eigvec = numpy.linalg.eig(p.T)
    vec_stat = eigvec[:,numpy.argmin(abs(eigval - 1.0))].real
    return vec_stat / vec_stat.sum()

    
def transition_matrix(inter):
    return inter / inter.sum(axis=1)[:,numpy.newaxis]
    
    
def fundamental_matrix(trans, stat):
    return numpy.linalg.inv(numpy.eye(stat.size) - trans + stat[numpy.newaxis,:])
    
    
def df_to_relations(df, from_id_col, to_id_col, strength_col):
    all_ids = numpy.array(list(sorted(set(
        list(df[from_id_col].unique())
        + list(df[to_id_col].unique())
    ))))
    n = len(all_ids)
    relmatrix = numpy.zeros((n,n), dtype=df[strength_col].dtype)
    from_ids = numpy.searchsorted(all_ids, df[from_id_col])
    to_ids = numpy.searchsorted(all_ids, df[to_id_col])
    relmatrix[from_ids,to_ids] = df[strength_col].values
    return all_ids, relmatrix
    
def info(debug, *args):
    if debug:
        # print(*args, file=outfile)
        print(*args)
    
    
def regionalize_markov(stat, trans, mfpts, min_crit=0.01, crit=None, dissolve=False, debug=False, ids=None):
    if dissolve:
        raise NotImplementedError
    indices = numpy.arange(stat.size)
    # initial assignments are each to its own region
    assignments = indices.copy()
    hierarchy = indices.copy()
    if crit is None:
        crit = stat
    if ids is None:
        ids = indices
    regcrit = crit.copy()
    min_reg = regcrit.argmin()
    # steps = [])
    trans_orig = trans.copy()
    mfpt_diag = numpy.diag(mfpts)
    while regcrit[min_reg] < min_crit:
        info(debug)
        info(debug, 'minreg chosen:', ids[min_reg], 'crit', regcrit[min_reg])
        in_reg = assignments == min_reg
        info(debug, 'within chosen:', [ids[i] for i in numpy.nonzero(in_reg)[0]])
        if dissolve and in_reg.sum() > 1:
            info(debug, 'dissolving region')
            # if region of more than one unit, replace regional assignment
            # back to individual units
            assignments[in_reg] = indices[in_reg]
            # restore individual unit criteria
            regcrit[in_reg] = crit[in_reg]
            # clear bindings of the members to the regional center and continue
            mfpts[in_reg][:,min_reg] = numpy.inf
            # --- temporary: restore the original transition matrix
            trans[:,in_reg] = trans_orig[:,in_reg]
            trans[in_reg,:] = trans_orig[in_reg,:]
            # --- end of temporary
            min_reg = regcrit.argmin()
            continue
        # find target to aggregate min_reg by smallest MFPT
        tgt_reg = mfpts[min_reg].argmin()
        next = mfpts[min_reg][indices!=tgt_reg].argmin()
        info(debug, 'target:', ids[tgt_reg], mfpts[min_reg,tgt_reg], regcrit[tgt_reg], '/ next:', ids[next], mfpts[min_reg,next], regcrit[next])
        if tgt_reg == min_reg: # strongest binding to itself
            info(debug, 'selfinter chosen, retrying')
            mfpts[min_reg,min_reg] = numpy.inf
            tgt_reg = mfpts[min_reg].argmin()
            info(debug, 'target:', ids[tgt_reg])
            mfpts[min_reg,min_reg] = mfpt_diag[min_reg]
        # region stationary probability will be sum of both regions
        regcrit[tgt_reg] += regcrit[min_reg]
        info(debug, 'new crit', regcrit[tgt_reg])
        # clear the criterion of the eliminated region
        regcrit[min_reg] = numpy.inf
        # update the MFPTs from and to the new region
        # --- temporary: update transition matrix and recompute all mfpts
        # cleared = numpy.isinf(mfpts).all(axis=0)
        # min_stat = stat[min_reg]
        # tgt_stat = stat[tgt_reg]
        # new_stat = min_stat + tgt_stat
        # selfinter = (
            # min_stat * (trans[min_reg,min_reg] + trans[min_reg,tgt_reg])
            # + tgt_stat * (trans[tgt_reg,min_reg] + trans[tgt_reg,tgt_reg])
        # ) / new_stat
        # trans[:,tgt_reg] += trans[:,min_reg]
        # trans[tgt_reg] = (min_stat * trans[min_reg] + tgt_stat * trans[tgt_reg]) / new_stat
        # trans[tgt_reg,tgt_reg] = selfinter
        # trans[min_reg] = 0
        # trans[min_reg,tgt_reg] = 1
        # info(debug, 'new trans')
        # info(debug, trans)
        # info(debug, trans.sum(axis=1))
        # mfpts = mfpt(trans)
        # mfpts[:,cleared] = numpy.inf
        # info(debug, 'new mfpts')
        # info(debug, mfpts)
        # --- end of temporary
        if dissolve:
            assignments[min_reg] = tgt_reg
        else:
            # perform assignment
            assignments[in_reg] = tgt_reg
            # record hierarchical relationship
            hierarchy[min_reg] = tgt_reg
        # select next aggregation target
        min_reg = regcrit.argmin()
    info(debug, 'terminating with', ids[min_reg], regcrit[min_reg])
    if dissolve:
        hierarchy[:] = assignments[:]
    return assignments, hierarchy
    
    
def entropy_criterion(rels, assignments, mass=None, entropy_weight=0.25):
    if mass is None:
        mass = rels.sum(axis=1)
    in_same_region = (assignments[:,numpy.newaxis] == assignments[numpy.newaxis,:])
    same_region_relfrac = (rels * in_same_region).sum() / rels.sum()
    entropy = division_entropy(assignments, mass=mass)
    print(same_region_relfrac, entropy)
    # return numpy.sqrt(same_region_relfrac * entropy)
    return (entropy ** entropy_weight * same_region_relfrac) ** (1 + 1 / entropy_weight)
    
def division_entropy(assignments, mass):
    regmass = numpy.bincount(assignments, mass)
    regmass_fracs = regmass[regmass>0] / regmass.sum()
    # n_regions = regmass_fracs.size
    entropy = -(regmass_fracs * numpy.log(regmass_fracs)).sum()
    # if n_regions < 2:
        # return entropy
    # else:
        # return entropy / numpy.log(n_regions) # normalize to log_n_regions
    return entropy / numpy.log(assignments.size)

def regionalize_directly(rels, crit=None, min_crit=None, ids=None):
    ids = [id.encode('utf8') for id in ids]
    if crit is None:
        crit = rels.sum(axis=1)
    if min_crit is None:
        min_crit = numpy.sqrt(crit.sum())
    regcrit = crit.copy()
    min_reg = regcrit.argmin()
    assignments = numpy.arange(regcrit.size)
    selfrels = numpy.diag(rels)
    if ids is None:
        ids = numpy.arange(regcrit.size)
    n_iter = 0
    rels_sel = rels.copy()
    rels_sel -= numpy.diag(selfrels)
    while regcrit[min_reg] < min_crit:
        print()
        print('min', ids[min_reg], '(', regcrit[min_reg], ')')
        in_min_reg = (assignments == min_reg)
        print('contained', [ids[i] for i in numpy.flatnonzero(in_min_reg)])
        tgt_reg = rels_sel[min_reg].argmax()
        # if tgt_reg == min_reg: # strongest binding to itself
            # print('selfinter chosen, retrying')
            # rels[min_reg,min_reg] = -numpy.inf
            # tgt_reg = rels[min_reg].argmax()
            # rels[min_reg,min_reg] = selfrels[min_reg]
        print('target', ids[tgt_reg], '(', regcrit[tgt_reg], ') <-', rels[min_reg,tgt_reg])
        # assign
        assignments[in_min_reg] = tgt_reg
        regcrit[tgt_reg] += regcrit[min_reg]
        regcrit[min_reg] = numpy.inf
        # remove relations to min_reg
        rels_sel[:,min_reg] = 0
        print('new crit', regcrit[tgt_reg])
        min_reg = regcrit.argmin()
        print('global fitness', criterion(rels, assignments, mass=crit))
        n_iter += 1
        # if n_iter > 50:
            # break
    return assignments
    
def maxflow_hierarchy_deterministic(rels, ids=None):
    rels_sel = rels.copy()
    numpy.fill_diagonal(rels_sel, 0)
    return rels_sel.argmax(axis=1)
    
    # while regcrit[min_reg] < min_crit:
        # print()
        # print('min', ids[min_reg], '(', regcrit[min_reg], ')')
        # in_min_reg = (assignments == min_reg)
        # print('contained', [ids[i] for i in numpy.flatnonzero(in_min_reg)])
        # tgt_reg = rels_sel[min_reg].argmax()
        # # if tgt_reg == min_reg: # strongest binding to itself
            # # print('selfinter chosen, retrying')
            # # rels[min_reg,min_reg] = -numpy.inf
            # # tgt_reg = rels[min_reg].argmax()
            # # rels[min_reg,min_reg] = selfrels[min_reg]
        # print('target', ids[tgt_reg], '(', regcrit[tgt_reg], ') <-', rels[min_reg,tgt_reg])
        # # assign
        # assignments[in_min_reg] = tgt_reg
        # regcrit[tgt_reg] += regcrit[min_reg]
        # regcrit[min_reg] = numpy.inf
        # # remove relations to min_reg
        # rels_sel[:,min_reg] = 0
        # print('new crit', regcrit[tgt_reg])
        # min_reg = regcrit.argmin()
        # print('global fitness', criterion(rels, assignments, mass=crit))
        # n_iter += 1
        # # if n_iter > 50:
            # # break
    # return assignments
    
    
# def regionalize_selfcont(rels, min_crit=None, ids=None):
    # insums, outsums = [rels.sum(axis=i) for i in (0, 1)]
    # selfinter = numpy.diag(rels)
    # # sc = insums / (insums + outsums - selfinter)
    # sc = selfinter / outsums
    # most_closed = numpy.argsort(sc)[-10:]
    # for i in most_closed:
        # print(ids[i], sc[i])
        
def floyd_warshall_multiplicative(adj, debug=False):
    if debug:
        print('FLOYD-WARSHALL')
        print(adj)
        print()
    dim = adj.shape[0]
    indices = numpy.arange(dim)
    lengths = adj.copy()
    for i in indices:
        for j in indices:
            for k in indices:
                auglen = lengths[j,i] * lengths[i,k]
                if auglen < lengths[j,k]:
                    if debug: print('{}->{}: {}->{}'.format(j, k, lengths[j,k], auglen), end=', ')
                    lengths[j,k] = auglen
        if debug: print('via', i)
    return lengths

def floyd_warshall_additive(adj, debug=False):
    if debug:
        print('FLOYD-WARSHALL')
        print(adj)
        print()
    dim = adj.shape[0]
    indices = numpy.arange(dim)
    lengths = adj.copy()
    for i in indices:
        for j in indices:
            for k in indices:
                auglen = lengths[j,i] + lengths[i,k]
                if auglen < lengths[j,k]:
                    if debug: print('{}->{}: {}->{}'.format(j, k, lengths[j,k], auglen), end=', ')
                    lengths[j,k] = auglen
        if debug: print('via', i)
    return lengths

def floyd_warshall_additive2(adj):
    lengths = adj.copy()
    for k in range(adj.shape[0]):
        lengths = numpy.stack((
            lengths,
            lengths[:,k][:,numpy.newaxis] + lengths[k,:][numpy.newaxis,:]
        )).min(axis=0)
    return lengths


def maxflow_hierarchy_stochastic(rels):
    # probability of choosing a parent is proportional to transition probability
    # avoid self-relations!
    n = rels.shape[0]
    outsums = rels.sum(axis=1)
    selfrels = numpy.diag(rels)
    outsums_noself = outsums - selfrels
    cumrels = (rels - numpy.diag(selfrels)).cumsum(axis=1)
    hierarchy = numpy.diag(numpy.apply_along_axis(numpy.searchsorted, 1,
        cumrels,
        numpy.random.rand(n) * outsums_noself
    )).copy()
    # choose hierarchy root proportionally to squared inflow size
    rootpos = maxinflow_root_stochastic(rels)
    hierarchy[rootpos] = rootpos
    # find all cycles, declare them organic and add their binding higher
    organicity = numpy.zeros(n, dtype=bool)
    cycles = list(find_cycles(hierarchy))
    n_tries = 1
    while len(cycles) > 1:
        for cycle in cycles:
            if len(cycle) == 1:
                continue # root cycle (no other unit may point to itself)
            # declare organic by finding the largest (deterministically)
            orghead = cycle[outsums[cycle].argmax()]
            # assign the others to it organically
            hierarchy[cycle] = orghead
            organicity[cycle] = True
            # assign the head somewhere else
            hierarchy[orghead] = numpy.searchsorted(
                cumrels[orghead],
                numpy.random.rand() * outsums_noself[orghead]
            )
            organicity[orghead] = False
        cycles = list(find_cycles(hierarchy))
        n_tries += 1
        if n_tries > 50:
            return maxflow_hierarchy_stochastic(rels)
    return hierarchy, organicity

    
def maxinflow_root_stochastic(rels):
    sqinflows = rels.sum(axis=0) ** 2
    return numpy.searchsorted(
        sqinflows.cumsum(),
        numpy.random.rand() * sqinflows.sum()
    )

def find_cycles(hierarchy):
    print('HIER', hierarchy)
    n = hierarchy.size
    visited = numpy.zeros(n, dtype=bool)
    n_vis = 0
    current = None
    while n_vis < n:
        if current is None:
            current = visited.argmin()
            path = []
            # print('selecting next', current)
        else:
            current = hierarchy[current]
            n_vis += 1
            # print('moving on to', current)
        if current in path: # we found a cycle
            # print('cycle found in', path)
            yield path[path.index(current):]
            current = None
        elif visited[current]: # we ran into an explored portion
            # print('previous component found')
            current = None
        else:
            path.append(current)
            visited[current] = True
            # print('incrementing visited')
        # print('next')

        
def hierarchy_criterion(rels, hierarchy, organicity, probs=None, k=1):
    n = len(hierarchy)
    weights = rels.sum(axis=1)
    all_organicity = organicity.copy()
    all_organicity[hierarchy[organicity]] = True
    within_organic = (all_organicity[:,numpy.newaxis] & all_organicity[numpy.newaxis,:])
    if probs is None:
        probs = rels / weights[:,numpy.newaxis]
        # print((probs * 1000).astype(int))
    # cohesion for organic subsystems
    selfprobs = numpy.diag(probs)
    cohesion = ((probs * within_organic).sum(axis=1) - selfprobs) / (1 - selfprobs)
    cohesion[~all_organicity] = 1
    # print('coh', cohesion)
    # reciprocity for organic subsystems
    orgflows = rels * within_organic
    # print(orgflows)
    org_inflows = orgflows.sum(axis=0)
    org_outflows = orgflows.sum(axis=1)
    reciprocity = 1 - numpy.abs(org_outflows - org_inflows) / (org_inflows + org_outflows - 2 * numpy.diag(rels))
    # print('recip', reciprocity)
    orgcrit = (cohesion * reciprocity) ** (1 / (2 * k))
    # build hierarchy graph
    to_org_probs = within_organic | numpy.diag(numpy.ones(n, dtype=bool))
    to_org_probs = (to_org_probs * weights[numpy.newaxis,:]).astype(float)
    to_org_probs /= to_org_probs.sum(axis=1)
    # print((to_org_probs * 100).astype(int))
    indices = numpy.arange(n)
    organic_parents = hierarchy[numpy.where(organicity, hierarchy, indices)]
    hiergraph = numpy.full((n,n), numpy.inf)
    hiergraph[indices,organic_parents] = 0
    hiergraph[organic_parents,indices] = -numpy.log(to_org_probs.dot(probs)[indices,organic_parents])
    hiergraph = numpy.where(within_organic, 0, hiergraph)
    numpy.fill_diagonal(hiergraph, 0)
    # hiergraph2 = numpy.log(hiergraph)
    print(numpy.where(numpy.isinf(hiergraph), -1, hiergraph * 100).astype(int))
    # print((numpy.where(numpy.isinf(hiergraph), 0, numpy.exp(hiergraph)) * 100).astype(int))
    # transmults = scipy.sparse.csgraph.floyd_warshall(numpy.ma.masked_values(hiergraph, numpy.inf))
    # print(numpy.where(numpy.isinf(transmults), -1, transmults * 100).astype(int))
    transprobs = numpy.exp(-floyd_warshall_additive2(hiergraph))
    # transprobs = numpy.exp(-scipy.sparse.csgraph.floyd_warshall(numpy.ma.masked_values(hiergraph, numpy.inf), directed=True))
    # hiercrit = (probs * numpy.exp(-transmults)).sum(axis=1)
    hiercrit = (probs * transprobs).sum(axis=1)
    print(numpy.where(numpy.isnan(transprobs), 0, transprobs * 100).astype(int))
    print('hier', hiercrit)
    crit = numpy.sqrt(hiercrit * orgcrit)
    return (crit * weights).sum() / weights.sum()
    
# def hierarchic_transition_probabilities(probs, graph):
    # print(graph)
    # print(probs)
    # trans = numpy.zeros_like(probs)
    # numpy.fill_diagonal(trans, 1)
    
    
    
    
if __name__ == '__main__':
    # df = pd.read_csv('..\\rels\\ee_data\\rels.csv_homework_muni.csv', sep=';')
    # ids, rels = df_to_relations(df, *df.columns[:3])
    # x = rels
    rels = numpy.array([
        [30, 3, 4, 2, 1, 0, 1, 1],
        [ 6,15,10, 3, 2, 1, 0, 1],
        [ 7, 9,12, 3, 1, 2, 0, 0],
        [10, 3, 2,10, 0, 1, 2, 1],
        [ 5, 8, 3, 1, 9, 4, 0, 0],
        [ 3, 3, 7, 1, 2, 8, 1, 0],
        [ 4, 0, 1, 6, 0, 1,10, 2],
        [ 3, 1, 1, 5, 0, 1, 1, 6],
    ])
    # probs = 1 / (rels / rels.sum(axis=1)[:,numpy.newaxis])
    # print(probs)
    # print(scipy.sparse.csgraph.floyd_warshall(probs, directed=True))
    # raise RuntimeError
    # print(rels)
    # hier = numpy.array((0,0,1,0,1,1,3,3))
    hier = numpy.array((0,0,1,0,1,2,3,3))
    # organ = numpy.array((0,0,1,0,0,0,0,0)).astype(bool)
    # print(hierarchy_criterion(rels, hier, organ, k=1))
    hier_wts = 1 / ((rels / rels.sum(axis=1)[:,numpy.newaxis])[numpy.arange(len(hier)),hier])
    hier_wts[0] = 1
    print(hier_wts)
    
    # maxcrit = 0
    # maxhier, maxorgan = None, None
    # for x in range(100):
        # hier, organ = maxflow_hierarchy_stochastic(rels)
    # # organ = numpy.zeros_like(hier).astype(bool)
    # # print(hier)
    # # print(organ)
        # crit = hierarchy_criterion(rels, hier, organ, k=1)
        # print(crit)
        # if crit > maxcrit:
            # maxhier, maxorgan = hier, organ
            # maxcrit = crit
            
    # print(maxhier)
    # print(maxorgan)
    # print(maxcrit)
    
    # print(ids)
    # print(rels)
    # print(rels.shape)
    # print(x.sum(axis=0).min())
    # print(x.sum(axis=1).min())
    # x = numpy.array([[50,  2,  1,  6],
                     # [ 7, 20,  2,  1],
                     # [ 4,  2, 20,  7],
                     # [ 8,  0,  2, 40]])
    # xint = x - numpy.diag(numpy.diag(x))
    # e = xint.sum(axis=1).reshape(-1,1) * xint.sum(axis=0)
    # e = x.sum(axis=1).reshape(-1,1) * x.sum(axis=0)
    # e = numpy.where(numpy.isfinite(e), e, 1)
    # x = x / e
    # ecoef = (x * e).sum() / (e ** 2).sum()
    # enorm = e * ecoef
    # import itertools
    # for regdiv in itertools.product(range(4), repeat=4):
        # print(regdiv, criterion(x, numpy.array(regdiv)))
    # print(x)
    # print(rat)
    # print(transition_matrix(rat))
    # print(enorm)
    # print(x - enorm)
    
    # x -= numpy.diag(numpy.diag(x))
    # ids = [1,2]
    # x = numpy.array(((9,1),(2,8)))
    # x = numpy.array(((0,1),(1,0)))
    # x = numpy.array([[6,2,2],
                     # [4,4,2],
                     # [7,1,2]])
    # ids = ['a', 'b', 'c']
    # ids = ['a', 'b', 'c', 'd']
    # ass = regionalize_directly(x, ids=ids, min_crit=2000)
    # assig = regionalize_directly(x, ids=ids, min_crit=50000)
    # regionalize_selfcont(x, ids=ids)
    # assig = maxflow_hierarchy_deterministic(x)
    # print(assig.shape)
    # print(len(df.index))
    # pd.DataFrame({
        # 'assignment' : assig,
        # 'assignment_name' : ids[assig],
        # 'id' : ids,
    # }).to_csv('..\\rels\\ee_data\\rels.csv_homework_muni_hier.csv', sep=';', index=False)
    # print(criterion(x, numpy.array((0,0,3,3))))
    # p = transition_matrix(x)
    # stat = stationary_distribution(p)
    # markov = mfpt(p, stat)
    # print(markov)
    # ass, hier = regionalize(stat, p, markov, min_crit=0.25, debug=True, ids=ids)
    # print(markov.min())
    # print(markov.max())
    # mini = markov.argmin()
    # minr, minc = mini // len(ids), mini % len(ids)
    # print(ids[minr], ids[minc], markov[minr,minc])

    # tsne = sklearn.manifold.TSNE(metric='precomputed')
    # coords = tsne.fit_transform(markov) # - numpy.diag(numpy.diag(markov)))
    # print(len(ids))
    # print(coords.shape)
    # plt.scatter(coords[:,0], coords[:,1])
    # ax = plt.gca()
    # for i, row in enumerate(coords):
        # ax.annotate(ids[i], xy=row, xytext=row)
    # plt.show()
    
    
# outfile.close()