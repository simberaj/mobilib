import numpy

DEFAULT_HOME_CODE = 'k'
DEFAULT_WORK_CODE = 't'
DEFAULT_MULTIFX_CODE = 'm'


class UntypedAimer:
    def aim(self, types):
        return numpy.ones(types.shape)
    
    def __repr__(self):
        return '<UntypedAimer>'

        
class TypedAimer:
    def __init__(self, type_fractions):
        self.type_fractions = type_fractions
        
    def aim(self, types):
        return numpy.stack([
            (types == code) * weight
            for code, weight in self.type_fractions.items()
        ]).sum(axis=0)
        
    def __repr__(self):
        return '<Aimer{}>'.format(self.type_fractions)

        
GENERAL_AIMER = UntypedAimer()
        
EE_HOME_AIMER = TypedAimer({
    'k' : 1,
    'm' : .5,
})

EE_WORK_AIMER = TypedAimer({
    't' : 1,
    'm' : .5,
})

def ipf(values, rowsums, colsums, tol=1e-9):
    while True:
        prevalues = values
        prerowsums = values.sum(axis=1)
        values = values * (
            rowsums / numpy.where(prerowsums == 0, 1, prerowsums)
        ).reshape(-1, 1)
        precolsums = values.sum(axis=0)
        values = values * (
            colsums / numpy.where(precolsums == 0, 1, precolsums)
        ).reshape(1, -1)
        diff = abs(prevalues - values).sum()
        if diff <= tol:
            prerowsums = values.sum(axis=1)
            values = values * (
                rowsums / numpy.where(prerowsums == 0, 1, prerowsums)
            ).reshape(-1, 1)
            print(values)
        

class RelationGenerator:
    def __init__(self, name=None, source_aimer=GENERAL_AIMER, target_aimer=GENERAL_AIMER, selfinter=True):
        self.name = name
        self.source_aimer = source_aimer
        self.target_aimer = target_aimer
        self.selfinter = selfinter
        
    def relate(self, importances, types):
        n = len(importances)
        rels = self.eligibility(n).astype(float)
        sources = importances * self.source_aimer.aim(types) * rels.max(axis=1)
        source_sum = sources.sum()
        if source_sum == 0:
            return numpy.zeros((n,n))
        targets = importances * self.target_aimer.aim(types) * rels.max(axis=0)
        target_sum = targets.sum()
        if target_sum == 0:
            return numpy.zeros((n,n))
        result = ipf(rels, sources / source_sum, targets / target_sum)
        return result
        
    def eligibility(self, n):
        elig = numpy.ones((n,n), dtype=bool)
        if not self.selfinter:
            numpy.fill_diagonal(elig, False)
        return elig
        
    def __repr__(self):
        return '<RelGen({0.name},{0.source_aimer}-{0.target_aimer},selfinter={0.selfinter})>'.format(self)
   
        
if __name__ == '__main__':
    imps = numpy.array([4,2,2,1,1,1])
    codes = numpy.array([
        DEFAULT_HOME_CODE,
        DEFAULT_WORK_CODE,
        DEFAULT_MULTIFX_CODE,
        DEFAULT_WORK_CODE,
        '',
        '',
    ])
    gens = [
        RelationGenerator(),
        RelationGenerator(selfinter=False),
        RelationGenerator('home', EE_HOME_AIMER),
        RelationGenerator('home', EE_HOME_AIMER, selfinter=False),
        RelationGenerator('hw', EE_HOME_AIMER, EE_WORK_AIMER),
        RelationGenerator('hw', EE_HOME_AIMER, EE_WORK_AIMER, selfinter=False),
    ]
    for gen in gens:
        print(gen)
        print((gen.relate(imps, codes) * 1000).astype(int))
    print()
            