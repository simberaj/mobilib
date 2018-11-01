import numpy

DEFAULT_HOME_CODE = 'k'
DEFAULT_WORK_CODE = 't'
DEFAULT_MULTIFX_CODE = 'm'

def build_codes(main_code, sec_code, sec_fraction=.5):
    return {main_code : 1, sec_code : sec_fraction}

DEFAULT_HOME_CODES = build_codes(
    DEFAULT_HOME_CODE,
    DEFAULT_MULTIFX_CODE,
)
DEFAULT_WORK_CODES = build_codes(
    DEFAULT_WORK_CODE,
    DEFAULT_MULTIFX_CODE,
)

class RelationGenerator:
    def relate(self, importances, types, selfinter=True):
        sources = self.sources(importances, types)
        source_sum = sources.sum()
        n = len(importances)
        if source_sum == 0:
            return numpy.zeros((n,n))
        targets = self.targets(importances, types)
        eligibility = self.eligibility(n, selfinter=selfinter)
        targeting_weight = eligibility * targets[numpy.newaxis,:]
        weight_sum = targeting_weight.sum(axis=1)
        has_targets = weight_sum != 0
        targeted_sources = sources * has_targets
        tsource_sum = targeted_sources.sum()
        if tsource_sum == 0:
            return numpy.zeros((n,n))
        transfer = targeted_sources / (tsource_sum * numpy.where(has_targets, weight_sum, 1))
        return transfer[:,numpy.newaxis] * targeting_weight
        
    @staticmethod
    def sources(importances, types):
        return importances
        
    @staticmethod
    def targets(importances, types):
        return importances
        
    @staticmethod
    def eligibility(n, selfinter=True):
        elig = numpy.ones((n,n), dtype=bool)
        if not selfinter:
            numpy.fill_diagonal(elig, False)
        return elig
        
    @staticmethod
    def type_importances(importances, types, codes):
        return importances * numpy.stack([
            (types == code) * weight for code, weight in codes.items()
        ]).sum(axis=0)
        
        
class GeneralRelationGenerator(RelationGenerator):
    name = 'gen'

        
class HomeBaseRelationGenerator(RelationGenerator):
    name = 'home'
    
    def __init__(self, home_codes=DEFAULT_HOME_CODES):
        self.home_codes = home_codes
        
    def sources(self, importances, types):
        return self.type_importances(importances, types, self.home_codes)


class HomeWorkRelationGenerator(HomeBaseRelationGenerator):
    name = 'homework'
    
    def __init__(self, home_codes=DEFAULT_HOME_CODES, work_codes=DEFAULT_WORK_CODES):
        self.home_codes = home_codes
        self.work_codes = work_codes
        
    def targets(self, importances, types):
        return self.type_importances(importances, types, self.work_codes)
    
        
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
        GeneralRelationGenerator(),
        HomeBaseRelationGenerator(),
        HomeWorkRelationGenerator(),
    ]
    for selfinter in (False, True):
        print('si', selfinter)
        for gen in gens:
            print(gen)
            print((gen.relate(imps, codes, selfinter=selfinter) * 1000).astype(int))
        print()
            