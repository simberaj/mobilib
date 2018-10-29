import numpy


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
    pass

        
class HomeBaseRelationGenerator(RelationGenerator):
    def __init__(self, home_codes={'k' : 1, 'm' : .5}):
        self.home_codes = home_codes
        
    def sources(self, importances, types):
        return self.type_importances(importances, types, self.home_codes)


class HomeWorkRelationGenerator(HomeBaseRelationGenerator):
    def __init__(self, home_codes={'k' : 1, 'm' : .5}, work_codes={'t' : 1, 'm' : .5}):
        self.home_codes = home_codes
        self.work_codes = work_codes
        
    def targets(self, importances, types):
        return self.type_importances(importances, types, self.work_codes)
    
        
if __name__ == '__main__':
    imps = numpy.array([10,2,2,1,1,1])
    codes = numpy.array(['k','','t','','',''])
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
            