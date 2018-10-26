import numpy


class RelationGenerator:
    def relate(self, importances, types, selfinter=True):
        raise NotImplementedError
        
    @staticmethod
    def type_indices(types, codes):
        return numpy.stack([
            types == code for code in codes
        ]).any(axis=0)
        
        
class GeneralRelationGenerator(RelationGenerator):
    def relate(self, importances, types, selfinter=True):
        rels = importances[:,numpy.newaxis] * importances[numpy.newaxis,:]
        if not selfinter:
            numpy.fill_diagonal(rels, 0)
        return rels / rels.sum()

        
class HomeBaseRelationGenerator(RelationGenerator):
    def __init__(self, home_codes=['k', 'm']):
        self.home_codes = home_codes
    
    def relate(self, importances, types, selfinter=True):
        home_imps = importances * self.type_indices(types, self.home_codes)
        home_sum = home_imps.sum()
        if home_sum == 0: # no home anchors
            return numpy.zeros(len(types))
        rels = (
            home_imps[:,numpy.newaxis] * importances[numpy.newaxis,:]
        ) / (
            # we know this is nonzero because home_sum = k * sum(imps)
            home_sum * importances.sum()
        )
        if not selfinter:
            numpy.fill_diagonal(rels, 0)
            return rels / rels.sum()
        else:
            return rels


class HomeWorkRelationGenerator(RelationGenerator):
    def __init__(self, home_codes=['k', 'm'], work_codes=['t', 'm']):
        self.home_codes = home_codes
        self.work_codes = work_codes
    
    def relate(self, importances, types, selfinter=True):
        homes = self.type_indices(types, self.home_codes)
        works = self.type_indices(types, self.work_codes)
        boths = homes & works
        home_imps = importances * (homes - .5 * boths)
        work_imps = importances * (works - .5 * boths)
        rels = (home_imps[:,numpy.newaxis] * work_imps[numpy.newaxis,:])
        relsum = rels.sum()
        if relsum == 0:
            # no relations - only anchors of one type, rels is all zeros
            if selfinter:
                # produce only self-interactions of homes proportional to importances
                home_sum = home_imps.sum()
                return numpy.diag(home_imps / home_sum) if home_sum else rels
            else:
                return rels
        else:
            return rels / relsum
            
        
if __name__ == '__main__':
    imps = numpy.array([3,2,2,1,1,1])
    codes = numpy.array(['k','t','','','',''])
    gens = [
        GeneralRelationGenerator(),
        HomeBaseRelationGenerator(),
        HomeWorkRelationGenerator(),
    ]
    for selfinter in (False, True):
        print('si', selfinter)
        for gen in gens:
            print(gen)
            print((gen.relate(imps, codes, selfinter=selfinter) * 100).astype(int))
        print()
            