import numpy



def rels(strengths):
    denom = 0.5 * (sum(strengths) ** 2 - sum(strength ** 2 for strength in strengths))
    for i in range(len(strengths)):
        for j in range(i+1, len(strengths)):
            yield i, j, strengths[i] * strengths[j] / denom
            
print(list(rels([21,7,2,2])))



class RelationGenerator:
    def relate(importances, types):
        raise NotImplementedError
        
        
class ImportanceRelationGenerator:
    def relate(self, importances, types):
        rels = importances[:,numpy.newaxis] * importances[numpy.newaxis,:]
        numpy.fill_diagonal(rels, 0)
        return rels / rels.sum()
        
class HomeRelationGenerator:
    def __init__(self, home_types=['k', 'm']):
        self.home_types = home_types
    
    def relate(self, importances, types):
        home_indices = numpy.stack([
            types == type for type in self.home_types
        ]).any(axis=0)
        if home_indices.sum() == 1: # single home location
            return 