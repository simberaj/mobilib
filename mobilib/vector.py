
import numpy

def length(vectors):
    return numpy.linalg.norm(vectors, axis=-1)
    
def angle(vectors):
    if len(vectors.shape) == 2:
        return numpy.arctan2(*vectors.T[::-1])
    else:
        slicer = (slice(None), ) * (len(vectors.shape) - 1)
        return numpy.arctan2(vectors[slicer + (1,)], vectors[slicer + (0,)])
