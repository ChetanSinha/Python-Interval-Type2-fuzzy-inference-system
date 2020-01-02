import numpy as np

def trilmf(x, abch):
    '''
    returns the triangular lower membership values.
    
    keyword arguments:
    
    x -- x range of the variable
    abch -- list of values with upper bound included
    '''
    
    assert len(abch) == 4, 'abc parameter must have exactly three elements.'
    a, b, c, h = np.r_[abch]     # Zero-indexing in Python
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = np.zeros(len(x))

    # Left side
    if a != b:
        idx = np.nonzero(np.logical_and(a < x, x < b))[0]
        y[idx] = ((x[idx] - a) / float(b - a)) * h #left side bounded by height h

    # Right side
    if b != c:
        idx = np.nonzero(np.logical_and(b < x, x < c))[0]
        y[idx] = ((c - x[idx]) / float(c - b)) * h #right side bounded by height h

    idx = np.nonzero(x == b)
    y[idx] = h #center part at height h
    return y



def traplmf(x, abcdh):
    '''
    returns the trapzoidal lower membership values.
    
    keyword arguments:
    
    x -- x range of the variable
    abcdh -- list of values with upper bound included
    '''
    
    assert len(abcdh) == 5, 'abcd parameter must have exactly four elements.'
    a, b, c, d, h = np.r_[abcdh]
    assert a <= b and b <= c and c <= d, 'abcd requires the four elements a <= b <= c <= d.'
    y = np.full(np.shape(x),h)

    idx = np.nonzero(x <= b)[0]
    y[idx] = triLmf(x[idx], np.r_[a, b, b, h])

    idx = np.nonzero(x >= c)[0]
    y[idx] = triLmf(x[idx], np.r_[c, c, d, h])

    idx = np.nonzero(x < a)[0]
    y[idx] = np.zeros(len(idx))

    idx = np.nonzero(x > d)[0]
    y[idx] = np.zeros(len(idx))

    return y



def gausslmf(x, mean, sigma, h):
    '''
    returns the gaussian lower membership values.
    
    keyword arguments:
    
    x -- x range of the variable
    mean -- mean of the graph
    sigma -- standard deviation along the range
    h -- upper bound of the LMF
    '''
    return (np.exp(-((x - mean)**2.) / (2 * sigma**2.)))*h
