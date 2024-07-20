

def compute(rank: list, groundtruth: list)->object:
    """
    This function generates a TOC object

    Parameters:
    -----------
    rank: array
        Independent variable

    groundtruth:
        Binary dependent variable. 0 is absence, 1 is presence.


    Returns:
    --------

    T: TOC object

    
    
    Example:
    --------
    >>> rank = [0,1,2,3,4,5,6]
    >>> groundTruth = [1, 0, 1, 1, 0, 1, 0]
    >>> toc.compute(np.array(rank), np.array(groundTruth))
    {'ndata': 7,
    'type': 'TOC',
    'npos': 4,
    'TP+FP': array([0, 1, 2, 3, 4, 5, 6, 7]),
    'TP': array([0, 1, 1, 2, 3, 3, 4, 4]),
    'thresholds': array([1.e-06, 0.e+00, 1.e+00, 2.e+00, 3.e+00, 4.e+00, 5.e+00, 6.e+00]),
    'areaRatio': 0.6666666666666666}
    """

    import numpy as np 
    T=dict()
    
    #Sorting the classification rank and getting the indices
    indices=sorted(range(len(rank)),key=lambda index: rank[index],reverse=True)    
    
    #Data size, this is the total number of samples
    T['ndata']=n=len(rank)
    T['type']='TOC'
    
    #This is the number of class 1 in the input data
    T['npos']=P=sum(groundtruth==1)
    T['TP+FP']=np.append(np.array(range(n)),n)
    T['TP']=np.append(0,np.cumsum(groundtruth[indices]))
    T['thresholds']=np.append(rank[indices[0]]+1e-6,rank[indices])
    T['areaRatio']=(sum(T['TP'])-0.5*T['TP'][-1]-(P*P/2))/((n-P)*P)

    return T    
    
    
def normalize(T: object)-> object:
    """
    This method returns a TOC object mapped to 0-1 in Hits+ False Alarms and Hits Axes


    Example:
    --------
    
     

    """
    T['TP+FP']=T['TP+FP']/T['ndata']
    T['TP']=T['TP']/T['npos']
    T['type']='normalized'
    return T