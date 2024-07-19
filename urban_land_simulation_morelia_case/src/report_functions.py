def simulate(xtrain, ytrain, xvalid, yvalid, Models):
    dict ={}
    for i, M in enumerate(Models):
        dict["m_{i}_haty_train".format(i=i)] = M.predict(xtrain)
        dict["m_{i}_haty_valid".format(i=i)] = M.predict(xvalid)
    
    return dict