#!/usr/bin/env python3

import numpy as np 
import time

#Implementation for computing the
#TOC= Total Operating Charcateristic Curve
#Author:S. Ivvan Valdez.  Programador del módulo
#Author: R. Lopez-Farias. Programador del módulo y tester como herramienta 
#Centro de Investigación en Ciencias de Información Geoespacial AC
#Querétaro, México.



#This function computes the coordinates in:
#   The x axis= Hits + False Alarms, this can be accessed as toc['TP+FP']
#   The y axis= Hits, this can be accessed as toc['TP']
#   The threshold for each coordinate
#   The area ratio, that is to say: Area under the curve/ Parellelepiped area


# from memory_profiler import profile
# @profile(precision=12)
#Computes the Total operating Characteristic given a rank and ground truth it assumes that the largest is the rank the most probable class 1 is.
#rank is the numpy rank array 
#groundtruth numpy array of +binary labels [0,1]


def compute(rank, groundtruth):

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


#This method normalize/scales the TOC into the range of [0,1] for both axis
def normalize(T):
    T['TP+FP']=T['TP+FP']/T['ndata']
    T['TP']=T['TP']/T['npos']
    T['type']='normalized'
    return T


#This method generates a curve with a similar shape than T1 but with number of samples of T2.
#The normalized area of the resulting curve approximates that of T1
def resample(T1,T2):
    n1=T1['ndata']
    n2=T2['ndata']
    pn1=T1['npos']/n1
    pn2=T2['npos']/n2

    if (n2<n1):
        print("The second curve T2 most have more elements than the first.")

    T3=T2.copy()
    T3['npos']=pn1*n2
    n3=n2
    T3['TP']=np.zeros(np.size(T2['TP']))
    dfp=(T3['npos'])/(T1['npos'])
    df=(T3['npos']*n1)/(T1['npos']*n3)
    j=1
    for i in range(n1):
        while(j<=n2  and  (T3['TP+FP'][j]/n2)<=(T1['TP+FP'][i+1]/n1)):
            h=T1['TP'][i]*dfp
            T3['TP'][j]=T1['TP'][i]*dfp+(float(j)-(i*n3/n1))*df*float(T1['TP'][i+1]-T1['TP'][i])
            j+=1

    T3['npos']=np.round(T3['npos'])
    T3['TP']=np.round(T3['TP'])
    n3=T3['ndata']
    T3['areaRatio']=(sum(T3['TP'])-0.5*T3['TP'][-1]-(T3['npos']*T3['npos']/2))/((n3-T3['npos'])*T3['npos'])
    return(T3)

#This method computes the difference between two TOC curves, the 
def TOCdiff(T1,T2):
    n1=T1['ndata']
    n2=T2['ndata']
    if (n1<n2):
        Tw1=resample(T1,T2)
        Tw2=T2
    if (n2<n1):
        Tw2=resample(T2,T1)
        Tw1=T1
    n1=Tw1['ndata']
    n2=Tw2['ndata']
    
    pn1=Tw1['npos']/n1    
    pn2=Tw2['npos']/n2
    Tdiff=Tw1.copy()
    Tarea=((1-pn1)+(1.0-pn2))/2.0
    Tdiff['TP']=Tw1['TP']/Tw1['npos']-Tw2['TP']/Tw2['npos']
    Tdiff['npos']=max(abs(Tdiff['TP']))
    Tdiff['areaRatio']=sum(abs(Tdiff['TP']))/(Tarea*n1)
    Tdiff['type']='diff'
    return(Tdiff)

#This methos graphically compare two TOCs curves
def plotComp(T1,T2,TOCname1='TOC1',TOCname2='TOC2',title="default",filename='',height=1800,width=1800,dpi=300):
    import numpy as np 
    import matplotlib.pyplot as plt
    if (T1['type']!='TOC' or T2['type']!='TOC'):
        return;
    n1=T1['ndata']
    n2=T2['ndata']
    pn1=T1['npos']/n1
    pn2=T2['npos']/n2
    rx1=np.array([0,pn1,1,1-pn1])
    rx2=np.array([0,pn2,1,1-pn2])
    ry=np.array([0,1,1,0])
    plt.clf()
    fig=plt.figure(figsize=(height/dpi, width/dpi), dpi=dpi)
    plt.ylim(0, 1.01)
    plt.xlim(0, 1.01)
    plt.xlabel("Hits+False Alarms")
    plt.ylabel("Hits")
    if title=='default':
        plt.title("Comparison"+" "+TOCname1+' vs '+TOCname2)
    else:
        plt.title(title)
    plt.text(0.5,0.025,'AUC-'+TOCname2+'=')
    plt.text(0+0.675,0.025,str(round(T2['areaRatio'],4)))
    plt.text(0.5,0.075,'AUC-'+TOCname1+'=')
    plt.text(0+0.675,0.075,str(round(T1['areaRatio'],4)))
    #str(T1['areaRatio'])
    plt.plot(rx1, ry,'r--')
    plt.plot(rx2, ry,'b--')
    plt.plot(np.array([0,1]),np.array([0,1]),'k-.')
    plt.plot(T1['TP+FP']/n1,T1['TP']/T1['npos'],'r-',label=TOCname1,linewidth=3)
    plt.plot(T2['TP+FP']/n2,T2['TP']/T2['npos'],'b-',label=TOCname2,linewidth=3)
    plt.legend(loc='lower right')
    if (filename!=''):
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()
    
    
    
    
    

#This function plots the TOC to the terminal or a file
def plot(T, filename = '',title='default',TOCname='TOC',normalized=False,height=1800,width=1800,dpi=300):


    import matplotlib.pyplot as plt
    import numpy as np
    plt.clf()
    if (filename!=''):
        fig=plt.figure(figsize=(height/dpi, width/dpi), dpi=dpi)
    plt.xlabel("Hits+False Alarms")
    plt.ylabel("Hits")    
    if (T['type']=='TOC'):
        if (title=='default'):
                title="Total operating characteristic"
        plt.title(title)
        if(normalized):        
            rx=np.array([0,T['npos']/T['ndata'],1,1-T['npos']/T['ndata']])
            ry=np.array([0,1,1,0])            
            plt.ylim(0, 1.01)
            plt.xlim(0, 1.01)            
            plt.text(0.575,0.025,'AUC=')
            plt.text(0.675,0.025,str(round(T['areaRatio'],4)))
            plt.plot(rx, ry,'b--')
            plt.plot(T['TP+FP']/T['ndata'],T['TP']/T['npos'],'r-',label=TOCname,linewidth=3)
        else:
            rx=np.array([0,T['npos'],T['ndata'],T['ndata']-T['npos']])
            ry=np.array([0,T['npos'],T['npos'],0])
            plt.ylim(0, 1.01*T['npos'])
            plt.xlim(0, 1.01*T['ndata'])
            plt.text(0.575*T['ndata'],0.025*T['npos'],'AUC=')
            plt.text(0.675*T['ndata'],0.025*T['npos'],str(round(T['areaRatio'],4)))
            plt.plot(rx, ry,'b--')
            plt.plot(T['TP+FP'],T['TP'],'r-',label="TOC")
        plt.legend(loc='lower right')
    if(T['type']=='diff'):
        if (title=='default'):
            title="Difference between 2 TOCs"
        plt.title(title)
        plt.ylim(-1.01, 1.01)
        plt.xlim(-0.01, 1.01)            
        plt.text(0.575,0.025,'AUC=')
        plt.text(0.675,0.025,str(round(T['areaRatio'],4)))
        plt.plot(T['TP+FP']/T['ndata'],T['TP']/T['npos'],'r-',label=TOCname,linewidth=3)
        plt.legend(loc='lower right')       
    if(T['type']=='QSpline'):
        if (title=='default'):
            title="Approximation of TOC via qspline"
        plt.title(title)
        rx=np.array([0,T['npos'],T['ndata'],T['ndata']-T['npos']])
        ry=np.array([0,T['npos'],T['npos'],0])
        plt.ylim(0, 1.01*T['npos'])
        plt.xlim(0, 1.01*T['ndata'])
        plt.text(0.575*T['ndata'],0.025*T['npos'],'AUC=')
        plt.text(0.675*T['ndata'],0.025*T['npos'],str(round(T['areaRatio'],4)))
        plt.plot(rx, ry,'b--')
        plt.plot(T['TP+FP'],T['TP'],'r-',label="TOC")
    if(T['type']=='vector'):
        if (title=='default'):
            title="Approximation of TOC via average vectors"
        plt.title(title)
        rx=np.array([0,T['npos'],T['ndata'],T['ndata']-T['npos']])
        ry=np.array([0,T['npos'],T['npos'],0])
        plt.ylim(0, 1.01*T['npos'])
        plt.xlim(0, 1.01*T['ndata'])
        plt.text(0.575*T['ndata'],0.025*T['npos'],'AUC=')
        plt.text(0.675*T['ndata'],0.025*T['npos'],str(round(T['areaRatio'],4)))
        plt.plot(rx, ry,'b--')
        plt.plot(T['TP+FP'],T['TP'],'r-',label="TOC")
    if(T['type']=='nvector'):
        if (title=='default'):
            title="Approximation of normalized TOC via average vectors"
        plt.title(title)
        rx=np.array([0,T['npos']/T['ndata'],1,1-T['npos']/T['ndata']])
        ry=np.array([0,1,1,0])  
        plt.ylim(0, 1.01)
        plt.xlim(0, 1.01)    
        plt.text(0.575,0.025,'AUC=')
        plt.text(0.675,0.025,str(round(T['areaRatio'],4)))
        plt.plot(rx, ry,'b--')
        plt.plot(T['TP+FP'],T['TP'],'r-',label="TOC")        
    if (T['type']=='normalized'):
        if (title=='default'):
                title="Total operating characteristic"
        plt.title(title)
        rx=np.array([0,T['npos']/T['ndata'],1,1-T['npos']/T['ndata']])
        ry=np.array([0,1,1,0])            
        plt.ylim(0, 1.01)
        plt.xlim(0, 1.01)            
        plt.text(0.575,0.025,'AUC=')
        plt.text(0.675,0.025,str(round(T['areaRatio'],4)))
        plt.plot(rx, ry,'b--')
        plt.plot(T['TP+FP'],T['TP'],'r-',label=TOCname,linewidth=3)
        plt.legend(loc='lower right')
    if (filename!=''):
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()
        
        


def qspline(T, nsegments=30,verbose=0):
    if ((2*nsegments+1)>T['ndata']):
        print('ERROR: The number of segments have to be less than half the number of data in the TOC curve.')
    #nsegments=30
    n=T['ndata']+1
    ns=nsegments
    y=np.zeros(2*ns+1)
    y[0]=0
    y[-1]=T['TP'][-1]
    itmax=n+1000
    it=0
    error=100.0
    ya=y.copy()
    while(error>1.0e-6 and it<itmax):
        xkinf=-1
        xksup=-1
        x2kinf=-1
        x2ksup=-1
        for i in range(0,2*ns-2,2):
            eip1=np.zeros(11)
            eip2=np.zeros(11)
            b=np.zeros(2)
            xkinf=round(i*T['ndata']/(2.0*ns)+0.5)
            if (xkinf==xksup):
                xkinf+=1
            xksup=int((i+2.0)*T['ndata']/(2.0*ns))
            xk=(np.linspace(xkinf,xksup,xksup+1-xkinf)).astype('int64')
            yk=T['TP'][xk]
            xi=i*T['ndata']/(2.0*ns)
            xip1=(i+1.0)*T['ndata']/(2.0*ns)
            xip2=(i+2.0)*T['ndata']/(2.0*ns)
            Diip1=(xk-xip1)/(xi-xip1)
            Diip2=(xk-xip2)/(xi-xip2)
            Dip1i=(xk-xi)/(xip1-xi)
            Dip1ip2=(xk-xip2)/(xip1-xip2)
            Dip2i=(xk-xi)/(xip2-xi)
            Dip2ip1=(xk-xip1)/(xip2-xip1)
            
            #Para la ecuacion de yi+1
            eip1[0]=sum(Diip1*Diip2*Dip1i*Dip1ip2) #yi
            eip1[1]=sum(Dip1i**2*Dip1ip2**2) #yi+1
            eip1[2]=sum(Dip2i*Dip2ip1*Dip1i*Dip1ip2) #yi+2
            #Para la ecuacion de yi+2
            eip2[0]=sum(Diip1*Diip2*Dip2i*Dip2ip1) #yi
            eip2[1]=sum(Dip1i*Dip1ip2*Dip2i*Dip2ip1) #yi+1
            eip2[2]=sum(Dip2i**2.0*Dip2ip1**2.0) #yi+2    

            b[0]+=sum(Dip1i*Dip1ip2*yk)
            b[1]+=sum(Dip2i*Dip2ip1*yk)

            xip3=(i+3.0)*T['ndata']/(2.0*ns)
            xip4=(i+4.0)*T['ndata']/(2.0*ns)
            x2kinf=round((i+2.0)*T['ndata']/(2.0*ns)+0.5)
            if (x2kinf==x2ksup):
                x2kinf+=1
            x2ksup=int((i+4.0)*T['ndata']/(2.0*ns))
            xk=np.linspace(x2kinf,x2ksup,x2ksup+1-x2kinf).astype('int64')
            yk=T['TP'][xk]
            Dip2ip3=(xk-xip3)/(xip2-xip3)
            Dip2ip4=(xk-xip4)/(xip2-xip4)
            Dip3ip2=(xk-xip2)/(xip3-xip2)
            Dip3ip4=(xk-xip4)/(xip3-xip4)
            Dip4ip2=(xk-xip2)/(xip4-xip2)
            Dip4ip3=(xk-xip3)/(xip4-xip3)
            #Para la ecuacion de yi+2
            eip2[2]+=sum(Dip2ip3**2*Dip2ip4**2) #yi+2
            eip2[3]+=sum(Dip3ip2*Dip3ip4*Dip2ip3*Dip2ip4) #yi+3
            eip2[4]+=sum(Dip4ip2*Dip4ip3*Dip2ip3*Dip2ip4)#yi+4
            
            b[1]+=sum(Dip2ip3*Dip2ip4*yk)

            k=0
            sm=np.zeros(2)
            for j in range(i,(i+5)):
                if(j!=(i+1)):
                    sm[0]+=y[j]*eip1[k]
                if(j!=(i+2)):
                    sm[1]+=y[j]*eip2[k]
                k+=1
                
            y[i+1]=(b[0]-sm[0])/(eip1[1]+1.0e-36)
            y[i+2]=(b[1]-sm[1])/(eip2[2]+1.0e-36)

        i=2*ns-2
        eip1=np.zeros(11)
        b=np.zeros(2)
        xkinf=round(i*T['ndata']/(2.0*ns)+0.5)
        if (xkinf==xksup):
            xkinf+=1
            
        xksup=int((i+2.0)*T['ndata']/(2.0*ns))-1
        xk=(np.linspace(xkinf,xksup,xksup+1-xkinf)).astype('int64')
        yk=T['TP'][xk]
        xi=i*T['ndata']/(2.0*ns)
        xip1=(i+1.0)*T['ndata']/(2.0*ns)
        xip2=(i+2.0)*T['ndata']/(2.0*ns)
        Diip1=(xk-xip1)/(xi-xip1)
        Diip2=(xk-xip2)/(xi-xip2)
        Dip1i=(xk-xi)/(xip1-xi)
        Dip1ip2=(xk-xip2)/(xip1-xip2)
        Dip2i=(xk-xi)/(xip2-xi)
        Dip2ip1=(xk-xip1)/(xip2-xip1)
        #Para la ecuacion de yi+1
        eip1[0]=sum(Diip1*Diip2*Dip1i*Dip1ip2) #yi
        eip1[1]=sum(Dip1i**2*Dip1ip2**2) #yi+1
        eip1[2]=sum(Dip2i*Dip2ip1*Dip1i*Dip1ip2) #yi+2
        b[0]+=sum(Dip1i*Dip1ip2*yk)
        k=0
        sm=np.zeros(2)
        for j in range(i,(i+3)):
            if(j!=(i+1)):
                sm[0]+=y[j]*eip1[k]
            k+=1

        y[i+1]=(b[0]-sm[0])/(eip1[1]+1.0e-36)
        error=sum(abs(ya-y))
        if (verbose==1):
            print('error=', error)
        ya=y.copy()
        
    area=0.0
    x=np.linspace(0,T['ndata'],2*ns+1)
    
    for i in range(0,2*ns,2):
        axi=(y[i]*(-(x[i+2]**3-3*x[i+1]*x[i+2]**2)/6-((6*x[i]*x[i+1]-3*x[i]**2)*x[i+2]-3*x[i]**2*x[i+1]+2*x[i]**3)/6))/((x[i]-x[i+1])*(x[i]-x[i+2]))
        axip1=(y[i+1]*(-(x[i+2]**3-3*x[i]*x[i+2]**2)/6-(3*x[i]**2*x[i+2]-x[i]**3)/6))/((x[i+1]-x[i])*(x[i+1]-x[i+2]))
        axip2=(((2*x[i+2]**3+(-3*x[i+1]-3*x[i])*x[i+2]**2+6*x[i]*x[i+1]*x[i+2])/6-(3*x[i]**2*x[i+1]-x[i]**3)/6)*y[i+2])/((x[i+2]-x[i])*(x[i+2]-x[i+1]))
        area+=axi+axip1+axip2
        
    TS=dict()
    #Data size, this is the total number of samples
    TS['ndata']=T['ndata']
    TS['nsp']=ns*2
    TS['type']='QSpline'
    #This is the number of class 1 in the input data
    TS['npos']=T['npos']
    TS['TP+FP']=np.linspace(0,T['ndata'],2*ns+1)
    TS['TP']=y
    TS['thresholds']=T['thresholds'][TS['TP+FP'].astype('int64')]+(T['thresholds'][(TS['TP+FP']+0.5).astype('int64')]-T['thresholds'][TS['TP+FP'].astype('int64')])*(TS['TP+FP']-TS['TP+FP'].astype('int64'))
    TS['areaRatio']=(area-(T['npos']*T['npos']/2))/((n-T['npos'])*T['npos']) 
    TS['error']=error
    return(TS)


def vector(T, nmeans=30,verbose=0):
    #nsegments=30
    n=T['ndata']+1
    if (nmeans<1 or nmeans>(n/2)):
        print('ERROR: The number of means has to be less than half the number of data in the TOC curve.')
    ns=nmeans
    y=np.zeros(ns+2)
    x=np.zeros(ns+2)
    y[0]=0
    x[0]=0
    area=0.0
    for i in range(ns):
        ini=int(i*n/ns)
        ifi=int((i+1)*n/ns)
        xk=(np.linspace(ini,ifi-1,ifi-ini)).astype('int64')
        y[i+1]=np.mean(T['TP'][xk])
        x[i+1]=np.mean(xk)
        area+=(y[i+1]+y[i])/2
        
    TV=dict()        
    TV['type']='vector'
    if (T['type']!='normalized'):
        y[-1]=T['npos']
        x[-1]=T['ndata']
    else:
        x=x/T['ndata']
        x[-1]=1
        y[-1]=1
        TV['type']='nvector'
    area+=(y[-1]+y[-2])/2
    area=area*(x[-1])/(ns+1)
    # print ('area:'+str(area)+'\n')
    #Data size, this is the total number of samples
    TV['ndata']=T['ndata']
    TV['nmeans']=ns
    #This is the number of class 1 in the input data
    TV['npos']=T['npos']
    TV['TP+FP']=x
    TV['TP']=y
    TV['thresholds']=T['thresholds'][TV['TP+FP'].astype('int64')]+(T['thresholds'][(TV['TP+FP']+0.5).astype('int64')]-T['thresholds'][TV['TP+FP'].astype('int64')])*(TV['TP+FP']-TV['TP+FP'].astype('int64'))
    if (T['type']=='normalized'):
        TV['areaRatio']=(area-0.5*TV['npos']/TV['ndata'])/(1-TV['npos']/TV['ndata'])
    else:
        TV['areaRatio']=(area-0.5*TV['npos']*TV['npos'])/(TV['npos']*TV['ndata']-TV['npos']*TV['npos'])
        
    return(TV)





def density(T, smoothing = 0, order = 1, verbose = 0):

    import matplotlib.pyplot as plt
    import numpy as np
    # area=0.0
    
    density=dict()
    n = np.shape(T['TP+FP'])[0]
    df = np.zeros(n) 
    y = T['TP']
    h = T['TP+FP'][1]-T['TP+FP'][0]
    if (T['type']=='normalized' or T['type']=='nvector'):
        x2=T['TP+FP']*T['ndata']-T['TP']*T['npos']
    else:
        x2=T['TP+FP']-T['TP']
    x2=x2/np.max(x2)
    h2=x2[1]-x2[0]
    
    if (order==1):
        df[1:-1]=(y[2:]-y[0:-2])/(2*h)
        df[0]=df[1]
        df[-1]=df[-2]
    if (order==2):
        df[2:-2]=-(y[4:]-y[2:-2])/(12*h)+(y[0:-4]-y[2:-2])/(12*h)+2*(y[3:-1]-y[2:-2])/(3*h)-2*(y[1:-3]-y[2:-2])/(3*h)
        df[0]=df[2]
        df[1]=df[2]
        df[-1]=df[-3]
        df[-2]=df[-3]
        
    density=dict()       
    density['df']=df
    density['h']=h
    sm=np.zeros(n) 
    
    if (smoothing>0):
        sm[0:smoothing]=  np.mean(df[0:smoothing])  
        sm[(n-smoothing):n]=  np.mean(df[(n-smoothing):n])  
        for i in range(smoothing,n-smoothing):
            sm[i]=np.mean(df[(i-smoothing):(i+smoothing)])  
        density['smooth']=sm
    else:
        density['smooth']=df
        
#####Aqui voy!        
    areaS=np.zeros(n)    
    area=np.zeros(n)
    area[:]=np.concatenate(([0],np.cumsum((df[0:-1]+df[1:])/2)))*h
    areaS[:]=np.concatenate(([0],np.cumsum((sm[0:-1]+sm[1:])/2)))*h
    density['area']=area
    density['areaSm']=areaS
    density['TP+FP']=np.zeros(n)
    density['TP+FP'][:]=T['TP+FP']
    density['FP']=np.zeros(n)
    density['FP'][:]=x2
    density['factorFP']=h2/h

    return(density)














