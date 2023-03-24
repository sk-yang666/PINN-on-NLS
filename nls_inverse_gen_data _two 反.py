import numpy as np
from pyDOE import lhs
import random

n=200

lines= []
file=open('1data_x_new.csv', 'w')
file.write("x,t,r1,r2,r3,r4\n")
k1=1.5
k2=1
l1=1.5
l2=-1
nf1=0
nf2=0
ng1=0
ng2=0
k=complex(k1,-k2)
kk=k.conjugate()
l=complex(l1,-l2)
ll=l.conjugate()
nf=complex(nf1,-nf2)
nff=nf.conjugate()
ng=complex(ng1,-ng2)
ngg=ng.conjugate()
X=lhs(1,samples=n)*20-10
for v in range(n):

    x=float(X[v])
    t=0 #4*random.random()-2
    f=np.exp((0-1j)*(k**2)*(-t) + k*x+ nf)
    ff=f.conjugate()
    g=np.exp((0-1j)*(l**2)*(-t) + l*x+ ng)
    gg=g.conjugate()
    phi1=1+2*f*ff/((k+kk)**2)
    phi2=1+2*g*gg/((l+ll)**2)
    q1=f/phi1+g/phi2
    q2=1j*(f/phi1-g/phi2)
    result1= q1.real
    result2= -q1.imag
    result3= q2.real
    result4=-q2.imag

    line = "{},{},{},{},{},{}".format(x, t, result1.real, result2.real, result3.real, result4.real).replace("(","").replace(")","")

    print(line)
    lines.append(line)


file.write("\n".join(lines))



filet=open('1data_t_new.csv', 'w')
filet.write("t,x1,x2\n")
linest=[]
T=lhs(1,samples=n)*4-2
for v in range(n):

    x1=10
    x2=-10

    t=float(T[v])



    line = "{},{},{}".format(t, x1, x2).replace("(","").replace(")","")

    print(line)
    linest.append(line)


filet.write("\n".join(linest))


filexx=open('1data_x_x.csv', 'w')
filexx.write("x,t,r1,r2,r3,r4\n")
linesxx=[]
index=8000
X=lhs(1,samples=index)*20-10
T=lhs(1,samples=index)*4-2
for v in range(index):

    x=float(X[v])
    t=float(T[v])

    f=np.exp((0-1j)*(k**2)*(-t) + k*x+ nf)
    ff=f.conjugate()
    g=np.exp((0-1j)*(l**2)*(-t) + l*x+ ng)
    gg=g.conjugate()
    phi1=1+2*f*ff/((k+kk)**2)
    phi2=1+2*g*gg/((l+ll)**2)
    q1=f/phi1+g/phi2
    q2=1j*(f/phi1-g/phi2)
    result1= q1.real
    result2= -q1.imag
    result3= q2.real
    result4=-q2.imag


    line = "{},{},{},{},{},{}".format(x, t, result1.real, result2.real, result3.real, result4.real).replace("(","").replace(")","")

    print(line)

    linesxx.append(line)


filexx.write("\n".join(linesxx))

