import numpy as np
import random

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

n=100

lines= []
file=open('data_x_new.csv', 'w')
file.write("x,t,r1,r2,r3,r4\n")
for v in range(n):

    x=20*random.random()-10
    t=0 #4*random.random()-2

    f=np.exp((0-1j)*(k**2)*t + k*x + nf)
    ff=f.conjugate()
    g=np.exp((0-1j)*(l**2)*t + l*x + ng)
    gg=g.conjugate()
    phi1=1+2*f*ff/((k+kk)**2)
    phi2=1+2*g*gg/((l+ll)**2)

    
    f1=f.real
    f2=-f.imag
    g1=g.real
    g2=-g.imag

    result1= f1/phi1 + g1/phi2
    result2= f2/phi1 - g2/phi2

    result3= f2/phi1 + g2/phi2
    result4=-f1/phi1 + g1/phi2

    line = "{},{},{},{},{},{}".format(x, t, result1.real, result2.real, result3.real, result4.real).replace("(","").replace(")","")

    print(line)
    lines.append(line)


file.write("\n".join(lines))



filet=open('data_t_new.csv', 'w')
filet.write("t,x1,x2\n")
linest=[]
for v in range(n):

    x1=10
    x2=-10

    t=4*random.random()-2


    line = "{},{},{}".format(t, x1, x2).replace("(","").replace(")","")

    print(line)
    linest.append(line)


filet.write("\n".join(linest))


filexx=open('data_x_x.csv', 'w')
filexx.write("x,t,r1,r2,r3,r4\n")
linesxx=[]
for v in range(5000):

    x=20*random.random()-10
    t=4*random.random()-2

    f=np.exp((0-1j)*(k**2)*t + k*x + nf)
    ff=f.conjugate()
    g=np.exp((0-1j)*(l**2)*t + l*x + ng)
    gg=g.conjugate()
    phi1=1+2*f*ff/((k+kk)**2)
    phi2=1+2*g*gg/((l+ll)**2)

    
    f1=f.real
    f2=-f.imag
    g1=g.real
    g2=-g.imag

    result1= f1/phi1 + g1/phi2
    result2= f2/phi1 - g2/phi2

    result3= f2/phi1 + g2/phi2
    result4=-f1/phi1 + g1/phi2

    line = "{},{},{},{},{},{}".format(x, t, result1.real, result2.real, result3.real, result4.real).replace("(","").replace(")","")

    print(line)
    linesxx.append(line)


filexx.write("\n".join(linesxx))
