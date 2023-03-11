import numpy as np
from pyDOE import lhs
import random

n=200

lines= []
file=open('data_x_new.csv', 'w')
file.write("x,t,r1,r2,r3,r4\n")
c11=1
c21=2
c31=-2
c41=4
lambda1=0.01-1j

v_1=(c31.conjugate()**2+c41.conjugate()**2)*(c11*c31+c21*c41)
v_2=(c11**2+c21**2)*(c11.conjugate()**2+c21.conjugate()**2)
v_3=(c31.conjugate()**2+c41.conjugate()**2)*(c21*c31-c11*c41)

eta1=np.log((c11**2+c21**2)*(c11.conjugate()*c31.conjugate()+c21.conjugate()*c41.conjugate())/v_1)/2
eta2=np.log(v_2/((c11*c11.conjugate()+c21*c21.conjugate())*(c31*c31.conjugate()+c41*c41.conjugate())))/2
eta3=np.log((c11**2+c21**2)*(c21*c31-c11*c41)/v_3)/2
gamma1=(c31**2+c41**2)*(c31.conjugate()**2+c41.conjugate()**2)
X=lhs(1,samples=n)*20-10
for v in range(n):

    x=float(X[v])
    t=0 #4*random.random()-2
    xi1=-2*1j*lambda1**2*t-1j*lambda1*x
    q1=4*v_1*1j*(lambda1.conjugate()-lambda1)*np.exp(3*xi1+xi1.conjugate()+eta1)*np.cosh(xi1+xi1.conjugate()+eta1)/(2*v_2*np.exp(3*(xi1+xi1.conjugate())+eta2)*np.cosh(xi1+xi1.conjugate()+eta2)+gamma1)
    q2=4*v_3*1j*(lambda1-lambda1.conjugate())*np.exp(3*xi1+xi1.conjugate()+eta3)*np.cosh(xi1+xi1.conjugate()+eta3)/(2*v_2*np.exp(3*(xi1+xi1.conjugate())+eta2)*np.cosh(xi1+xi1.conjugate()+eta2)+gamma1)


    result1= q1.real
    result2= -q1.imag
    result3= q2.real
    result4=-q2.imag

    line = "{},{},{},{},{},{}".format(x, t, result1.real, result2.real, result3.real, result4.real).replace("(","").replace(")","")

    print(line)
    lines.append(line)


file.write("\n".join(lines))



filet=open('data_t_new.csv', 'w')
filet.write("t,x1,x2\n")
linest=[]
T=lhs(1,samples=n)*4-2
for v in range(n):

    x1=5*np.pi
    x2=-5*np.pi

    t=float(T[v])



    line = "{},{},{}".format(t, x1, x2).replace("(","").replace(")","")

    print(line)
    linest.append(line)


filet.write("\n".join(linest))


filexx=open('data_x_x.csv', 'w')
filexx.write("x,t,r1,r2,r3,r4\n")
linesxx=[]
index=8000
X=lhs(1,samples=index)*20-10
T=lhs(1,samples=index)*4-2
for v in range(index):

    x=float(X[v])
    t=float(T[v])

    xi1=-2*1j*lambda1**2*t-1j*lambda1*x
    q1=4*v_1*1j*(lambda1.conjugate()-lambda1)*np.exp(3*xi1+xi1.conjugate()+eta1)*np.cosh(xi1+xi1.conjugate()+eta1)/(2*v_2*np.exp(3*(xi1+xi1.conjugate())+eta2)*np.cosh(xi1+xi1.conjugate()+eta2)+gamma1)
    q2=4*v_3*1j*(lambda1-lambda1.conjugate())*np.exp(3*xi1+xi1.conjugate()+eta3)*np.cosh(xi1+xi1.conjugate()+eta3)/(2*v_2*np.exp(3*(xi1+xi1.conjugate())+eta2)*np.cosh(xi1+xi1.conjugate()+eta2)+gamma1)


    result1= q1.real
    result2= -q1.imag
    result3= q2.real
    result4=-q2.imag

    line = "{},{},{},{},{},{}".format(x, t, result1.real, result2.real, result3.real, result4.real).replace("(","").replace(")","")

    print(line)

    linesxx.append(line)


filexx.write("\n".join(linesxx))

