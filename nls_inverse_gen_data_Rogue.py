import numpy as np
from pyDOE import lhs
import random

n=200

lines= []
file=open('Rogue_data_x_new.csv', 'w')
file.write("x,t,r1,r2,r3,r4\n")
T=lhs(1,samples=n)*4-2
X=lhs(1,samples=n)*16-8
for v in range(n):

    x=float(X[v])
    #t=1 if random.random()<0.33 else (-1 if random.random()>0.66 else 0)
    t=0
    i = 0+1j
    delta1=((2 + np.sqrt(2))*(x**2) + 4*(3 + 2*np.sqrt(2))*(x**4) - 8*i*(3 + 2*np.sqrt(2))*(x**2)*t
            +8*(8*(3 + 2*np.sqrt(2))*(x**2) - 2 - np.sqrt(2))*(t**2) 
            - 64*i*(3 + 2*np.sqrt(2))*(t**3) + 256*(3 + 2*np.sqrt(2))*(t**4)
           )

    delta2=( -np.sqrt(2) - 2*np.sqrt(2)*(x**2) + 8*(3 + 2*np.sqrt(2))*(x**4) 
            - 8*i*(1 + np.sqrt(2) + (6+4*np.sqrt(2))*(x**2))*t 
            + 16*(np.sqrt(2) + 24*(x**2) + 16*np.sqrt(2)*(x**2))*(t**2)
            -128*i*(3 + 2*np.sqrt(2))*(t**3) + 512*(3 + 2*np.sqrt(2))*(t**4)
    )
    
    delta0=(1+ 32*(1+np.sqrt(2))*(t**2) + 4*(2+np.sqrt(2))*(x**2) +128*(3 + 2*np.sqrt(2))*(t**2)*(x**2)
            +512*(3 + 2*np.sqrt(2))*(t**4) + 8*(3 + 2*np.sqrt(2))*(x**4)
    
    )
    
    q1=np.exp(4*i*t)*(1-4*delta1/delta0)
    q2=np.exp(4*i*t)*(1-2*delta2/delta0)
    result1= q1.real
    result2= -q1.imag
    result3= q2.real
    result4=-q2.imag

    line = "{},{},{},{},{},{}".format(x, t, result1.real, result2.real, result3.real, result4.real).replace("(","").replace(")","")

    print(line)
    lines.append(line)


file.write("\n".join(lines))



filet=open('Rogue_data_t_new.csv', 'w')
filet.write("t,x1,x2\n")
linest=[]

for v in range(n):

    x1=8
    x2=-8

    t=float(T[v])



    line = "{},{},{}".format(t, x1, x2).replace("(","").replace(")","")

    print(line)
    linest.append(line)


filet.write("\n".join(linest))


# filexx=open('data_x_x.csv', 'w')
# filexx.write("x,t,r1,r2,r3,r4\n")
# linesxx=[]
# index=8000
# X=lhs(1,samples=index)*20-10
# T=lhs(1,samples=index)*4-2
# for v in range(index):

#     x=float(X[v])
#     t=float(T[v])

#     xi1=-2*1j*lambda1**2*t-1j*lambda1*x
#     q1=4*v_1*1j*(lambda1.conjugate()-lambda1)*np.exp(3*xi1+xi1.conjugate()+eta1)*np.cosh(xi1+xi1.conjugate()+eta1)/(2*v_2*np.exp(3*(xi1+xi1.conjugate())+eta2)*np.cosh(xi1+xi1.conjugate()+eta2)+gamma1)
#     q2=4*v_3*1j*(lambda1-lambda1.conjugate())*np.exp(3*xi1+xi1.conjugate()+eta3)*np.cosh(xi1+xi1.conjugate()+eta3)/(2*v_2*np.exp(3*(xi1+xi1.conjugate())+eta2)*np.cosh(xi1+xi1.conjugate()+eta2)+gamma1)


#     result1= q1.real
#     result2= -q1.imag
#     result3= q2.real
#     result4=-q2.imag

#     line = "{},{},{},{},{},{}".format(x, t, result1.real, result2.real, result3.real, result4.real).replace("(","").replace(")","")

#     print(line)

#     linesxx.append(line)


# filexx.write("\n".join(linesxx))

