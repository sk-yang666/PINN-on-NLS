"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np


def gen_traindata():

    x_lower = -10
    x_upper = 10
    t_lower = -2
    t_upper = 2
    x = np.linspace(x_lower, x_upper, 200)
    t = np.linspace(t_lower, t_upper, 200)
    u1w=(1/2) * np.cos(1.5*x-(-0.587/2)) * np.exp(2.5-1.5*x) / np.cosh(1.5*x+(-0.587/2))
    v1w=(1/2) * np.sin(1.5*x-(-0.587/2)) * np.exp(2.5-1.5*x) / np.cosh(1.5*x+(-0.587/2))
    u2w=np.cos(1.5*x-(-0.587/2)) * np.exp(2.5-1.5*x) / np.cosh(1.5*x+(-0.587/2))
    v2w=np.sin(1.5*x-(-0.587/2)) * np.exp(2.5-1.5*x) / np.cosh(1.5*x+(-0.587/2))
    
    x1 = np.linspace(x_lower, x_upper, 40000)
    #t = np.linspace(t_lower, t_upper, 200)
    u1=(1/2) * np.cos(1.5*x1-(-0.587/2)) * np.exp(2.5-1.5*x1) / np.cosh(1.5*x1+(-0.587/2))
    v1=(1/2) * np.sin(1.5*x1-(-0.587/2)) * np.exp(2.5-1.5*x1) / np.cosh(1.5*x1+(-0.587/2))
    u2=np.cos(1.5*x1-(-0.587/2)) * np.exp(2.5-1.5*x1) / np.cosh(1.5*x1+(-0.587/2))
    v2=np.sin(1.5*x1-(-0.587/2)) * np.exp(2.5-1.5*x1) / np.cosh(1.5*x1+(-0.587/2))

    
    
    X, T = np.meshgrid(x, t)
    X = np.reshape(X, (-1, 1))
    T = np.reshape(T, (-1, 1))
    U1=np.reshape(u1,(-1,1))
    V1=np.reshape(v1,(-1,1))
    U2=np.reshape(u2,(-1,1))
    V2=np.reshape(v2,(-1,1))
    
    return np.hstack((X, T)), U1,V1,U2,V2


#kf = dde.Variable(0.05)
#D = dde.Variable(1.0)
m0=dde.Variable(1.0)
m1=dde.Variable(2.0)
m2=dde.Variable(-1.0)
m3=dde.Variable(0.0)

def pde(x,y):
    
    u1 = y[:, 0:1]
    v1 = y[:, 1:2]
    u2 = y[:, 2:3]
    v2 = y[:, 3:4]
    

    # 在'jacobian'中，i 是输出分量，j 是输入分量
    u1_t = dde.grad.jacobian(y, x, i=0, j=1)
    v1_t = dde.grad.jacobian(y, x, i=1, j=1)
    u2_t = dde.grad.jacobian(y, x, i=2, j=1)
    v2_t = dde.grad.jacobian(y, x, i=3, j=1)

    u1_x = dde.grad.jacobian(y, x, i=0, j=0)
    v1_x = dde.grad.jacobian(y, x, i=1, j=0)
    u2_x = dde.grad.jacobian(y, x, i=2, j=0)
    v2_x = dde.grad.jacobian(y, x, i=3, j=0)

    # 在“hessian”中，i 和 j 都是输入分量。 （Hessian 原则上可以是 d^2y/dxdt、d^2y/d^2x 等）
    # 输出组件由“组件”选择
    u1_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    v1_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    u2_xx = dde.grad.hessian(y, x, component=2, i=0, j=0)
    v2_xx = dde.grad.hessian(y, x, component=3, i=0, j=0)

    #f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
    #f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u
    
    #f_u1 = u1_t + m0*v1_xx + m1*(v1 ** 2 + u1 ** 2 + m2*(v2 ** 2 + u2 ** 2)) * v1 - m3*v1*(v2 ** 2 - u2 ** 2) - 2*m3*v2*u1*u2
    f_u1 = (
        u1_t + v1_xx  
        + 2*m0*(u1*(u1**2+v1**2)+2*u1*v1*v1)
        + 2*m1*(u1*(u2**2+v2**2)+2*u2*v1*v2)
        + 2*m2*(u1*u1*u2+2*u1*v1*v2+u2*v1*v1)
        + 2*m3*(u1*u1*u2-u2*v1*v1)
    )
    
    #f_v1 = v1_t - m0*u1_xx - m1*(v1 ** 2 + u1 ** 2 + m2*(v2 ** 2 + u2 ** 2)) * u1 - m3*u1*(v2 ** 2 - u2 ** 2) + 2*m3*v1*v2*u2
    f_v1 = (
        v1_t - u1_xx  
        + 2*m0*(v1*(u1**2+v1**2)+2*u1*u1*v1)
        + 2*m1*(v1*(u2**2+v2**2)+2*u1*u2*v2)
        + 2*m2*(v1*v1*v2-u1*u1*v2)
        + 2*m3*(v1*v1*v2+2*u1*u2*v1+u1*u1*v2)
    )
    
    #f_u2 = u2_t + m0*v2_xx + m1*(v2 ** 2 + u2 ** 2 + m2*(v1 ** 2 + u1 ** 2)) * v2 - m3*v2*(v1 ** 2 - u1 ** 2) - 2*m3*v1*u2*u1
    f_u2 = (
        u2_t + v2_xx  
        + 2*m0*(u2*(u1**2+v1**2)+2*u1*v1*v2)
        + 2*m1*(u2*(u2**2+v2**2)+2*u2*v2*v2)
        + 2*m2*(u1*u2*u2+2*u2*v1*v2+u1*v2*v2)
        + 2*m3*(u1*u2*u2-u1*v2*v2)
    )
    #f_v2 = v2_t - m0*u2_xx - m1*(v2 ** 2 + u2 ** 2 + m2*(v1 ** 2 + u1 ** 2)) * u2 - m3*u2*(v1 ** 2 - u1 ** 2) + 2*m3*v2*v1*u1
    f_v2 = (
        v2_t - u2_xx  
        + 2*m0*(v2*(u1**2+v1**2)+2*u1*u2*v1)
        + 2*m1*(v2*(u2**2+v2**2)+2*u2*u2*v2)
        + 2*m2*(v1*v2*v2-u2*u2*v1)
        + 2*m3*(v1*v2*v2+2*u1*u2*v2+u2*u2*v1)
    )

    return [f_u1, f_v1,f_u2,f_v2]


def fun_bc(x):
    return x[:, 0:1]


#def fun_init(x):
#    return np.exp(-20 * x[:, 0:1])

def init_cond_u1(x):
    "2 sech(x)"
    return (1/2) * np.cos(1.5*x[:, 0:1]-(-0.587/2)) * np.exp(2.5-1.5*x[:, 0:1]) / np.cosh(1.5*x[:, 0:1]+(-0.587/2) )
def init_cond_u2(x):
    "2 tanh(x)"
    return np.cos(1.5*x[:, 0:1]-(-0.587/2)) * np.exp(2.5-1.5*x[:, 0:1]) / np.cosh(1.5*x[:, 0:1]+(-0.587/2))

def init_cond_v1(x):
    return (1/2) * np.sin(1.5*x[:, 0:1]-(-0.587/2)) * np.exp(2.5-1.5*x[:, 0:1]) / np.cosh(1.5*x[:, 0:1]+(-0.587/2))

def init_cond_v2(x):
    return np.sin(1.5*x[:, 0:1]-(-0.587/2)) * np.exp(2.5-1.5*x[:, 0:1]) / np.cosh(1.5*x[:, 0:1]+(-0.587/2))


#geom = dde.geometry.Interval(0, 1)
#timedomain = dde.geometry.TimeDomain(0, 10)
#geomtime = dde.geometry.GeometryXTime(geom, timedomain)

x_lower = -10
x_upper = 10
t_lower = -2
t_upper = 2

# 创建 2D 域（用于绘图和输入）


# 整个域变平
#X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# 空间和时间域/几何（对于 deepxde 模型）
geom = dde.geometry.Interval(x_lower, x_upper)
time_domain = dde.geometry.TimeDomain(t_lower, t_upper)
geomtime = dde.geometry.GeometryXTime(geom, time_domain)




bc_u1_0 = dde.icbc.DirichletBC(
    geomtime,fun_bc, lambda _, on_boundary: on_boundary, component=0
)
bc_u1_1 = dde.icbc.DirichletBC(
    geomtime, fun_bc, lambda _, on_boundary: on_boundary,  component=0
)
bc_v1_0 = dde.icbc.DirichletBC(
    geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=1
)
bc_v1_1 = dde.icbc.DirichletBC(
    geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=1
)
bc_u2_0 = dde.icbc.DirichletBC(
    geomtime,fun_bc, lambda _, on_boundary: on_boundary,  component=2
)
bc_u2_1 = dde.icbc.DirichletBC(
    geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=2
)
bc_v2_0 = dde.icbc.DirichletBC(
    geomtime,fun_bc, lambda _, on_boundary: on_boundary,component=3
)
bc_v2_1 = dde.icbc.DirichletBC(
    geomtime, fun_bc, lambda _, on_boundary: on_boundary,  component=3
)




#ic1 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=0)
#ic2 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=1)

ic_u1 = dde.icbc.IC(geomtime, init_cond_u1, lambda _, on_initial: on_initial, component=0)
ic_v1 = dde.icbc.IC(geomtime, init_cond_v1, lambda _, on_initial: on_initial, component=1)
ic_u2 = dde.icbc.IC(geomtime, init_cond_u2, lambda _, on_initial: on_initial, component=2)
ic_v2 = dde.icbc.IC(geomtime, init_cond_v2, lambda _, on_initial: on_initial, component=3)

observe_x, u1,v1,u2,v2 = gen_traindata()
observe_y1 = dde.icbc.PointSetBC(observe_x, u1, component=0)
observe_y2 = dde.icbc.PointSetBC(observe_x, v1, component=1)
observe_y3 = dde.icbc.PointSetBC(observe_x, u2, component=2)
observe_y4 = dde.icbc.PointSetBC(observe_x, v2, component=3)

data = dde.data.TimePDE(
    geomtime,
    pde,
    #[bc_a, bc_b, ic1, ic2, observe_y1, observe_y2],
    [bc_u1_0, bc_u1_1, bc_v1_0, bc_v1_1,bc_u2_0, bc_u2_1, bc_v2_0, bc_v2_1, ic_u1, ic_v1,ic_u2, ic_v2,observe_y1, observe_y2,observe_y3,observe_y4],
    num_domain=200,
    num_boundary=100,
    num_initial=100,
    anchors=observe_x,
    num_test=50000,
)
net = dde.nn.FNN([2] + [20] * 3 + [4], "sin", "Glorot uniform")



model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[m0,m1,m2,m3])
variable = dde.callbacks.VariableValue([m0,m1,m2,m3], period=50, filename="variables.dat")
losshistory, train_state = model.train(iterations=800, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)