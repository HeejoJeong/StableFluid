import taichi as ti

# window resolution
Lx = 800        # Lx
Ly = 400        # Ly

# cell sides' length
dx = 1
dy = 1
dx_inv = 1 / dx
dy_inv = 1 / dy

# grid resolution
nx = int(Lx // dx) + 2
ny = int(Ly // dy) + 2

dt = 0.02
rho = 1.0
rho_inv = 1 / rho

EMPTY = -1
SOLID = 0
FLUID = 1

_A_elements = ti.Vector.field(5,dtype=ti.f32,shape = (nx,ny))      #0: diag, 1 : x+1, 2: y+1, 3:x-1, 4: y-1
_b = ti.field(dtype=ti.f32, shape=(nx, ny))

_iter_var = ti.Vector.field(4,dtype=ti.f32,shape=(nx,ny)) # 0: x solution, 1 : residual, 2 : search vector, 3 : A * (search vector)
_old_rTr = ti.field(dtype=ti.f32, shape=())

max_iter = 0
tol = 0

def set_solver_param(amax_iter=5000, atol = 1e-5):
    global max_iter, tol
    max_iter = amax_iter
    tol = atol
def build_LinearSystem_LHS_Poisson(cell_type, scalerA):
    global _A_elements
    __build_LinearSystem_LHS_Poisson(_A_elements,cell_type,scalerA)
@ti.kernel
def __build_LinearSystem_LHS_Poisson(A_elements : ti.template(),cell_type:ti.template(),scalerA:ti.f32) :
    A_elements.fill(0.0)

    for i,j in cell_type :
        if cell_type[i,j] == FLUID :
            if cell_type[i +1 , j] == FLUID:
                A_elements[i,j][0] +=scalerA
                A_elements[i,j][1] = -scalerA
            elif cell_type[i + 1, j] == EMPTY:
                A_elements[i,j][0] +=scalerA

            if cell_type[i - 1, j] == FLUID:
                A_elements[i,j][0] +=scalerA
                A_elements[i,j][3] = -scalerA
            elif cell_type[i - 1, j] == EMPTY:
                A_elements[i,j][0] +=scalerA

            if cell_type[i, j + 1] == FLUID:
                A_elements[i,j][0] +=scalerA
                A_elements[i,j][2] = -scalerA
            elif cell_type[i, j + 1] == EMPTY:
                A_elements[i,j][0] +=scalerA

            if cell_type[i , j-1] == FLUID:
                A_elements[i,j][0] +=scalerA
                A_elements[i,j][4] = -scalerA
            elif cell_type[i , j-1] == EMPTY:
                A_elements[i,j][0] +=scalerA



def build_LinearSystem_RHS_Poisson(div, cell_type):
    global _b
    __build_LinearSystem_RHS_Poisson(b=_b,div=div,cell_type=cell_type)

@ti.kernel
def __build_LinearSystem_RHS_Poisson(b:ti.template(),div : ti.template(), cell_type : ti.template()):
    b.fill(0.0)
    for i,j in cell_type:
        if cell_type[i,j] == FLUID :
            b[i,j] = -1.0*div[i,j]

def solveCG(cell_type) :
    global _iter_var,_old_rTr,tol,_A_elements,_b

    __CG_iter_init(iter_var = _iter_var,b=_b,oldrTr=_old_rTr)

    iter = 0
    while iter<max_iter :
        __CG_iterate(iter_var=_iter_var, A_elements=_A_elements, cell_type=cell_type, oldrTr=_old_rTr)
        iter+=1
        # if iter % 30 == 0 :
        #     if _old_rTr[None] <= tol:
        #         break;

    # print("iter end : {}, rTr = {}".format(iter,_old_rTr[None]))



@ti.kernel
def __CG_iter_init(iter_var:ti.template(),b:ti.template(),oldrTr: ti.template()):

    oldrTr[None] = 0.0
    for i,j in b :
        iter_var[i,j][1] = b[i,j]   # residual = b
        iter_var[i,j][2] = iter_var[i,j][1] # search vector = residual
        iter_var[i,j][0] = 0.0 # solution =0
        iter_var[i,j][3] = 0.0 # A*search vector = 0
        oldrTr[None] += iter_var[i,j][1] * iter_var[i,j][1]


@ti.kernel
def __CG_iterate(iter_var : ti.template(),A_elements:ti.template(),cell_type : ti.template(),oldrTr : ti.template()) :
    pTAp = 0.0
    for i, j in cell_type:
        if cell_type[i, j] == FLUID:
            iter_var[i,j][3] = A_elements[i, j][0] * iter_var[i,j][2] + A_elements[i, j][3] * iter_var[i - 1, j][2] +\
                               A_elements[i, j][1] * iter_var[i +1 , j][2] + A_elements[i, j][4] * iter_var[i, j-1][2] + A_elements[i, j][2] * iter_var[i, j+1][2]
            pTAp +=iter_var[i,j][2] * iter_var[i,j][3]
    alpha = oldrTr[None] / pTAp

    newrTr = 0.0
    for i, j in cell_type:
        if cell_type[i, j] == FLUID:
            iter_var[i,j][0] += alpha * iter_var[i,j][2]
            iter_var[i,j][1] -= alpha * iter_var[i,j][3]
            newrTr += iter_var[i,j][1] * iter_var[i,j][1]

    beta = newrTr / oldrTr[None]
    oldrTr[None] = newrTr
    for i, j in cell_type:
        if cell_type[i, j] == FLUID:
            iter_var[i,j][2] *= beta
            iter_var[i,j][2] += iter_var[i,j][1]



def copy_Solution(dst) :
    global _iter_var
    _copy_sol(dst,_iter_var)

@ti.kernel
def _copy_sol(dst:ti.template(), iter_var:ti.template()):
    for i,j in dst :
        dst[i,j] =iter_var[i,j][0]



