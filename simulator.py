import taichi as ti
from solver import ConjugateGradientSolver as solver



# SCENE = "SQR"
# SCENE = "CIRCLES"
SCENE = "SMOKE_PLUME"

num_solver_iter = 100
# cell sides' length
dx = 1
dy = 1
dx_inv = 1 / dx
dy_inv = 1 / dy


Lx = solver.Lx  # Lx
Ly = solver.Ly  # Ly
nx = solver.nx
ny = solver.ny

dt = 0.02
rho = 1.0
rho_inv = 1 / rho
damping = 1


u0 = ti.field(float, shape=(nx + 1, ny))
u1 = ti.field(float, shape=(nx + 1, ny))
v0 = ti.field(float, shape=(nx, ny + 1))
v1 = ti.field(float, shape=(nx, ny + 1))
p0 = ti.field(float, shape=(nx, ny))
p1 = ti.field(float, shape=(nx, ny))
d0 = ti.field(float, shape=(nx, ny))
d1 = ti.field(float, shape=(nx, ny))
divs = ti.field(float, shape=(nx, ny))
curls = ti.field(float, shape=(nx, ny))

# cell type
cell_type = ti.field(int, shape=(nx, ny))
EMPTY = -1
SOLID = 0  # neumann boundary
FLUID = 1

# circle
Ox = nx / 4
Oy = ny / 2
radius = Ly / 20

# square
# xmin = int(nx * 9.5 // 40)
# xmax = int(nx * 10.5 // 40)
# ymin = int(ny * 9.5 // 20)
# ymax = int(ny * 10.5 // 20)
xmin = int(nx * 8.5 // 40)
xmax = int(nx * 11.5 // 40)
ymin = int(ny * 8.5 // 20)
ymax = int(ny * 11.5 // 20)

wxmin = int(Lx * xmin / nx)
wxmax = int(Lx * xmax / nx)
wymin = int(Ly * ymin / ny)
wymax = int(Ly * ymax / ny)

@ti.kernel
def init_boundary(ct: ti.template()):
    for i, j in ct:
        if SCENE == "SQR" or SCENE == "CIRCLES" :
            cond = False

            if SCENE == "SQR":
                cond = i >= xmin and i < xmax and j >= ymin and j < ymax
            elif SCENE == "CIRCLES":
                circle1 = ((i - Ox) ** 2 + (j - Oy) ** 2 <= radius ** 2)
                circles2 = ((i - 1.5*Ox) ** 2 + (j - 0.8*Oy) ** 2 <= radius ** 2) or ((i - 1.5*Ox) ** 2 + (j - 1.2*Oy) ** 2 <= radius ** 2)
                circles3 = ((i - 2.0*Ox) ** 2 + (j - 0.5*Oy) ** 2 <= radius ** 2) or ((i - 2.0*Ox) ** 2 + (j - 1.0*Oy) ** 2 <= radius ** 2) or ((i - 2.0*Ox) ** 2 + (j - 1.5*Oy) ** 2 <= radius ** 2)
                cond = circle1 or circles2 or circles3

            if cond:
                ct[i, j] = SOLID
            elif i == nx - 1:
                ct[i, j] = EMPTY
            elif i == 0 or j == 0 or j == ny - 1:
                ct[i, j] = SOLID
            else:
                ct[i, j] = FLUID

        elif SCENE == "SMOKE_PLUME" :
            if i == 0:
                ct[i, j] = SOLID
            elif j == 0 or j == ny - 1:
                ct[i, j] = SOLID
            elif i == nx - 1:
                ct[i, j] = EMPTY
            else:
                ct[i, j] = FLUID


# solver
init_boundary(cell_type)

solver.set_solver_param(num_solver_iter, 1e-5)
solver.build_LinearSystem_LHS_Poisson(cell_type, dt / (rho * dx * dx))


### ADVECTION ###

@ti.func
def usample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I[0] = ti.math.clamp(I[0], 0, nx)
    I[1] = ti.math.clamp(I[1], 0, ny - 1)
    return qf[I]


@ti.func
def vsample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I[0] = ti.math.clamp(I[0], 0, nx - 1)
    I[1] = ti.math.clamp(I[1], 0, ny)
    return qf[I]


@ti.func
def qsample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I[0] = ti.math.clamp(I[0], 0, nx - 1)
    I[1] = ti.math.clamp(I[1], 0, ny - 1)
    return qf[I]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp_u(u0, p):
    u, v = p
    s, t = u * dx_inv, v * dy_inv - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = usample(u0, iu, iv)
    b = usample(u0, iu + 1, iv)
    c = usample(u0, iu, iv + 1)
    d = usample(u0, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


@ti.func
def bilerp_v(v0, p):
    u, v = p
    s, t = u * dx_inv - 0.5, v * dy_inv
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = vsample(v0, iu, iv)
    b = vsample(v0, iu + 1, iv)
    c = vsample(v0, iu, iv + 1)
    d = vsample(v0, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

@ti.func
def bilerp_c(vf, p):
    u, v = p
    s, t = u * dx_inv - 0.5, v * dy_inv - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = qsample(vf, iu, iv)
    b = qsample(vf, iu + 1, iv)
    c = qsample(vf, iu, iv + 1)
    d = qsample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


@ti.func
def backtrace(u0: ti.template(), v0: ti.template(), p, dt_: ti.template()):
    # 3rd order Runge-Kutta
    u1 = bilerp_u(u0, p)
    v1 = bilerp_v(v0, p)
    p1 = p - 0.5 * dt_ * ti.Vector([u1, v1])
    u2 = bilerp_u(u0, p1)
    v2 = bilerp_v(v0, p1)
    p2 = p - 0.75 * dt_ * ti.Vector([u2, v2])
    u3 = bilerp_u(u0, p2)
    v3 = bilerp_v(v0, p2)
    p -= dt_ * (ti.Vector([(2 / 9) * u1 + (1 / 3) * u2 + (4 / 9) * u3, (2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3]))
    return p

    # Basic Advection Method
    # v1 = bilerp_u(u0, p)
    # return p - v1 * dt

    # advection-reflection
    # v1 = bilerp(vf, p)
    # p1 = p - 0.5 * dt * v1
    # v2 = bilerp(vf, p1)
    # p2 = p1 - 0.5 * dt * v2
    # return p2


@ti.kernel
def advect_u(u0: ti.template(), v0: ti.template(), cur_u: ti.template(), nxt_u: ti.template()):
    for i, j in cur_u:
        if i == 0 or i == nx: continue
        p = ti.Vector([i * dx, (j + 0.5) * dy])
        p = backtrace(u0, v0, p, dt)
        nxt_u[i, j] = bilerp_u(u0, p)


@ti.kernel
def advect_v(u0: ti.template(), v0: ti.template(), cur_v: ti.template(), nxt_v: ti.template()):
    for i, j in cur_v:
        if j == 0 or j == ny: continue
        p = ti.Vector([(i + 0.5) * dx, j * dy])
        p = backtrace(u0, v0, p, dt)
        nxt_v[i, j] = bilerp_v(v0, p) * damping


@ti.kernel
def advect_c(u0: ti.template(), v0: ti.template(), cur_f: ti.template(), nxt_f: ti.template()):
    for i, j in cur_f:
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1: continue
        p = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dy])
        p = backtrace(u0, v0, p, dt)
        nxt_f[i, j] = bilerp_c(cur_f, p) * damping


@ti.kernel
def apply_den(df: ti.template()):
    den = 1
    for i, j in df:
        if SCENE == "CIRCLES" or SCENE == "SQR" :
            if j >= int(ny * 8.6 // 20) and j <= int(ny * 8.7 // 20): df[0, j] = den
            if j >= int(ny * 9.5 // 20) and j <= int(ny * 9.6 // 20): df[0, j] = den
            if j >= int(ny * 10.4 // 20) and j <= int(ny * 10.5 // 20): df[0, j] = den
            if j >= int(ny * 11.3 // 20) and j <= int(ny * 11.4 // 20): df[0, j] = den

        if SCENE == "CIRCLES" :
            if j >= int(ny * 5.9 // 20) and j <= int(ny * 6.0 // 20): df[0, j] = den
            if j >= int(ny * 6.8 // 20) and j <= int(ny * 6.9 // 20): df[0, j] = den
            if j >= int(ny * 7.7 // 20) and j <= int(ny * 7.8 // 20): df[0, j] = den
            if j >= int(ny * 12.2 // 20) and j <= int(ny * 12.3 // 20): df[0, j] = den
            if j >= int(ny * 13.1 // 20) and j <= int(ny * 13.2 // 20): df[0, j] = den
            if j >= int(ny * 14.0 // 20) and j <= int(ny * 14.1 // 20): df[0, j] = den

        if SCENE == "SMOKE_PLUME" :
            if j >= int(ny * 7.8 // 20) and j <= int(ny * 8.4 // 20): df[0, j] = den
            if j >= int(ny * 8.9 // 20) and j <= int(ny * 9.3 // 20): df[0, j] = den
            if j >= int(ny * 9.8 // 20) and j <= int(ny * 10.2 // 20): df[0, j] = den
            if j >= int(ny * 10.7 // 20) and j <= int(ny * 11.1 // 20): df[0, j] = den
            if j >= int(ny * 11.6 // 20) and j <= int(ny * 12.2 // 20): df[0, j] = den


@ti.kernel
def enforce_boundary(u0: ti.template(), v0: ti.template(), ct: ti.template(), pf: ti.template()):
    for i, j in ct:
        if ct[i, j] == SOLID:
            u0[i, j] = u0[i + 1, j] = v0[i, j] = v0[i, j + 1] = 0.0

        if ct[i, j] == EMPTY:
            pf[i, j] = 0.0


    for j in ti.ndrange(ny):
        u0[0, j] = u0[1, j] = Lx / 10.0
        u0[nx, j] = u0[nx - 1, j] = u0[nx - 2, j]
        v0[0, j] = v0[1, j] = v0[nx - 1, j] = 0.0


    # for j in ti.ndrange(ny):
    #     u0[0, j] = u0[1, j] = Lx / 10.0
    #     v0[0, j] = v0[nx - 1, j] = 0.0
    #     u0[nx, j] = u0[nx-1, j] #= u0[nx-2,j]



    # for i in ti.ndrange(nx):
    #     v0[i,1] =  v0[i,ny-1]= 0.0

@ti.kernel
def enforce_boundary_plume(u0: ti.template(), v0: ti.template(), ct: ti.template(), pf: ti.template()):
    for i, j in ct:
        if ct[i, j] == SOLID:
            u0[i, j] = u0[i + 1, j] = v0[i, j] = v0[i, j + 1] = 0.0

        if ct[i, j] == EMPTY:
            pf[i, j] = 0.0


    # for j in ti.ndrange(ny):
    #     if (j < 0.75 * ny and j > 0.25 * ny):
    #         u0[0, j] = u0[1, j] = Lx / 5.0
    #         v0[0, j] = v0[1, j] = 0.0
    #     else :
    #         u0[0, j] = u0[1, j] = u0[2, j]
    for j in ti.ndrange(ny):
        if (j < 0.65 * ny and j > 0.35 * ny):
            u0[0, j] = u0[1, j] = Lx / 10.0
            v0[0, j] = v0[1, j] = 0.0

        u0[nx, j] = u0[nx - 1, j] = u0[nx - 2, j]


    # for j in ti.ndrange(ny):
    #     u0[0, j] = u0[1, j] = Lx / 10.0
    #     v0[0, j] = v0[nx - 1, j] = 0.0
    #     u0[nx, j] = u0[nx-1, j] #= u0[nx-2,j]



    # for i in ti.ndrange(nx):
    #     v0[i,1] =  v0[i,ny-1]= 0.0

@ti.kernel
def divergence(u0: ti.template(), v0: ti.template(), divs: ti.template(), ct: ti.template()):
    for i, j in divs:
        if ct[i, j] == SOLID or ct[i, j] == EMPTY: continue

        vl = usample(u0, i, j)
        vr = usample(u0, i + 1, j)
        vb = vsample(v0, i, j)
        vt = vsample(v0, i, j + 1)

        # # Open boundary
        # if i == 0: vl = -vr
        if i == nx - 2: vr = vl

        # # Closed boundary
        # if j == 0: vb = -vt
        # if j == ny - 1: vt = -vb

        divs[i, j] = (vr - vl) * dx_inv + (vt - vb) * dy_inv


@ti.kernel
def vorticity(u0: ti.template(), v0: ti.template(), curls: ti.template(), ct: ti.template()):
    for i, j in curls:
        if ct[i, j] == SOLID or ct[i, j] == EMPTY: continue

        vl = bilerp_v(v0, ti.Vector([(i + 0.0) * dx, (j + 0.5) * dy]))
        vr = bilerp_v(v0, ti.Vector([(i + 1.0) * dx, (j + 0.5) * dy]))
        vb = bilerp_u(u0, ti.Vector([(i + 0.5) * dx, (j + 0.0) * dy]))
        vt = bilerp_u(u0, ti.Vector([(i + 0.5) * dx, (j + 1.0) * dy]))

        # vl = bilerp_v(v0, ti.Vector([i * dx, j * dy]))
        # vr = bilerp_v(v0, ti.Vector([(i + 1) * dx, j * dy]))
        # vb = bilerp_u(u0, ti.Vector([i * dx, j * dy]))
        # vt = bilerp_u(u0, ti.Vector([i * dx, (j + 1) * dy]))

        curls[i, j] = (vr - vl) * dx_inv - (vt - vb) * dy_inv


@ti.kernel
def pressure_jacobi(pf: ti.template(), nxt_pf: ti.template(), ct: ti.template()):
    for i, j in pf:
        if ct[i, j] == SOLID or ct[i, j] == EMPTY: continue

        pl = qsample(pf, i - 1, j)
        pr = qsample(pf, i + 1, j)
        pb = qsample(pf, i, j - 1)
        pt = qsample(pf, i, j + 1)
        div = divs[i, j]
        h2dij = rho * dx * dy * div / dt

        # Open boundary (out-flow)
        # if i == width - 1: pr = 0

        # if i == 0 : pl = 0
        # if i == nx - 1: pr = 0

        num_neigh_fluid_cell = ti.abs(ct[i - 1, j]) + ti.abs(ct[i + 1, j]) + ti.abs(ct[i, j - 1]) + ti.abs(ct[i, j + 1])

        if num_neigh_fluid_cell > 0:
            nxt_pf[i, j] = (ct[i - 1, j] * pl + ct[i + 1, j] * pr + ct[i, j - 1] * pb + ct[
                i, j + 1] * pt - h2dij) / num_neigh_fluid_cell

        # nxt_pf[i, j] = 0.25 * (pl + pr + pb + pt - h2dij)


@ti.kernel
def subtract_gradient_u(u0: ti.template(), pf: ti.template(), ct: ti.template()):
    for i, j in u0:
        if i < nx:
            if ct[i, j] == SOLID or ct[i, j] == EMPTY: continue

        pl = qsample(pf, i - 1, j)
        pc = qsample(pf, i, j)

        u0[i, j] -= dt * (pc - pl) * dx_inv * rho_inv

        # pl = qsample(pf, i - 1, j)
        # pr = qsample(pf, i + 1, j)

        # u0[i, j] -= dt * (pr - pl) * 0.5 * dx_inv


@ti.kernel
def subtract_gradient_v(v0: ti.template(), pf: ti.template(), ct: ti.template()):
    for i, j in v0:
        if j < ny:
            if ct[i, j] == SOLID or ct[i, j] == EMPTY: continue

        pb = qsample(pf, i, j - 1)
        pc = qsample(pf, i, j)

        v0[i, j] -= dt * (pc - pb) * dy_inv * rho_inv

        # pb = qsample(pf, i, j - 1)
        # pt = qsample(pf, i, j)

        # v0[i, j] -= dt * (pt - pb) * 0.5 * dy_inv


def step():
    global u0, u1, v0, v1, p0, p1, d0, d1, divs, curls
    global solver
    global cell_type

    # apply u
    apply_den(d0)
    if SCENE == "SQR" or SCENE == "CIRCLES":
        enforce_boundary(u0, v0, cell_type, p0)

    elif SCENE == "SMOKE_PLUME":
        enforce_boundary_plume(u0, v0, cell_type, p0)

    # advection
    advect_u(u0, v0, u0, u1)
    advect_v(u0, v0, v0, v1)
    advect_c(u0, v0, d0, d1)
    u0, u1, v0, v1 = u1, u0, v1, v0
    d0, d1 = d1, d0

    p0.fill(0.0)
    p1.fill(0.0)
    divergence(u0, v0, divs, cell_type)

    solver.build_LinearSystem_RHS_Poisson(divs, cell_type)
    solver.solveCG(cell_type)
    solver.copy_Solution(p0)

    # projection
    subtract_gradient_u(u0, p0, cell_type)
    subtract_gradient_v(v0, p0, cell_type)
