import taichi as ti
import simulator

Lx = simulator.Lx        
Ly = simulator.Ly

# rendering kernels - no archiving required
render_curls = ti.Vector.field(3, float, shape=(Lx, Ly))
render_vels  = ti.Vector.field(3, float, shape=(Lx, Ly))
render_pres  = ti.Vector.field(3, float, shape=(Lx, Ly))
render_dens  = ti.Vector.field(3, float, shape=(Lx, Ly))

@ti.func
def get_rgb(x, M, m)->ti.Vector([ti.f64, ti.f64 ,ti.f64]):
    l = 5 / (M-m)
    return ti.Vector([
        ti.math.clamp(l * (x - (M*2+m)/3) + 1, 0, 1),
        ti.math.clamp(2 - l * ti.abs(x - (M+m)/2), 0, 1),
        ti.math.clamp(-l * (x - (M+2*m)/3) + 1, 0, 1)
    ])

@ti.kernel
def rendering_curl(curl: ti.template(), rend_curl: ti.template()):
    for i, j in rend_curl:
        p = ti.Vector([i, j])
        t = simulator.bilerp_c(curl, p)
        rend_curl[i, j] = get_rgb(t, 1.5, -1.5)

@ti.kernel
def rendering_u(u0: ti.template(), rend_vel: ti.template()):
    # M, m = 1.33e2, 0.6e0
    M, m = 5.00, 0.0e0
        
    for i, j in rend_vel:
        p = ti.Vector([i, j])
        t = simulator.bilerp_u(u0, p)
        rend_vel[i, j] = get_rgb(t, M, m)

@ti.kernel
def rendering_v(v0: ti.template(), rend_vel: ti.template()):
    M, m = 6.5e1, -6.5e1
        
    for i, j in rend_vel:
        p = ti.Vector([i, j])
        t = simulator.bilerp_u(v0, p)
        rend_vel[i, j] = get_rgb(t, M, m)

@ti.kernel
def rendering_pre(pre: ti.template(), rend_pre: ti.template()):
    for i, j in rend_pre:
        p = ti.Vector([i, j])
        pv = simulator.bilerp_c(pre, p)
        rend_pre[i, j] = get_rgb(pv, 0.1, -0.1)


@ti.kernel
def rendering_dens(den: ti.template(), rend_den: ti.template()):
    M, m = 1.0, 0.0
    
    for i, j in rend_den:
        p = ti.Vector([i, j])
        pd = (simulator.bilerp_c(den, p) - m) / (M - m)
        rend_den[i, j] = ti.Vector([pd, pd, pd])


def reset():
    simulator.u0.fill(0)  
    simulator.u1.fill(0)
    simulator.v0.fill(0)
    simulator.v1.fill(0)
    simulator.p0.fill(0)
    simulator.p1.fill(0)
    simulator.d0.fill(0)
    simulator.d1.fill(0)
    simulator.divs.fill(0)
    simulator.curls.fill(0)

def main():
    visualize_c = False      #visualize curl
    visualize_u = False    #visualize velocity x (u)
    visualize_v = False    #visualize velocity y (v)
    visualize_p = False     #visualize pressure
    visualize_d = True     # visualize density
    paused = False

    gui = ti.GUI('Stable Fluid', (Lx, Ly))
        
    Frame = 0
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == 'r':
                paused = True
                reset()
            elif e.key == 'x':
                visualize_c = False
                visualize_u = True
                visualize_v = False
                visualize_p = False
                visualize_d = False
            elif e.key == 'y':
                visualize_c = False
                visualize_u = False
                visualize_v = True
                visualize_p = False
                visualize_d = False
            elif e.key == 'c':
                visualize_c = True
                visualize_u = False
                visualize_v = False
                visualize_p = False
                visualize_d = False
            elif e.key == 'p':
                visualize_c = False
                visualize_u = False
                visualize_v = False
                visualize_p = True
                visualize_d = False
            elif e.key == 'd':
                visualize_c = False
                visualize_u = False
                visualize_v = False
                visualize_p = False
                visualize_d = True
            elif e.key == ti.GUI.SPACE:
                paused = not paused
                
        Frame += 1

        if not paused:
            simulator.step()
        if visualize_c:
            simulator.vorticity(simulator.u0, simulator.v0, simulator.curls, simulator.cell_type)
            rendering_curl(simulator.curls, render_curls)
            gui.set_image(render_curls)
        elif visualize_u:
            rendering_u(simulator.u0, render_vels)
            gui.set_image(render_vels)
        elif visualize_v:
            rendering_v(simulator.v0, render_vels)
            gui.set_image(render_vels)
        elif visualize_p:
            rendering_pre(simulator.p0, render_pres)
            gui.set_image(render_pres)
        elif visualize_d:
            rendering_dens(simulator.d0, render_dens)
            gui.set_image(render_dens)

        gui.show()

if __name__ == '__main__':
    main()