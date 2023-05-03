import numpy as np
import time
import mujoco
import mujoco.viewer
from fancy_plots import fancy_plots_2, fancy_plots_1
import matplotlib.pyplot as plt

def get_system_states(system):
    # System values angular displacement
    q1 = system.qpos[0:3]

    # System values angular velocities
    q1p = system.qvel[0:3]

    x = np.array([q1, q1p], dtype=np.double)
    x_aux = x.reshape((6, ))
    return x_aux

def get_system_energy(system):
    e1 = system.energy[0]
    et = e1 
    return et

def control_action(system, u):
    u1 = u[0]
    u2 = u[1]
    system.qvel[0] = u1
    system.qvel[1] = u2
    return None
def get_system_force(model, data):
    # Section Read Force and torques of the system
    forcetorque_general = np.zeros((6,), dtype=np.double)
    force = np.zeros((3,), dtype=np.double)
    for j,c in enumerate(data.contact):
        mujoco.mj_contactForce(model, data, j, forcetorque_general)
        force = force + forcetorque_general[0:3]
    norm_force = np.linalg.norm(force)
    return norm_force

def main():
    # Load Model form XML file
    m = mujoco.MjModel.from_xml_path('force.xml')
    # Print color of the box and position of the red box

    # Get information form the xml
    data = mujoco.MjData(m)

    # Simulation time parameters
    ts = 0.001
    tf = 60
    t = np.arange(0, tf+ts, ts, dtype=np.double)
    q = np.zeros((6, t.shape[0] + 1), dtype=np.double)
    forcetorque_general = np.zeros((6,), dtype=np.double)
    force = np.zeros((1, t.shape[0] + 1), dtype=np.double)

    # Control signals
    u = np.zeros((3, t.shape[0]), dtype=np.double)
    u[0, :] = 0*np.pi/180*np.cos(t)
    u[1, :] = 0*np.pi/180*np.cos(t)
    u[2, :] = 0*np.pi/180*np.cos(t)

    # Define Paramerters for the software
    m.opt.timestep = ts

    # Reset Properties system
    mujoco.mj_resetDataKeyframe(m, data, 0)  # Reset the state to keyframe 0

    with mujoco.viewer.launch_passive(m, data) as viewer:
        if viewer.is_running():
            # initial positions read
            data.qpos[0:3] = np.array([0, 0, 0.1], dtype = np.double)
            q[:, 0] = get_system_states(data)
            #force[:, 0] = get_system_force(m, data)
            for k in range(0, t.shape[0]):
                tic = time.time()
                #control_action(data, u[:,k])
                print(force[:, k])

                # System evolution
                mujoco.mj_step(m, data)

                # System evolution visualization
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                # Get system states
                q[:, k+1] = get_system_states(data)
                force[:, k+1] = get_system_force(m, data)

                # Section to guarantee same sample times
                while (time.time() - tic <= m.opt.timestep):
                    None
                toc = time.time() - tic
        fig2, ax11 = fancy_plots_1()
        ## Axis definition necesary to fancy plots
        ax11.set_xlim((t[0], t[-1]))

        force_plt, = ax11.plot(t,force[0,0:t.shape[0]],
                    color='#00429d', lw=2, ls="-")

        ax11.set_ylabel(r"$[Joule]$", rotation='vertical')
        ax11.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
        ax11.legend([force_plt],
            [r'$E$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
        ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

        fig2.savefig("forces.eps")
        fig2.savefig("forces.png")
        fig2
        plt.show()

if __name__ == '__main__':
    try:
        main()
    except(KeyboardInterrupt):
        print("Error System")
        pass
    else:
        print("Complete Execution")
        pass