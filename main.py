import numpy as np
import time
import mujoco
import mujoco.viewer
from fancy_plots import fancy_plots_2, fancy_plots_1
import matplotlib.pyplot as plt

def get_system_states(system):
    # System values angular displacement
    q1 = system.qpos[0]
    q2 = system.qpos[1]

    # System values angular velocities
    q1p = system.qvel[0]
    q2p = system.qvel[1]

    x = np.array([q1, q2, q1p, q2p], dtype=np.double)
    return x
def get_system_states_sensor(system):
    q0 = system.sensor("q_0").data.copy()
    q1 = system.sensor("q_1").data.copy()
    q0p = system.sensor("q_0p").data.copy()
    q1p = system.sensor("q_1p").data.copy()
    x = np.array([q0[0], q1[0], q0p[0], q1p[0]], dtype=np.double)
    return x

def get_system_states_box(system):
    x = system.sensor("box_pos").data.copy()
    print(x)
    return None
def get_system_energy(system):
    e1 = system.energy[0]
    e2 = system.energy[1]
    et = e1 + e1
    return et

def control_action(system, u):
    u1 = u[0]
    u2 = u[1]
    system.ctrl[0] = u1
    system.ctrl[1] = u2
    return None
def set_torque_control(system):
    # q_0
    system.actuator_gainprm[0, 0] = 1
    # q_1
    system.actuator_gainprm[1, 0] = 1
    return None
def main():
    # Load Model form XML file
    m = mujoco.MjModel.from_xml_path('prueba_1.xml')
    # Print color of the box and position of the red box

    # Get information form the xml
    data = mujoco.MjData(m)

    # Simulation time parameters
    ts = 0.01
    tf = 60
    t = np.arange(0, tf+ts, ts, dtype=np.double)

    # States System
    q = np.zeros((4, t.shape[0]+1), dtype=np.double)
    q_n = np.zeros((4, t.shape[0]+1), dtype=np.double)

    # Control signals
    u = np.zeros((2, t.shape[0]), dtype=np.double)
    u[0, :] = 20*np.pi/180*np.cos(t)
    u[1, :] = 40*np.pi/180*np.cos(t)

    # States Energy system
    E = np.zeros((1, t.shape[0]+1), dtype=np.double)

    # Define Paramerters for the software
    m.opt.timestep = ts

    # Reset Properties system
    mujoco.mj_resetDataKeyframe(m, data, 0)  # Reset the state to keyframe 0

    # Initial conditions system
    data.qpos[0] = 120*np.pi/180
    data.qpos[1] = -90*np.pi/180
    set_torque_control(m)


    with mujoco.viewer.launch_passive(m, data) as viewer:
        if viewer.is_running():
            # Initial data System
            q[:, 0] = get_system_states(data)
            # Initial Energy System
            E[:, 0] = get_system_energy(data)
            q_n[:, 0] = get_system_states_sensor(data)

            # Simulation of the system
            for k in range(0, t.shape[0]):
                tic = time.time()
                control_action(data, u[:,k])

                # System evolution
                mujoco.mj_step(m, data)

                # System evolution visualization
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                # Get system states
                get_system_states_box(data)
                q[:, k+1] = get_system_states(data)
                E[: k+1] = get_system_energy(data)
                q_n[:, k+1] = get_system_states_sensor(data)


                # Section to guarantee same sample times
                while (time.time() - tic <= m.opt.timestep):
                    None
                toc = time.time() - tic
        fig1, ax1, ax2 = fancy_plots_2()
        ## Axis definition necesary to fancy plots
        ax1.set_xlim((t[0], t[-1]))
        ax2.set_xlim((t[0], t[-1]))
        ax1.set_xticklabels([])

        state_1, = ax1.plot(t,q[0,0:t.shape[0]],
                    color='#00429d', lw=2, ls="-")
        state_3, = ax1.plot(t,q[2,0:t.shape[0]],
                    color='#9e4941', lw=2, ls="-.")

        state_2, = ax2.plot(t,q[1,0:t.shape[0]],
                    color='#ac7518', lw=2, ls="-")
        state_4, = ax2.plot(t,q[3,0:t.shape[0]],
                    color='#97a800', lw=2, ls="-.")

        ax1.set_ylabel(r"$[rad],[rad/s]$", rotation='vertical')
        ax1.legend([state_1,state_3],
            [r'$x_1$', r'$x_3$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
        ax1.grid(color='#949494', linestyle='-.', linewidth=0.5)

        ax2.set_ylabel(r"$[rad],[rad/s]$", rotation='vertical')
        ax2.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
        ax2.legend([state_2, state_4],
            [r'$x_2$', r'$x_4$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
        ax2.grid(color='#949494', linestyle='-.', linewidth=0.5)
        fig1.savefig("system_states.eps")
        fig1.savefig("system_states.png")
        fig1
        plt.show()

        fig2, ax12, ax22 = fancy_plots_2()
        ## Axis definition necesary to fancy plots
        ax12.set_xlim((t[0], t[-1]))
        ax22.set_xlim((t[0], t[-1]))
        ax12.set_xticklabels([])

        state_12, = ax12.plot(t,q_n[0,0:t.shape[0]],
                    color='#00429d', lw=2, ls="-")
        state_32, = ax12.plot(t,q_n[2,0:t.shape[0]],
                    color='#9e4941', lw=2, ls="-.")

        state_22, = ax22.plot(t,q_n[1,0:t.shape[0]],
                    color='#ac7518', lw=2, ls="-")
        state_42, = ax22.plot(t,q_n[3,0:t.shape[0]],
                    color='#97a800', lw=2, ls="-.")

        ax12.set_ylabel(r"$[rad],[rad/s]$", rotation='vertical')
        ax12.legend([state_12,state_32],
            [r'$x_1$', r'$x_3$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
        ax12.grid(color='#949494', linestyle='-.', linewidth=0.5)

        ax22.set_ylabel(r"$[rad],[rad/s]$", rotation='vertical')
        ax22.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
        ax22.legend([state_22, state_42],
            [r'$x_2$', r'$x_4$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
        ax22.grid(color='#949494', linestyle='-.', linewidth=0.5)
        fig2.savefig("system_states_noise.eps")
        fig2.savefig("system_states_noise.png")
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