import numpy as np
import time
import mujoco
import mujoco.viewer
from fancy_plots import fancy_plots_2, fancy_plots_1
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def control_action(system, u):
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    u4 = u[3]
    
    system.ctrl[0] = u1
    system.ctrl[1] = u2
    system.ctrl[2] = u3
    system.ctrl[3] = u4
    return None

def controller_z(mass, gravity, qdp, qp, gain):
    # Control Gains
    Kp = gain*np.eye(3, 3)
    # Split values
    xp = qp[0]
    yp = qp[1]
    zp = qp[2]

    xdp = qdp[0]
    ydp = qdp[1]
    zdp = qdp[2]

    # Control error
    error = qdp - qp

    error_vector = error.reshape((3,1))

    # Control Law
    aux_control = Kp@error_vector

    # Gravity + compensation velocity
    control_value = mass*gravity + aux_control[2,0]
    
    return control_value

def controller_attitude(rate_d, rate, gain_p, gain_q, gain_r):
    # Control Gains
    Kp = np.array([[gain_p, 0, 0],[0, gain_q, 0],[0, 0, gain_r]], dtype=np.double)
    # Split values
    pd = rate_d[0]
    qd = rate_d[1]
    rd= rate_d[2]

    p = rate[0]
    q = rate[1]
    r = rate[2]

    # Control error
    error = rate_d - rate 

    error_vector = error.reshape((3,1))

    # Control Law
    aux_control = Kp@error_vector

    # Gravity + compensation velocity
    control_value = aux_control[0:3,0]
    
    return control_value

def get_system_force(model, data):
    # Section Read Force and torques of the system
    forcetorque_general = np.zeros((6,), dtype=np.double)
    force = np.zeros((3,), dtype=np.double)
    for j,c in enumerate(data.contact):
        mujoco.mj_contactForce(model, data, j, forcetorque_general)
        force = force + forcetorque_general[0:3]
    norm_force = np.linalg.norm(force)
    return norm_force

def get_system_states_pos_sensor(system):
    q0 = system.sensor("position_drone").data.copy()
    x = np.array([q0], dtype=np.double)
    return x

def get_system_states_vel_sensor(system):
    q0 = system.sensor("linear_velocity_drone").data.copy()
    x = np.array([q0], dtype=np.double)
    return x

def get_system_states_vel_a_sensor(system):
    q0 = system.sensor("angular_velocity_drone").data.copy()
    x = np.array([q0], dtype=np.double)
    return x

def get_system_states_ori_sensor(system):
    q0 = system.sensor("quat_drone").data.copy()
    x = np.array([q0[1], q0[2], q0[3], q0[0]], dtype=np.double)
    r = R.from_quat(x)
    return r.as_euler('xyz', degrees=True)

def get_system_states_quat_sensor(system):
    q0 = system.sensor("quat_drone").data.copy()
    x = np.array([q0], dtype=np.double)
    return x

def main():
    # Load Model form XML file
    m = mujoco.MjModel.from_xml_path('drone.xml')
    # Print color of the box and position of the red box

    # Get information form the xml
    data = mujoco.MjData(m)

    # Simulation time parameters
    ts = 0.001
    tf = 30
    t = np.arange(0, tf+ts, ts, dtype=np.double)

    # Parameters of the entire system
    mass = 0.4 + 4*(0.025) + 4*(0.015)
    g = 9.81

    # States System pose
    q = np.zeros((3, t.shape[0]+1), dtype=np.double)
    # States System Ori
    n = np.zeros((3, t.shape[0]+1), dtype=np.double)
    quat = np.zeros((4, t.shape[0]+1), dtype=np.double)

    # States System pose
    qp = np.zeros((3, t.shape[0]+1), dtype=np.double)
    rate_b = np.zeros((3, t.shape[0]+1), dtype=np.double)

    # States System Ori
    
    # Force Vector
    force = np.zeros((1, t.shape[0] + 1), dtype=np.double)

    # Control signals
    u = np.zeros((4, t.shape[0]), dtype=np.double)

    # Desired reference signals of the system
    qdp = np.zeros((3, t.shape[0]), dtype=np.double)
    qdp[2,:] = 0.5
    rate_d = np.zeros((3, t.shape[0]), dtype=np.double)
    rate_d[2,:] = 0.1*np.sin(5*t)

    # Controllers gains
    kp = 50

    # Define Paramerters for the software
    m.opt.timestep = ts

    # Reset Properties system
    mujoco.mj_resetDataKeyframe(m, data, 0)  # Reset the state to keyframe 0

    # Set initial Conditions
    data.qpos[0] = 0
    data.qpos[1] = 0
    data.qpos[2] = 0

    with mujoco.viewer.launch_passive(m, data) as viewer:
        if viewer.is_running():
            # Initial data System
            q[:, 0] = get_system_states_pos_sensor(data)
            qp[:, 0] = get_system_states_vel_sensor(data)
            n[: ,0] = get_system_states_ori_sensor(data)
            quat[: ,0] = get_system_states_quat_sensor(data)
            rate_b[: ,0] = get_system_states_vel_a_sensor(data)

            # Simulation of the system
            for k in range(0, t.shape[0]):
                tic = time.time()
                u[0, k] = controller_z(mass, g, qdp[:, k], qp[:,k ], kp)
                u[1:4, k] = controller_attitude(rate_d[:, k], rate_b[:,k], 1, 1, 1)
                control_action(data, u[:,k])

                # System evolution
                mujoco.mj_step(m, data)

                # System evolution visualization
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                # Get system states
                force[:, k+1] = get_system_force(m, data)
                q[:, k+1] = get_system_states_pos_sensor(data)
                qp[:, k+1] = get_system_states_vel_sensor(data)
                n[: ,k+1] = get_system_states_ori_sensor(data)
                quat[: ,k+1] = get_system_states_quat_sensor(data)
                rate_b[: ,k+1] = get_system_states_vel_a_sensor(data)


                # Section to guarantee same sample times
                while (time.time() - tic <= m.opt.timestep):
                    None
                toc = time.time() - tic 
                print(toc)

        fig1, ax1, ax2 = fancy_plots_2()
        ## Axis definition necesary to fancy plots
        ax1.set_xlim((t[0], t[-1]))
        ax2.set_xlim((t[0], t[-1]))
        ax1.set_xticklabels([])
        state_xd, = ax1.plot(t,qdp[0,0:t.shape[0]],
                    color='#9C1816', lw=2, ls="-")

        state_yd, = ax1.plot(t,qdp[1,0:t.shape[0]],
                    color='#179237', lw=2, ls="-")

        state_zd, = ax1.plot(t,qdp[2,0:t.shape[0]],
                    color='#175E92', lw=2, ls="-")
        state_x, = ax1.plot(t,qp[0,0:t.shape[0]],
                    color='#BB5651', lw=2, ls="-.")
        state_y, = ax1.plot(t,qp[1,0:t.shape[0]],
                    color='#76BB51', lw=2, ls="-.")
        state_z, = ax1.plot(t,qp[2,0:t.shape[0]],
                    color='#518EBB', lw=2, ls="-.")
        ax1.set_ylabel(r"$[m/s]$", rotation='vertical')
        ax1.legend([state_x,state_y,state_z,state_xd,state_yd,state_zd],
            [r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$', r'$\dot{x}_d$', r'$\dot{y}_d$', r'$\dot{z}_d$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
        ax1.grid(color='#949494', linestyle='-.', linewidth=0.5)

        ax2.set_xlim((t[0], t[-1]))
        ax2.set_xticklabels([])
        state_pd, = ax2.plot(t,rate_d[0,0:t.shape[0]],
                    color='#9C1816', lw=2, ls="-")

        state_qd, = ax2.plot(t,rate_d[1,0:t.shape[0]],
                    color='#179237', lw=2, ls="-")

        state_rd, = ax2.plot(t,rate_d[2,0:t.shape[0]],
                    color='#175E92', lw=2, ls="-")

        state_p, = ax2.plot(t,rate_b[0,0:t.shape[0]],
                    color='#BB5651', lw=2, ls="-.")
        state_q, = ax2.plot(t,rate_b[1,0:t.shape[0]],
                    color='#76BB51', lw=2, ls="-.")
        state_r, = ax2.plot(t,rate_b[2,0:t.shape[0]],
                    color='#518EBB', lw=2, ls="-.")
        ax2.set_ylabel(r"$[r/s]$", rotation='vertical')
        ax2.legend([state_p,state_q,state_r,state_pd,state_qd,state_rd],
            [r'${p}$', r'${q}$', r'${r}$', r'${p}_d$', r'${q}_d$', r'${r}_d$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
        ax2.grid(color='#949494', linestyle='-.', linewidth=0.5)

        fig1.savefig("system_states_controller.eps")
        fig1.savefig("system_states_controller.png")
        fig1
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