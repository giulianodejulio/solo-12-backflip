import crocoddyl as croco
import example_robot_data
import numpy as np
import pinocchio as pin
from pinocchio.utils import *
import matplotlib.pyplot as plt



############################################################### DATA
## LISTS OF DATA
feet_name_list = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
hind_feet_name_list = ['HL_FOOT', 'HR_FOOT']
no_feet_list = []

################################################################ Global variables
robot = example_robot_data.load('solo12')
robot.model.effortLimit[6:] = np.ones(12) * 1.6
nq = robot.nq
q0 = robot.q0
nv = robot.nv
v0 = robot.v0

q_start = q0.copy()
# q_start[] = 
x0 = np.concatenate([q_start, v0])

pin.framesForwardKinematics(robot.model, robot.data, q0)
feet_ids = [robot.model.getFrameId(n) for n in feet_name_list]
feet_poses = [pin.updateFramePlacement(robot.model, robot.data, feet_id) for feet_id in feet_ids]
feet_pos_list = [feet_pose.translation for feet_pose in feet_poses]

## Physical constants
dt = 2e-3 # in [s]


## Creation state / actuation
state = croco.StateMultibody(robot.model)
actuation = croco.ActuationModelFloatingBase(state)

base_link_id  = robot.model.getFrameId('base_link')
lf_shoulder_id = robot.model.getFrameId('FL_SHOULDER')
rf_shoulder_id = robot.model.getFrameId('FR_SHOULDER')


actions_model = []

## Free Command Regulation Cost
u_free_res = croco.ResidualModelControl(state, actuation.nu)
u_free_cost = croco.CostModelResidual(state, u_free_res)

# ## Contact Command Regulation Cost
# u_contact_res = croco.ResidualModelControl(state, actuation.nu)
# u_contact_cost = croco.CostModelResidual(state, u_contact_res)

## State Regulation Cost
x_res = croco.ResidualModelState(state, x0, actuation.nu)
x_cost = croco.CostModelResidual(state, x_res)


def feet_contact_cost(contact_model, cost_model, stance_feet_name_list):
    motion_zero = pin.Motion(np.array([.0, .0, .0]), np.array([.0, .0, .0]))
    for idx, foot_name in enumerate(stance_feet_name_list):
        foot_id = robot.model.getFrameId(foot_name)
        foot_pos = feet_pos_list[idx]
        foot_contact = croco.ContactModel3D(state, foot_id, foot_pos, actuation.nu)
        contact_model.addContact(foot_name, foot_contact)
        foot_pos_res = croco.ResidualModelFrameTranslation(state, foot_id, foot_pos, actuation.nu)
        foot_pos_cost = croco.CostModelResidual(state, foot_pos_res)
        cost_model.addCost(foot_name+' position', foot_pos_cost, 10)
        foot_vel_res = croco.ResidualModelFrameVelocity(state, foot_id, motion_zero, pin.LOCAL, actuation.nu)
        foot_vel_cost = croco.CostModelResidual(state, foot_vel_res)
        cost_model.addCost(foot_name+' velocity', foot_vel_cost, 10)



###################################################################################### Action 1 Preparation (NAVIGATION)
action_1_T = int(0.4/dt) #0.5

## COSTS
action_1_cost_model  = croco.CostModelSum(state, actuation.nu)

# Reg cost
action_1_cost_model.addCost('uReg', u_free_cost, 1e2) #1e-3 
# action_1_cost_model.addCost('xReg', x_cost, 1e0)


## CONTACTS
action_1_contact_model = croco.ContactModelMultiple(state, actuation.nu)

# Feet Contact & Cost
feet_contact_cost(action_1_contact_model, action_1_cost_model, feet_name_list)

## Integrated Action
action_1_dif_model = croco.DifferentialActionModelContactFwdDynamics(state, actuation, action_1_contact_model, action_1_cost_model)
action_1_model = croco.IntegratedActionModelEuler(action_1_dif_model, dt)

actions_model = actions_model + [action_1_model]*action_1_T


###################################################################################### Action 2 Takeoff
## We need this action because we need a transition from contact to free differential action model and integral action models
action_2_T = 1

## COSTS
action_2_cost_model  = croco.CostModelSum(state, actuation.nu)

# Reg cost
action_2_cost_model.addCost('uReg', u_free_cost, 1e2)
# action_2_cost_model.addCost('xReg', x_cost, 1e-4)


## CONTACTS
action_2_contact_model = croco.ContactModelMultiple(state, actuation.nu)

# Feet Contact & Cost
feet_contact_cost(action_2_contact_model, action_2_cost_model, feet_name_list)

## Integrated Action
action_2_dif_model = croco.DifferentialActionModelContactFwdDynamics(state, actuation, action_2_contact_model, action_2_cost_model)
action_2_model = croco.IntegratedActionModelEuler(action_2_dif_model, dt)

actions_model = actions_model + [action_2_model]*action_2_T


###################################################################################### Action 3 Flying (NAVIGATION)
action_3_T = int(0.2/dt)
## COSTS
action_3_cost_model  = croco.CostModelSum(state, actuation.nu)

# Reg cost
action_3_cost_model.addCost('uReg', u_free_cost, 1e2)
action_3_cost_model.addCost('xReg', x_cost, 1e0)


## CONTACTS
action_3_contact_model = croco.ContactModelMultiple(state, actuation.nu)

# Feet Contact & Cost
feet_contact_cost(action_3_contact_model, action_3_cost_model, hind_feet_name_list)


## Integrated Action
action_3_dif_model = croco.DifferentialActionModelContactFwdDynamics(state, actuation, action_3_contact_model, action_3_cost_model)
action_3_model = croco.IntegratedActionModelEuler(action_3_dif_model, dt)

actions_model = actions_model + [action_3_model]*action_3_T


###################################################################################### Action 4 (TASK): Reach position to start the flight
action_4_T = 1

## COSTS
action_4_cost_model  = croco.CostModelSum(state, actuation.nu)

## Reg cost
action_4_cost_model.addCost('uReg', u_free_cost, 1e-4)

q_custom = q0.copy()

#HEIGHT OF THE BASE
# q_custom[2] = 1

#BASE ORIENTATION
torad = np.pi / 180
# pitch = -90 * torad
pitch = -40 * torad
q_custom[3] = 0
q_custom[4] = np.sin(pitch/2)
q_custom[5] = 0
q_custom[6] = np.cos(pitch/2)

x_custom = np.concatenate([q_custom, v0])
x4_res = croco.ResidualModelState(state, x_custom, actuation.nu)
# base_angle_weight = 1e2
base_angle_weight = 1e3
joint_angles_weight = 1e2
joint_velocities_weight = 0 #1e0
shoulder_weight = 0
shoulder_velocities_weight = 0                

x4_activation_weights = np.array([0]*3 
                                    + [base_angle_weight]*3 
                                    + [shoulder_weight]*1 + [joint_angles_weight]*2 
                                    + [shoulder_weight]*1 + [joint_angles_weight]*2 
                                    + [shoulder_weight]*1 + [joint_angles_weight]*2
                                    + [shoulder_weight]*1 + [joint_angles_weight]*2
                                    + [0]*6
                                    #  + [joint_velocities_weight]*12)
                                    +[shoulder_velocities_weight]*1 + [0]*2
                                    +[shoulder_velocities_weight]*1 + [0]*2
                                    +[shoulder_velocities_weight]*1 + [0]*2
                                    +[shoulder_velocities_weight]*1 + [0]*2 ) 
x4_activation = croco.ActivationModelWeightedQuad(x4_activation_weights)
x4_cost = croco.CostModelResidual(state, x4_activation, x4_res)
action_4_cost_model.addCost('final joints task cost', x4_cost, 1)


## CONTACTS
action_4_contact_model = croco.ContactModelMultiple(state, actuation.nu)

# Feet Contact & Cost
feet_contact_cost(action_4_contact_model, action_4_cost_model, hind_feet_name_list)


## Integrated Action
action_4_dif_model = croco.DifferentialActionModelContactFwdDynamics(state, actuation, action_4_contact_model, action_4_cost_model)
action_4_model = croco.IntegratedActionModelEuler(action_4_dif_model, dt)

actions_model = actions_model + [action_4_model]*action_4_T


# ###################################################################################### Action x Takeoff
# ## We need this action because we need a transition from contact to free differential action model and integral action models
# action_x_T = 1

# ## COSTS
# action_x_cost_model  = croco.CostModelSum(state, actuation.nu)

# # Reg cost
# action_x_cost_model.addCost('uReg', u_free_cost, 1e2)
# # action_x_cost_model.addCost('xReg', x_cost, 1e-4)


# ## CONTACTS
# action_x_contact_model = croco.ContactModelMultiple(state, actuation.nu)

# # Feet Contact & Cost
# feet_contact_cost(action_x_contact_model, action_x_cost_model, hind_feet_name_list)

# velocity = np.array([0, 0, 1, 0, 0, 0])
# foot_vel = pin.Motion(velocity)
# xx_hl_res = croco.ResidualModelFrameVelocity(state, robot.model.getFrameId('HL_FOOT'), foot_vel, pin.WORLD, actuation.nu)
# xx_hr_res = croco.ResidualModelFrameVelocity(state, robot.model.getFrameId('HR_FOOT'), foot_vel, pin.WORLD, actuation.nu)


# xx_hl_cost = croco.CostModelResidual(state, xx_hl_res)
# xx_hr_cost = croco.CostModelResidual(state, xx_hr_res)
# action_x_cost_model.addCost('hl cost', xx_hl_cost, 1e3)
# action_x_cost_model.addCost('hr cost', xx_hr_cost, 1e3)

# ## Integrated Action
# action_x_dif_model = croco.DifferentialActionModelContactFwdDynamics(state, actuation, action_x_contact_model, action_x_cost_model)
# action_x_model = croco.IntegratedActionModelEuler(action_x_dif_model, dt)

# actions_model = actions_model + [action_x_model]*action_x_T



# ###################################################################################### Action y Flying (NAVIGATION)
# action_y_T = int(0.2/dt)
# ## COSTS
# action_y_cost_model  = croco.CostModelSum(state, actuation.nu)

# # Reg cost
# action_y_cost_model.addCost('uReg', u_free_cost, 1e2)
# action_y_cost_model.addCost('xReg', x_cost, 1e0)


# ## CONTACTS
# action_y_contact_model = croco.ContactModelMultiple(state, actuation.nu)


# ## Integrated Action
# action_y_dif_model = croco.DifferentialActionModelFreeFwdDynamics(state, actuation, action_y_cost_model)
# action_y_model = croco.IntegratedActionModelEuler(action_y_dif_model, dt)

# actions_model = actions_model + [action_y_model]*action_y_T

# ###################################################################################### Action 5 (TASK): Reach position to start the flight
# ## COSTS
# action_5_cost_model  = croco.CostModelSum(state, actuation.nu)

# ## Reg cost
# action_5_cost_model.addCost('uReg', u_free_cost, 1e-4)

# q_custom = q0.copy()


# #BASE ORIENTATION
# torad = np.pi / 180
# # pitch = -90 * torad
# pitch = -80 * torad
# q_custom[3] = 0
# q_custom[4] = np.sin(pitch/2)
# q_custom[5] = 0
# q_custom[6] = np.cos(pitch/2)

# x_custom = np.concatenate([q_custom, v0])
# x5_res = croco.ResidualModelState(state, x_custom, actuation.nu)


# # base_angle_weight = 1e2
# base_angle_weight = 1e3
# joint_angles_weight = 1e2
# joint_velocities_weight = 0 #1e0
# shoulder_weight = 0
# shoulder_velocities_weight = 0                

# x5_activation_weights = np.array([0]*3 
#                                     + [base_angle_weight]*3 
#                                     + [shoulder_weight]*1 + [joint_angles_weight]*2 
#                                     + [shoulder_weight]*1 + [joint_angles_weight]*2 
#                                     + [shoulder_weight]*1 + [joint_angles_weight]*2
#                                     + [shoulder_weight]*1 + [joint_angles_weight]*2
#                                     + [0]*6
#                                     #  + [joint_velocities_weight]*12)
#                                     +[shoulder_velocities_weight]*1 + [0]*2
#                                     +[shoulder_velocities_weight]*1 + [0]*2
#                                     +[shoulder_velocities_weight]*1 + [0]*2
#                                     +[shoulder_velocities_weight]*1 + [0]*2 )
# x5_activation = croco.ActivationModelWeightedQuad(x5_activation_weights)
# x5_cost = croco.CostModelResidual(state, x5_activation, x5_res)

# action_5_cost_model.addCost('final joints task cost', x5_cost, 1)


# ## CONTACTS
# # action_5_contact_model = croco.ContactModelMultiple(state, actuation.nu)

# # Feet Contact & Cost
# # feet_contact_cost(action_5_contact_model, action_5_cost_model, hind_feet_name_list)


# ## Integrated Action
# action_5_dif_model = croco.DifferentialActionModelFreeFwdDynamics(state, actuation, action_5_cost_model)
# action_5_model = croco.IntegratedActionModelEuler(action_5_dif_model, dt)


############################################################################################################################################################################

## Solve
problem = croco.ShootingProblem(x0, actions_model, action_4_model)
solver = croco.SolverBoxFDDP(problem)
solver.setCallbacks([croco.CallbackVerbose()])
solver.solve([], [], 500)

# Save the trajectory
# np.savez('ressource/jump_move', xs=solver.xs, us=solver.us)
# np.savez('jump_solo12_05m', xs=solver.xs, us=solver.us)


# ----- DISPLAY -----

display = croco.GepettoDisplay(robot)
# display.displayFromSolver(solver, factor=1)
display.displayFromSolver(solver, factor=3)
# display.displayFromSolver(solver, factor=6)



# ----- PLOTS -----
ts_a = np.array([dt * i for i in range(len(solver.us))])
us_a = np.vstack(solver.us)

fig0, axs0 = plt.subplots(nrows=4, ncols=1)
names = [
    'FL_SHOULDER_0', 'FL_SHOULDER_1', 'FL_ELBOW', 'FR_SHOULDER_0',
    'FR_SHOULDER_1', 'FR_ELBOW', 'RL_SHOULDER_0', 'RL_SHOULDER_1', 'RL_ELBOW',
    'RR_SHOULDER_0', 'RR_SHOULDER_1', 'RR_ELBOW'
]
for idx, ax in enumerate(axs0):
    for i in range(3):
        ax.plot(ts_a, us_a[:, idx * 3 + i])
    ax.legend(names[idx * 3:idx * 3 + 3])

# Plot state base_link
ts_a = np.array([dt * i for i in range(len(solver.xs))])
xs_a = np.vstack(solver.xs)

print("last state: \n {}".format(xs_a[-1,:]))

# fig, ax = plt.subplots()
# ax.plot(ts_a, xs_a[:,2])
# ax.legend('z')


# quaternion_names=['qx','qy','qz','qw']
# fig1, axs1 = plt.subplots()
# for i in range(4):
#     axs1.plot(ts_a, xs_a[:, i+3])
# axs1.legend(quaternion_names)



# fig1, axs1 = plt.subplots(nrows=4, ncols=1)

# pos
# for i in range(3):
#     axs1[0].plot(ts_a, xs_a[:, i])
# axs1[0].legend(['x', 'y', 'z'])


# lin vel
# for i in range(3):
#     axs1[2].plot(ts_a, xs_a[:, robot.model.nq + i])
# axs1[2].legend(['x', 'y', 'z'])

# # Plot contact forces
# fs = display.getForceTrajectoryFromSolver(solver)
# fs_a = np.zeros([len(fs), 4])

# for idx_feet, f_feet in enumerate(fs):
#     for idx_foot, f_foot in enumerate(f_feet):
#         fs_a[idx_feet, idx_foot] = np.linalg.norm(f_foot['f'])

# ts_a = np.array([dt * i for i in range(len(fs))])

# fig2, axs2 = plt.subplots(nrows=4, ncols=1)

# for idx, ax in enumerate(axs2):
#     ax.plot(ts_a, fs_a[:, idx])

plt.show()
