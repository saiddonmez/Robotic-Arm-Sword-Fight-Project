import gymnasium as gym
import numpy as np
import time
import robotic as ry

C = ry.Config()
C.addFile("rai-robotModels/scenarios/pandasFight.g")
C.setJointState(np.array([ 0.  , -1.  ,  0.  , -2.  ,  0.  ,  2.  ,  0.  ,  0.  ,
       -1.  ,  0.  , -2.  ,  0.  ,  2.  ,  0.]))

C.view()

initialFrameState = C.getFrameState()
qhome = C.getJointState()
q0 = qhome
armed = np.load('armed.npy')

def simulationCloseGrippers(S, tau=0.01, render=True):
    S.closeGripper('l_gripper', width=0.0001, speed=1)
    while (not S.getGripperIsGrasping('l_gripper')) and (S.getGripperWidth('l_gripper') > 0.001):
        S.step([], tau, ry.ControlMode.none)
        if render:
            C.view()
            time.sleep(tau)

    S.closeGripper('r_gripper', width=0.0001, speed=1)
    while (not S.getGripperIsGrasping('r_gripper')) and (S.getGripperWidth('r_gripper') > 0.001):
        S.step([], tau, ry.ControlMode.none)
        if render:
            C.view()
            time.sleep(tau)

def simulationOpenGrippers(S, tau=0.01, render=True):
    S.openGripper('l_gripper', width=0.05, speed=0.5)
    while S.getGripperWidth('l_gripper') < 0.05 - 0.01:
        S.step([], tau, ry.ControlMode.none)
        if render:
            C.view()
            time.sleep(tau)
    S.openGripper('r_gripper', width=0.05, speed=0.5)
    while S.getGripperWidth('r_gripper') < 0.05 - 0.01:
        S.step([], tau, ry.ControlMode.none)
        if render:
            C.view()
            time.sleep(tau)

def simulationGoTo(S, q, tau=0.01, checkCol=False, render=True):
    checkColTime = 0.1
    timer = 0
    while np.linalg.norm(S.get_q() - q) > 0.01:
        S.step(q, tau, ry.ControlMode.position)
        timer += tau
        if render:
            C.view()
            time.sleep(tau)

        if timer > 2:
            print("Target cannot be reached within 2 seconds.")
            break
        if checkCol:
            if timer > checkColTime:
                if findCollision(C, 'sword_1'):
                    return
                else:
                    checkColTime += 0.1

def simulationGoHome(S, tau=0.01, render=True):
    checkColTime = 0.1
    timer = 0
    while np.linalg.norm(S.get_q() - q0) > 0.01:
        S.step(q0, tau, ry.ControlMode.position)
        timer += tau
        if render:
            C.view()
            time.sleep(tau)
        # if timer > checkColTime:
        #     if findCollision(C, 'sword_1'):
        #         break
        #     else:
        #         checkColTime += 0.1
            
def simulationFollowPath(S, path, tau=0.01, render=True):
    checkColTime = 0.1
    timer = 0
    for i in range(len(path)):
        while np.linalg.norm(S.get_q() - path[i]) > 0.3:
            print(i)
            print(np.linalg.norm(S.get_q() - path[i]))
            S.step(path[i], tau, ry.ControlMode.position)
            timer += tau
            if render:
                C.view()
                time.sleep(tau)
            if timer > checkColTime:
                print('check_col')
                if findCollision(C, 'sword_1'):
                    return
                else:
                    checkColTime += 0.1

    simulationGoTo(S, path[-1], checkCol=True)

def simulationWait(S, t, tau=0.01, render=True):
    for k in range(int(t / tau)):
        S.step([], tau, ry.ControlMode.none)
        if render:
            C.view()
            time.sleep(tau)

def simulationTorqueCtrl(S, t, torque, tau=0.01, render=True):
    for k in range(int(t / tau)):
        S.step(torque, tau, ry.ControlMode.acceleration)
        if render:
            C.view()
            time.sleep(tau)

def simulationVelocityCtrl(S,t,vel,tau=0.01, render=True):
    for k in range(int(t/tau)):
        S.step(vel,tau,ry.ControlMode.velocity)
        if render:
            C.view()
            time.sleep(tau)

def followInterpolatedPath(S,path,tau=0.01, render=True):
    checkColTime = 0.1
    timer = 0
    for i in range(len(path)):
        print(i)
        print(np.linalg.norm(S.get_q()- path[i]))
        S.step(path[i],tau,ry.ControlMode.position)
        if render:
            C.view()
            time.sleep(tau)
        timer += tau
        if timer > checkColTime:
            if findCollision(C,'sword_1'):
                return
            else:
                checkColTime += 0.1
    simulationGoTo(S,path[-1],checkCol=True)

def findCollision(C,object1):
    collisions = [col for col in C.getCollisions(0) if object1 in col and not col[1].startswith('l_') and not col[0].startswith('l_')]
    if len(collisions) > 0:
        return True
    else:
        return False
    
def findRewardingCollision(C,object1):
    collisions = [col for col in C.getCollisions(0) if object1 in col and col[1].startswith('r_') or col[0].startswith('r_')]
    if len(collisions) > 0:
        return True
    else:
        return False
    
    
# Use this code. It is awesome. It will follow the path and stop if there is a collision.    
def followSplinePath(S,path,t,tau=0.01, render=True):
    """
        Guides a simulation object along a specified spline path.
        Parameters:
        S (Simulation): The simulation object that will follow the path.
        path (array): A list of waypoints defining the spline path.
        t (float): The total time duration for following the path.
        tau (float, optional): The time step for each simulation step. Default is 0.01.
        The function performs the following steps:
        1. Resets the spline reference of the simulation object.
        2. Sets the spline reference with the given path and a time vector.
        3. Iteratively steps through the simulation, updating the view and checking for collisions.
        4. If a collision with 'sword_1' is detected, the function exits early.
        5. The loop runs for a duration slightly longer than `t` to ensure completion.
        Note:
        - The `+20` in the loop range is a heuristic to prevent early termination when `t` is low.
        - The function checks for collisions every 0.1 seconds.

    """
    checkColTime = 0.1
    timer = 0
    S.resetSplineRef() # Reset previous spline reference
    S.setSplineRef(path,np.linspace(0.01,t,len(path))) # Set new spline reference
    joint_data = np.empty((int(t/tau)+20, 2,S.get_q().shape[0])) # Initialize array to store joint data
    for k in range(int(t/tau)+20): # This +20 is heuristic. It was stopping after a short time when t was low.
        S.step([],tau,ry.ControlMode.spline)
        timer += tau
        if render:
            C.view()
            time.sleep(tau)
        inst_pos = S.get_q()
        inst_vel = S.get_qDot()
        joint_data[k,:,:] = [inst_pos,inst_vel]
        if timer > checkColTime: # Check for collisions every 0.1 seconds
            if findCollision(C,'sword_1'): # Check if sword is colliding with something (left robot is excluded (no self collision))
                return joint_data[:k,:,:][np.newaxis,:,:,:] # Return the joint data until collision
            else:
                checkColTime += 0.1 # Increment the collision check time

    return joint_data[np.newaxis,:,:,:]
    #simulationGoTo(S,path[-1],checkCol=True) 

def initalizeSimulation(render=True):
    C.setFrameState(initialFrameState)
    S = ry.Simulation(C, ry.SimulationEngine.physx, verbose=0)
    C.view()
    simulationGoTo(S,armed,render=render)
    simulationCloseGrippers(S,render=render)
    simulationGoHome(S,render=render)
    return S


class RobotSimEnv(gym.Env):
    def __init__(self, render=False):
        super(RobotSimEnv, self).__init__()
        
        self.action_space = gym.spaces.Box(low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973,  0.5   , -2.8973]), 
                                       high=np.array([2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.    ,  2.8973]), 
                                       dtype=np.float32)
        
        # Observation space: The agent's position in the 2D plane [x, y]
        self.observation_space = gym.spaces.Box(low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973,  0.5   , -2.8973,-2.8973, -1.7628, -2.8973, -3.0718, -2.8973,  0.5   , -2.8973, -9.70, -4.18, -8.32, -5.35, -5.96 ,-13.71, -6.12, -1.83, -2.05, -1.55, -1.45, -1.44, -0.44, -0.14, -9.70, -4.18, -8.32, -5.35, -5.96 ,-13.71, -6.12, -1.83, -2.05, -1.55, -1.45, -1.44, -0.44, -0.14]), 
                                       high=np.array([2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.    ,  2.8973, 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.    ,  2.8973, 9.34, 13.23, 10.33, 8.79, 6.37, 5.69, 5.11, 2.32, 0.87, 1.05, 1.26, 4.17, 0.40, 0.22,9.34, 13.23, 10.33, 8.79, 6.37, 5.69, 5.11, 2.32, 0.87, 1.05, 1.26, 4.17, 0.40, 0.22]), 
                                       dtype=np.float32)
        
        # Initial state
        self.state = np.array([ 0.  , -1.  ,  0.  , -2.  ,  0.  ,  2.  ,  0.  ,  0.  ,
       -1.  ,  0.  , -2.  ,  0.  ,  2.  ,  0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # Maximum steps per episode
        self.max_steps = 100
        self.current_steps = 0

        self.renderAll = render

        self.simulation = initalizeSimulation()

    def reset(self, initial_state=None):
        self.simulation = initalizeSimulation()
        """Reset the environment to an initial state."""
        self.current_steps = 0
        self.initial_state = initial_state
        if initial_state is not None:
            simulationGoTo(self.simulation, initial_state[:14])
            self.state = np.concatenate([self.simulation.get_q(), self.simulation.get_qDot()])
        else:
            self.state = np.array([ 0.  , -1.  ,  0.  , -2.  ,  0.  ,  2.  ,  0.  ,  0.  ,
       -1.  ,  0.  , -2.  ,  0.  ,  2.  ,  0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        return self.state

    def step(self, action, target=None):
        """Apply an action and return the new state, reward, done, and info."""
        self.current_steps += 1
        
        # Clip the action to ensure it stays within the action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        action = np.concatenate([action, self.initial_state[7:14]])
        # Update the state based on the action
        self.simulation.step(action, 0.01, ry.ControlMode.position)
        self.state = np.concatenate([self.simulation.get_q(), self.simulation.get_qDot()])

        if target is not None:
            # Calculate the distance to the target
            distance_to_target = np.linalg.norm(self.state - target)/10

            # Reward is the negative distance to the target
            reward = -distance_to_target
            print("reward: ",reward)
        
        else:
            # only give reward if sword collides with objects starting with r_
            if findRewardingCollision(C, 'sword_1'):
                reward = 1
            else:
                reward = 0

        # Check if the agent reached the target (within a small threshold)
        done = findCollision(C, 'sword_1')
        print("done : ",done)
        # Info dictionary (optional)
        info = {}
        
        return self.state, reward, done, info

    def render(self, mode='human'):
        """Render the environment (print the current state)."""
        C.view()
        time.sleep(0.01)

    def close(self):
        """Clean up resources (optional)."""
        del S
        del C