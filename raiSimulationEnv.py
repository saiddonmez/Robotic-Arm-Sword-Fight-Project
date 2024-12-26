import gymnasium as gym
import numpy as np
import time
import robotic as ry


class RobotSimEnv(gym.Env):
    def __init__(self, render=False):
        super(RobotSimEnv, self).__init__()
        
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1, -1, -1,  -1, -1]), 
                                       high=np.array([1,  1,  1, 1, 1, 1, 1]), 
                                       dtype=np.float32)
        
        # Observation space: The agent's position in the 2D plane [x, y]
        self.observation_space = gym.spaces.Box(low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973,  0.5   , -2.8973,-2.8973, -1.7628, -2.8973, -3.0718, -2.8973,  0.5   , -2.8973, -9.70, -4.18, -8.32, -5.35, -5.96 ,-13.71, -6.12, -9.70, -4.18, -8.32, -5.35, -5.96 ,-13.71, -6.12]), 
                                       high=np.array([2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.    ,  2.8973, 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.    ,  2.8973, 9.34, 13.23, 10.33, 8.79, 6.37, 5.69, 5.11,9.34, 13.23, 10.33, 8.79, 6.37, 5.69, 5.11]), 
                                       dtype=np.float32)
        
        # Initial state
        self.state = np.array([ 0.  , -1.  ,  0.  , -2.  ,  0.  ,  2.  ,  0.  ,  0.  ,
       -1.  ,  0.  , -2.  ,  0.  ,  2.  ,  0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # Maximum steps per episode
        self.max_steps = 100
        self.current_steps = 0

        self.renderAll = render


        self.C = ry.Config()
        self.C.addFile("rai-robotModels/scenarios/pandasFight.g")
        self.C.setJointState(np.array([ 0.  , -1.  ,  0.  , -2.  ,  0.  ,  2.  ,  0.  ,  0.  ,
            -1.  ,  0.  , -2.  ,  0.  ,  2.  ,  0.]))

        self.initialFrameState = self.C.getFrameState()
        qhome = self.C.getJointState()
        self.q0 = qhome
        self.armed = np.load('armed.npy')

        #self.simulation = initializeSimulation(render=self.renderAll)

    def reset(self, initial_state=None, randomize = False, seed=None):
        self.initializeSimulation(render=self.renderAll)
        """Reset the environment to an initial state."""
        self.current_steps = 0
        self.initial_state = initial_state
        info = None
        if initial_state is not None:
            info = self.simulationGoTo(initial_state[:14],render=self.renderAll)
            self.state = np.concatenate([self.simulation.get_q(), self.simulation.get_qDot()])
        else:
            self.state = np.array([ 0.  , -1.  ,  0.  , -2.  ,  0.  ,  2.  ,  0.  ,  0.  ,
        -1.  ,  0.  , -2.  ,  0.  ,  2.  ,  0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            if randomize == True:
                if seed is not None:
                    np.random.seed(seed)
                q = 0.2*np.random.uniform(-np.pi, np.pi, size=self.state.shape[0])
                q[14:] = 0
                self.state += q
            self.initial_state = self.state
        return self.state, info

    def step(self, action, target=None):
        """Apply an action and return the new state, reward, done, and info."""
        self.current_steps += 1
        actionAddition = action
        # Clip the action to ensure it stays within the action space bounds
        
        action = self.state[:14]
        action[:7] += actionAddition
        action[:7] = np.clip(action[:7], self.action_space.low, self.action_space.high)
        # Update the state based on the action
        self.simulation.step(action, 0.01, ry.ControlMode.position)
        self.state = np.concatenate([self.simulation.get_q(), self.simulation.get_qDot()])

        reward = 0
        
        if target is not None:
            # Calculate the distance to the target
            distance_to_target = (1)*np.max(self.state[:14] - target[:14])

            # Reward is the negative distance to the target
            reward -= distance_to_target
            #print("reward: ",reward)

        # Speed penalty
        #reward -= 0.001*np.linalg.norm(self.simulation.get_qDot())
        # only give reward if sword collides with objects starting with r_
        #if self.state - self.observation_space.low < 0.01:
        selfCollision = False
        success = False
        done = False
        swordFailedHit = False

        # cols = self.C.getCollisions(-0.001)
        # for col in cols:
        #     if 'sword_1' in col:
        #         if col[0].startswith('r_') or col[1].startswith('r_'):
        #             reward += 1
        #             success = True
        #             done = True
        #             break
        #         else:
        #             reward = -0.5
        #             done = True
        #             swordFailedHit = True
        #             break
        #     if col[0].startswith('l_') and col[1].startswith('l_'):
        #         reward = -1
        #         done = True
        #         selfCollision = True
        #         break
        # if self.findRewardingCollision('sword_1'):
        #     success = True
        #     reward += 1
        # else:
        #     success = False

        # Check if the agent reached the target (within a small threshold)
        #done = self.findCollision('sword_1')
        #print("done : ",done)
        # Info dictionary (optional)
        info= {"is_success": success, "self_collision": selfCollision, "sword_failed_hit": swordFailedHit}
        
        return self.state, reward, done, info

    def render(self, mode='human'):
        """Render the environment (print the current state)."""
        self.C.view()
        time.sleep(0.01)

    def close(self):
        """Clean up resources (optional)."""
        del self.simulation
        del self.C


    def simulationCloseGrippers(self, tau=0.01, render=True):
        self.simulation.closeGripper('l_gripper', width=0.0001, speed=1)
        while (not self.simulation.getGripperIsGrasping('l_gripper')) and (self.simulation.getGripperWidth('l_gripper') > 0.001):
            self.simulation.step([], tau, ry.ControlMode.none)
            if render:
                self.C.view()
                time.sleep(tau)

        self.simulation.closeGripper('r_gripper', width=0.0001, speed=1)
        while (not self.simulation.getGripperIsGrasping('r_gripper')) and (self.simulation.getGripperWidth('r_gripper') > 0.001):
            self.simulation.step([], tau, ry.ControlMode.none)
            if render:
                self.C.view()
                time.sleep(tau)

    def simulationOpenGrippers(self, tau=0.01, render=True):
        self.simulation.openGripper('l_gripper', width=0.05, speed=0.5)
        while self.simulation.getGripperWidth('l_gripper') < 0.05 - 0.01:
            self.simulation.step([], tau, ry.ControlMode.none)
            if render:
                self.C.view()
                time.sleep(tau)
        self.simulation.openGripper('r_gripper', width=0.05, speed=0.5)
        while self.simulation.getGripperWidth('r_gripper') < 0.05 - 0.01:
            self.simulation.step([], tau, ry.ControlMode.none)
            if render:
                self.C.view()
                time.sleep(tau)

    def simulationGoTo(self, q, tau=0.01, checkCol=False, render=True):
        checkColTime = 0.1
        timer = 0
        while np.linalg.norm(self.simulation.get_q() - q) > 0.01:
            self.simulation.step(q, tau, ry.ControlMode.position)
            timer += tau
            if render:
                self.C.view()
                time.sleep(tau)

            if timer > 2:
                print("Target cannot be reached within 2 seconds.")
                return "failedReach"
            if checkCol:
                if timer > checkColTime:
                    if self.findCollision(self.C, 'sword_1'):
                        return "collision"
                    else:
                        checkColTime += 0.1

    def simulationGoHome(self, tau=0.01, render=True):
        checkColTime = 0.1
        timer = 0
        while np.linalg.norm(self.simulation.get_q() - self.q0) > 0.01:
            self.simulation.step(self.q0, tau, ry.ControlMode.position)
            timer += tau
            if render:
                self.C.view()
                time.sleep(tau)
            # if timer > checkColTime:
            #     if findCollision(C, 'sword_1'):
            #         break
            #     else:
            #         checkColTime += 0.1
                
    # def simulationFollowPath(self, path, tau=0.01, render=True):
    #     checkColTime = 0.1
    #     timer = 0
    #     for i in range(len(path)):
    #         while np.linalg.norm(S.get_q() - path[i]) > 0.3:
    #             print(i)
    #             print(np.linalg.norm(S.get_q() - path[i]))
    #             S.step(path[i], tau, ry.ControlMode.position)
    #             timer += tau
    #             if render:
    #                 C.view()
    #                 time.sleep(tau)
    #             if timer > checkColTime:
    #                 print('check_col')
    #                 if findCollision(C, 'sword_1'):
    #                     return
    #                 else:
    #                     checkColTime += 0.1

    #     self.simulationGoTo(S, path[-1], checkCol=True,render=render)

    def simulationWait(self, t, tau=0.01, render=True):
        for k in range(int(t / tau)):
            self.simulation.step([], tau, ry.ControlMode.none)
            if render:
                self.C.view()
                time.sleep(tau)

    def simulationTorqueCtrl(self, t, torque, tau=0.01, render=True):
        for k in range(int(t / tau)):
            self.simulation.step(torque, tau, ry.ControlMode.acceleration)
            if render:
                self.C.view()
                time.sleep(tau)

    def simulationVelocityCtrl(self,t,vel,tau=0.01, render=True):
        for k in range(int(t/tau)):
            self.simulation.step(vel,tau,ry.ControlMode.velocity)
            if render:
                self.C.view()
                time.sleep(tau)

    # def followInterpolatedPath(S,path,tau=0.01, render=True):
    #     checkColTime = 0.1
    #     timer = 0
    #     for i in range(len(path)):
    #         print(i)
    #         print(np.linalg.norm(S.get_q()- path[i]))
    #         S.step(path[i],tau,ry.ControlMode.position)
    #         if render:
    #             C.view()
    #             time.sleep(tau)
    #         timer += tau
    #         if timer > checkColTime:
    #             if findCollision(C,'sword_1'):
    #                 return
    #             else:
    #                 checkColTime += 0.1
    #     simulationGoTo(S,path[-1],checkCol=True,render=render)

    def findCollision(self, object1):
        collisions = [col for col in self.C.getCollisions(0) if object1 in col and not col[1].startswith('l_') and not col[0].startswith('l_')]
        if len(collisions) > 0:
            return True
        else:
            return False
        
    def findRewardingCollision(self,object1):
        collisions = [col for col in self.C.getCollisions(-0.001) if object1 in col and col[1].startswith('r_') or col[0].startswith('r_')]
        if len(collisions) > 0:
            return True
        else:
            return False
        
        
    # Use this code. It is awesome. It will follow the path and stop if there is a collision.    
    def followSplinePath(self,path,t,tau=0.01, render=True):
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
        self.simulation.resetSplineRef() # Reset previous spline reference
        self.simulation.setSplineRef(path,np.linspace(0.01,t,len(path))) # Set new spline reference
        joint_data = np.empty((int(t/tau)+20, 2, self.simulation.get_q().shape[0])) # Initialize array to store joint data
        for k in range(int(t/tau)+20): # This +20 is heuristic. It was stopping after a short time when t was low.
            self.simulation.step([],tau,ry.ControlMode.spline)
            timer += tau
            if render:
                self.C.view()
                time.sleep(tau)
            inst_pos = self.simulation.get_q()
            inst_vel = self.simulation.get_qDot()
            joint_data[k,:,:] = [inst_pos,inst_vel]
            if timer > checkColTime: # Check for collisions every 0.1 seconds
                if self.findCollision('sword_1'): # Check if sword is colliding with something (left robot is excluded (no self collision))
                    return joint_data[:k,:,:][np.newaxis,:,:,:] # Return the joint data until collision
                else:
                    checkColTime += 0.1 # Increment the collision check time

        return joint_data[np.newaxis,:,:,:]
        #simulationGoTo(S,path[-1],checkCol=True) 

    def initializeSimulation(self,render=True):
        self.C.setFrameState(self.initialFrameState)
        self.simulation = ry.Simulation(self.C, ry.SimulationEngine.physx, verbose=0)
        if render:
            self.C.view()
        self.simulationGoTo(self.armed,render=render)
        self.simulationCloseGrippers(render=render)
        self.simulationGoHome(render=render)
