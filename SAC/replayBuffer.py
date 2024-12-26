import numpy as np

class ReplayBuffer(): # Hindsight Experience Replay Buffer

    def __init__(self,size, state_dim, action_dim): # size is the maximum number of samples that can be stored in the buffer
        self.dataset = {'state':np.zeros((size,state_dim),dtype=np.float32),
                        'action':np.zeros((size,action_dim),dtype=np.float32),
                        'reward':np.zeros(size,dtype=np.float32),
                        'nextState':np.zeros((size,state_dim),dtype=np.float32),
                        'terminated':np.zeros(size,dtype=bool)
                        }
        
        self.lastIndex = 0 # Stores the latest index where the sample was stored
        self.size = size # Maximum size of the buffer
        self.full = False # True if the buffer is full
        
    def recordSample(self,state,action,reward,nextState,terminated):
        
        if (self.lastIndex == self.size): # If the buffer is full then cycle back to the beginning
            self.lastIndex = 0
            self.full =  True # Set the flag to indicate that the buffer is full
        else:
            # Store the sample in the buffer
            self.dataset['state'][self.lastIndex,...] = state
            self.dataset['action'][self.lastIndex,...] = action
            self.dataset['reward'][self.lastIndex,...] = reward
            self.dataset['nextState'][self.lastIndex,...] = nextState
            self.dataset['terminated'][self.lastIndex,...] = terminated

            self.lastIndex += 1 # Increment the index to store the next sample
            
    def recordEpisode(self,episodeData):
        for i in range(len(episodeData)):
            # add actual episode samples
            state, action, reward, nextState, done = episodeData[i]
            self.recordSample(state, action, reward, nextState, done)


    def sample(self,batchSize):
        # Sample batchSize samples from the buffer
        if self.full:
            randomIndex = np.random.choice(self.size, batchSize, replace=False)
        else:
            randomIndex = np.random.choice(self.lastIndex, batchSize, replace=False)    
        sampledBatch = {key: self.dataset[key][randomIndex]
                        for key in self.dataset.keys()
                        }
        return sampledBatch
    