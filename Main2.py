import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from TorchQ import Agent
import matplotlib.pyplot as plt

#ENVS Declaration
channel = EngineConfigurationChannel()
env     = UnityEnvironment(file_name = "CoE202", side_channels = [channel])
channel.set_configuration_parameters(time_scale = 5, width = 600, height = 400)

#Begin Env
env.reset()
PurpleBehavior  = list(env.behavior_specs)[0]
BlueBehavior    = list(env.behavior_specs)[1]

#Define Functions
def sensor_front_sig(data):
    player      = []
    sensor_data = []

    for sensor in range(33):
        player.append(data[8*sensor : (8*sensor) + 8])

    for stack in range(3):
        sensor_data.append(player[11*stack : (11*stack) + 11])
    
    return sensor_data

def sensor_back_sig(data):
    player      = []
    sensor_data = []
    for sensor in range(9):
        player.append(data[8*sensor : (8*sensor) + 8])

    for stack in range(3):
        sensor_data.append(player[3*stack : (3*stack) + 3])
    return sensor_data

def plot(xDataSet, yDataSet, xLabel, yLabel, path):
    for i in yDataSet:
        plt.plot(xDataSet, yDataSet)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig('%s.png' % path)

#define class for player stuff
class Defender(Agent):
    def __init__(self, behavior, index, MemSize, BatchSize, 
    Output, Input, Gamma, EpsStart, EpsDecay, EpsEnd, MaxNN, Path):
        self.Behavior   = behavior
        self.Action     = [0, 0, 0] 
        self.ActSpace   = tuple([
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
            [0, 1, 0], [0, 2, 0]
        ])
        self.ActNum     = 0
        self.Reward     = 0
        self.TermState  = False
        super().__init__(MemSize, BatchSize, Input, Output, Gamma, EpsStart, EpsDecay, EpsEnd, MaxNN, Path)
        self.Dec        = None
        self.Next       = []
        self.State      = []
        self.Index      = index
    
    def RefreshDec(self):
        self.Dec , _    = env.get_steps(self.Behavior)
    
    def Read(self):
        self.RefreshDec()
        self.Next       = list()
        Front           = sensor_front_sig(self.Dec.obs[0][self.Index, :])
        Back            = sensor_back_sig(self.Dec.obs[1][self.Index, :])
        for i in range(0, 3):
            for j in range(0, 10):
                for k in range(0, 7):
                    self.Next.append(Front[i][j][k]*Front[i][j][7])
            for j in range(0, 3):
                for k in range(0, 7):
                    self.Next.append(Back[i][j][k]*Back[i][j][7])
    
    def UpdateState(self):
        self.State      = self.Next

    def SetAction(self):
        self.ActNum     = self.ChooseAction(self.State)
        self.Action     = self.ActSpace[self.ActNum]
    
    def GetAction(self):
        return self.Action
    
    def SetReward(self):
        self.RefreshDec()
        if(self.Dec.reward[self.Index] < 0):
            self.Reward     =   -2
            self.Term       =   True
        elif(self.Dec.reward[self.Index] > 0):
            self.Reward     =   0
            self.Term       =   True
        else:  
            self.Reward     =   0.001
            self.Term       =   False
        
    def SaveData(self):
        self.SaveState(self.State, self.Reward, self.ActNum, self.Next, self.Term)
        self.Term = False
    
class Striker(Agent):
    def __init__(self, behavior, index, MemSize, BatchSize, 
    Output, Input, Gamma, EpsStart, EpsDecay, EpsEnd, MaxNN, Path):
        self.Behavior   = behavior
        self.Action     = [0, 0, 0] 
        self.ActSpace   = tuple([
            [0, 0, 0], [0, 0, 1], [0, 0, 2],
            [0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0]
        ])
        self.ActNum     = 0
        self.Reward     = 0
        self.TermState  = False
        super().__init__(MemSize, BatchSize, Input, Output, Gamma, EpsStart, EpsDecay, EpsEnd, MaxNN, Path)
        self.Dec        = None
        self.Next       = []
        self.State      = []
        self.Index      = index
    
    def RefreshDec(self):
        self.Dec , _    = env.get_steps(self.Behavior)
    
    def Read(self):
        self.RefreshDec()
        self.Next       = []
        Front           = sensor_front_sig(self.Dec.obs[0][self.Index, :])
        Back            = sensor_back_sig(self.Dec.obs[1][self.Index, :])
        for i in range(0, 3):
            for j in range(0, 10):
                for k in range(0, 7):
                    self.Next.append(Front[i][j][k]*Front[i][j][7])
            for j in range(0, 3):
                for k in range(0, 7):
                    self.Next.append(Back[i][j][k]*Back[i][j][7])
    
    def UpdateState(self):
        self.State      = self.Next

    def SetAction(self):
        self.ActNum     = self.ChooseAction(self.State)
        self.Action     = self.ActSpace[self.ActNum]
    
    def GetAction(self):
        return self.Action
    
    def SetReward(self):
        self.RefreshDec()
        if(self.Dec.reward[self.Index] < 0):
            self.Reward     =   0
            self.Term       =   True
        elif(self.Dec.reward[self.Index] > 0):
            self.Reward     =   2
            self.Term       =   True
        else:   
            self.Reward     =   -0.001
            for i in range(0, 13):
                if(self.Next[i*7] != 0):
                    self.Reward += 1/self.Next[i*7]*(1e-5)
            self.Term       =   False
        
    def SaveData(self):
        self.SaveState(self.State, self.Reward, self.ActNum, self.Next, self.Term)

Blue    = [Defender(BlueBehavior, 0, 10000, 128, 5, 273, 0.99, 1.0, 0.0001, 0.1, 256, "DefP0"),
            Striker(BlueBehavior, 1, 10000, 128, 7, 273, 0.99, 1.0, 0.0001, 0.1, 256, "AttP0")]
Purple  = [Defender(BlueBehavior, 0, 10000, 128, 5, 273, 0.99, 1.0, 0.0001, 0.1, 256, "DefP1"),
            Striker(BlueBehavior, 1, 10000, 128, 7, 273, 0.99, 1.0, 0.0001, 0.1, 256, "AttP1")] 

for i in range(2):
    Blue[i].Read()
    #Purple[i].Read()
    Blue[i].UpdateState()
    #Purple[i].UpdateState()
scores = [list(), list()]
hist   = []
for episode in range(10):
    done = False
    score = [0, 0]
    count = 0
    env.reset()
    while not done:
        #Get Actions
        for j in range(2):
            Blue[j].UpdateState()
            Blue[j].SetAction()
            Purple[j].SetAction()

        #Set The Actions
        env.set_actions(BlueBehavior, np.array([Blue[0].GetAction(), Blue[1].GetAction()]))
        env.set_actions(PurpleBehavior, np.array([Purple[0].GetAction(), Purple[1].GetAction()]))
        env.step()

        for j in range(2):
            Blue[j].Read()
            Purple[j].Read()
            Blue[j].SetReward()
            #Purple[j].SetReward()
            Blue[j].SaveData()
            #Purple[j].SaveData()
            Blue[j].Train()
        #Check Resets
        score[0] += Blue[0].Reward
        score[1] += Blue[1].Reward
        if(Blue[0].Dec.reward[0] != 0):
            env.reset()
            print("Goal, Reset")
            done = True
        count+=1
        #Hard Update
    for j in range(2):
        Blue[j].UpdateNet()
        #Purple[j].UpdateNet()
        Blue[j].SaveNet()
        #Purple[j].SaveNet()
    env.reset()
    scores[0].append(score[0])
    scores[1].append(score[1])
    hist.append(episode)
    print(hist[episode], ' ' ,scores[0][episode], ' ', scores[1][episode])
    
plot(hist, scores, 'Episode', 'Score', 'Res')