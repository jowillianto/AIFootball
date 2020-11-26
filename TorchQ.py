import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class NeuralNetwork(nn.Module):
    def __init__(self, Input, Output, Hidden, Activation, Loss, LearnRate):
        super(NeuralNetwork,self).__init__()
        self.fc1        = nn.Linear(Input, Hidden)
        self.fc2        = nn.Linear(Hidden, Hidden)
        self.fc3        = nn.Linear(Hidden, Hidden)
        self.fc4        = nn.Linear(Hidden, Output)
        self.relu       = Activation
        self.optimizer  = optim.Adam(self.parameters(), LearnRate)
        self.loss       = Loss
        if(torch.cuda.is_available()):
            self.device     = torch.device('cuda')
        else:
            self.device     = 'cpu'
        self.to(self.device)

    def forward(self, input):
        input = self.relu(self.fc1(input))
        input = self.relu(self.fc2(input))
        input = self.relu(self.fc3(input))
        input = self.fc4(input)
        return input

class Agent():
    """
        A Customizable Agent used for Reinforcement Learning.
    """
    def __init__(self, MemSize : int, BatchSize : int, Input : int, 
        Output : int, Gamma : float, EpsStart : float, EpsDecay : float, EpsEnd : float, 
        MaxNN : int, FilePath : str):
        #declare Properties
        self.MemSize        =   MemSize
        self.BatchSize      =   BatchSize
        self.Input          =   Input
        self.Output         =   Output
        self.ActionSpace    =   [i for i in range(self.Output)]
        self.Pos            =   0
    
        #declare Eps
        self.Eps        = EpsStart
        self.EpsStart   = EpsStart
        self.EpsDecay   = EpsDecay
        self.EpsEnd     = EpsEnd
        self.Gamma      = Gamma

        #declare Memory Replay 
        self.StateMem   = np.zeros((self.MemSize, Input), dtype = np.float32)
        self.NextMem    = np.zeros((self.MemSize, Input), dtype = np.float32)
        self.ActionMem  = np.zeros(self.MemSize, dtype = np.int32)
        self.TermMem    = np.zeros(self.MemSize, dtype = bool)
        self.RewMem     = np.zeros(self.MemSize, dtype = np.float32)

        #Declare Networks
        self.PolicyNet  = NeuralNetwork(Input, Output, MaxNN, nn.ReLU(), nn.MSELoss(), 1e-5)
        self.TargetNet  = NeuralNetwork(Input, Output, MaxNN, nn.ReLU(), nn.MSELoss(), 1e-5)
        self.PolicyPath = 'Policy%s.model' % FilePath
        self.TargetPath = 'Target%s.model' % FilePath
        self.TargetNet.eval()
        pass

    def SaveNet(self):
        torch.save(self.PolicyNet.state_dict(), self.PolicyPath)
        torch.save(self.TargetNet.state_dict(), self.TargetPath)
        pass
    
    def LoadNet(self):
        if os.path.isfile(self.PolicyPath):
            self.PolicyNet.load_state_dict(torch.load(self.PolicyPath))
        if os.path.isfile(self.TargetPath):
            self.TargetNet.load_state_dict(torch.load(self.TargetPath))
        pass

    def UpdateNet(self):
        self.TargetNet.load_state_dict(self.PolicyNet.state_dict())
        pass
    
    def SaveState(self, state, reward, action, next, term):
        Index = self.Pos % self.MemSize
        self.StateMem[Index]    = state
        self.RewMem[Index]      = reward
        self.ActionMem[Index]   = action
        self.NextMem[Index]     = next
        self.TermMem[Index]     = term
        self.Pos += 1
        pass
    
    def TestTarget(self, state):
        Input = torch.tensor([state], dtype = torch.float32)
        return self.TargetNet(Input).argmax(dim = 1)
        
    def ChooseAction(self, state):
        if self.Eps < np.random.random():
            Input = torch.tensor([state], dtype = torch.float32).to(self.PolicyNet.device)
            return self.PolicyNet(Input).argmax(dim = 1)
        else:
            return np.random.choice(self.ActionSpace)

    def DecayEpsilon(self):
        self.Eps = self.EpsStart - self.EpsDecay
        if(self.Eps < self.EpsEnd):
            self.Eps = self.EpsEnd
        pass 
    
    def Train(self):
        if(self.Pos < self.BatchSize or self.Pos < 1000):
            return

        MaxIndex    = min(self.Pos, self.MemSize)
        Batch       = np.random.choice(MaxIndex, self.BatchSize, replace = False)
        BatchIndex  = np.arange(self.BatchSize, dtype = np.int32) 

        #Create Batches
        State   = torch.tensor(self.StateMem[Batch]).to(self.PolicyNet.device)
        Next    = torch.tensor(self.NextMem[Batch]).to(self.PolicyNet.device)
        Reward  = torch.tensor(self.RewMem[Batch]).to(self.PolicyNet.device)
        Term    = torch.tensor(self.TermMem[Batch]).to(self.PolicyNet.device)
        Action  = self.ActionMem[Batch]
        
        #Input to Network
        self.PolicyNet.optimizer.zero_grad()
        Policy  = self.PolicyNet(State)[BatchIndex, Action]
        Target  = self.TargetNet(Next)
        Target[Term] = 0.0
        Label   = Reward + self.Gamma*torch.max(Target, dim = 1)[0]
        loss    = self.PolicyNet.loss(Policy, Label)
        loss.backward()
        self.PolicyNet.optimizer.step()
        self.DecayEpsilon()
        return loss
