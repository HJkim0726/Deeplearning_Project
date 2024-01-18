import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import utils
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self,train_dt = torch.tensor):
        self.train_dt = train_dt
    
    def __len__(self):
        return len(self.train_dt)
    
    def __getitem__(self,idx):
        return self.train_dt[idx]
    


class ODE_Solver(nn.Module):
    def __init__(self):
        super(ODE_Solver, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(1,30),
            nn.Tanh(),

            nn.Linear(30,30),
            nn.Tanh(),

            nn.Linear(30,30),
            nn.Tanh(),

            nn.Linear(30,30),
            nn.Tanh(),

            nn.Linear(30,30),
            nn.Tanh(),

            nn.Linear(30,30),
            nn.Tanh(),

            nn.Linear(30,30),
            nn.Tanh(),

            nn.Linear(30,1)
        )

        for layer in self.layer:
            if isinstance(layer, nn.Linear):
                # Apply Xavier uniform initialization to weights
                nn.init.xavier_uniform_(layer.weight)

                # Set biases to 1
                nn.init.constant_(layer.bias, 1)

    def forward(self,x):
        return self.layer(x)
    
    

def loss_function(batch = torch.tensor, initial_point = torch.tensor, col_output = torch.tensor, initial_output = torch.tensor, eq_generator = any, 
                   initial_condition = None, device = "cpu"):

    mse = nn.MSELoss().to(device)
    y = col_output
    y_0 = initial_output
    dy = torch.autograd.grad(outputs = y, inputs = batch, grad_outputs = torch.ones_like(y), create_graph = True)[0]
    
    if eq_generator.order == 2:
        ddy = torch.autograd.grad(outputs = dy, inputs = batch, grad_outputs = torch.ones_like(dy), create_graph = True)[0]
        dy_0 = torch.autograd.grad(outputs = y_0, inputs = initial_point, grad_outputs = torch.ones_like(y_0), create_graph = True)[0]
    
    initial_output = torch.cat((y_0,dy_0), dim = -1).reshape(y_0.shape[0]*2,1).to(device)

    left_eq, right_eq = eq_generator.generate_torch_eq().split("=")
    ode = eval(left_eq) - eval(right_eq)
    ode = ode.to(device)

    if initial_condition == None:
        initial_condition = torch.tensor([*eq_generator.init_condition], dtype = torch.float).unsqueeze(dim = -1).to(device)

    ode_loss = mse(ode, torch.zeros_like(ode).to(device))
    initial_loss = mse(initial_output, initial_condition)

    total_loss = ode_loss + initial_loss

    return total_loss       



class ODE_Calculator(nn.Module):
    
    def __init__(self, ode_point_sampler = any, device = "cpu", stage = None):
        super(ODE_Calculator, self).__init__()          
    
        self.device = device
        self.ode_point_sampler = ode_point_sampler
        self.train_dt = self.ode_point_sampler.gen_train_dt().to(self.device)
        self.test_dt = self.ode_point_sampler.gen_test_dt().to(self.device)

        self.custom_datset = CustomDataset(self.train_dt)
        
        self.ode_solver = ODE_Solver().to(self.device)
        self.loss_function = loss_function
        self.eq_generator = utils.Eq_generator()

        if stage != None: 
            self.stage = stage
            self.aligned_t = self.test_dt
            self.step = self.ode_point_sampler.step
            self.stage = stage

            self.stage_train_dict = {}
            self.stage_initial_point = {}

            for i in range(self.stage):

                if i == self.stage-1:
                    self.stage_train_dict[f"stage{i}"] = self.aligned_t[:][torch.randperm(self.aligned_t[:].shape[0])]
                    break

                self.stage_train_dict[f"stage{i}"] = self.aligned_t[:(i+1)*(self.step//self.stage)][torch.randperm(self.aligned_t[:(i+1)*(self.step//self.stage)].shape[0])]

        print(f"Pytorch expression of your ODE : {self.eq_generator.generate_torch_eq()}")


    def train(self, epoch = int, bs = int, lr = float, optimizer_type = str, display_step = 10):
        
        data_loader = DataLoader(self.custom_datset, batch_size = bs, shuffle = True)
                
        if optimizer_type == "Adam":
            optimizer = optim.Adam(self.ode_solver.parameters(), lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = epoch//2, gamma=0.1)

        elif optimizer_type == "SGD":
            optimizer = optim.SGD(self.ode_solver.parameters(), lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = epoch//2, gamma=0.1)

        for epoch_cnt in range(1,epoch+1):

            display = False
            epoch_loss = 0

            if epoch_cnt % (epoch//display_step) == 0:
                print(f"EPOCH [{epoch_cnt}/{epoch}]")
                display = True

            initial_point = torch.zeros(1,1).requires_grad_(True).to(self.device)

            for batch in data_loader:

                batch.requires_grad_(True).to(self.device)
                col_output = self.ode_solver(batch)          
                initial_output = self.ode_solver(initial_point)

                loss = self.loss_function(batch, initial_point, col_output, initial_output, self.eq_generator, device = self.device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                if display == True:
                    epoch_loss += loss.item()

            scheduler.step()

            if display == True:
                print(f"LOSS = {epoch_loss}, lr = {optimizer.param_groups[0]['lr']}")
    

    def stage_train(self, epoch_per_stage = int, bs = int, lr = float, optimizer_type = str, display_step = 10):

        if optimizer_type == "Adam":
            optimizer = optim.Adam(self.ode_solver.parameters(), lr = lr)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = epoch_per_stage//2, gamma=0.5)

        elif optimizer_type == "SGD":
            optimizer = optim.SGD(self.ode_solver.parameters(), lr = lr)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = epoch_per_stage//2, gamma=0.5)
                
        initial_point = torch.zeros((1,1), dtype = torch.float, requires_grad = True).to(self.device)
        initial_condition = torch.tensor([*self.eq_generator.init_condition], dtype = torch.float).unsqueeze(dim = -1).to(self.device)

        for stage in range(self.stage):
            print(f"Stage{stage+1}/{self.stage} start")
            train_dt = self.stage_train_dict[f"stage{stage}"].to(self.device)
            custom_dataset = CustomDataset(train_dt)
            data_loader = DataLoader(custom_dataset, batch_size = bs, shuffle = True)

            for epoch_cnt in range(1,epoch_per_stage+1):
                display = False
                epoch_loss = 0

                if epoch_cnt % (epoch_per_stage//display_step) == 0:
                    print(f"EPOCH [{epoch_cnt}/{epoch_per_stage}]")
                    display = True

                for batch in data_loader:

                    batch.requires_grad_(True)
                    col_output = self.ode_solver(batch)          
                    initial_output = self.ode_solver(initial_point)

                    loss = self.loss_function(batch, initial_point, col_output, initial_output, self.eq_generator, initial_condition, self.device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if display == True:
                        epoch_loss += loss.item()

                if display == True:
                    print(f"LOSS = {epoch_loss}")

                # scheduler.step()
            # optimizer.param_groups[0]['lr'] /= 2
            print("\n")
            
            initial_point = torch.cat((initial_point, torch.tensor([torch.max(train_dt)]).unsqueeze(dim = -1).to(self.device)), dim = 0).detach().requires_grad_(True)
            initial_condition = torch.cat((initial_condition, self.ode_solver(initial_point).to(self.device)), dim = 0).detach().requires_grad_(True)



    def gen_graph(self,exact = None):

        predict = self.ode_solver(self.test_dt).to('cpu').detach()

        fig, ax = plt.subplots()

        ax.set_xlabel("t (intput)")
        ax.set_ylabel("y (output)")
        ax.plot(self.test_dt.to("cpu"), predict, label = "predict", c = 'b', linestyle = 'dashed')
        ax.set_title("Predicted answer of ODE")
        ax.grid()

        if exact != None:
            ax.plot(self.test_dt, exact, label = "exact", c = 'r', alpha = 0.5)

        ax.legend()
        plt.show()

    
    def gen_value(self, observe_t = float):
        self.ode_solver.to("cpu")

        obsv_point = torch.tensor([observe_t]).unsqueeze(dim = -1)
        obsv_output = self.ode_solver(obsv_point)

        print(f"Expected value at t = {observe_t} : {obsv_output.item()}" )
