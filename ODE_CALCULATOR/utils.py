import torch
import re

class ODE_Point_sampler():
    def __init__(self, t_min = float, t_max = float, step = int):
        self.t_min = t_min
        self.t_max = t_max
        self.step = step

    def gen_train_dt(self):
        train_dt = torch.linspace(self.t_min, self.t_max, self.step)[torch.randperm(self.step)].unsqueeze(dim = -1) # (step,1)
        return train_dt
    
    def gen_test_dt(self):
        test_dt = torch.linspace(self.t_min, self.t_max, self.step).unsqueeze(dim = -1) # (step,1)
        return test_dt
    

class Eq_generator():
    def __init__(self):
        self.order = int(input("Order of your ODE? : "))
        print("Format rules")
        print("1. No space")
        print("2. t is variable of y")
        print("3. Convert ^ -> ** except for e^(t)")
        print("4. Always contain '()' small brackets when using functions. Ex) sint -> sin(t), e^(t)")
        print("Input example : y''-2*y**2=sin(2*t)+e^(-2*t)")
        self.expr = input("Input your ODE : ")

        self.init_condition = []
        
        for i in range(self.order):
            self.init_condition.append(float(input(f"Initial condition {i+1}, y{'`'*i}(0) = ")))


    def replace_funcs(self,expr):

        # Replace 'e^...' expressions
        expr = re.sub(r'e\^([^\s]+)', lambda x: 'torch.exp' + x.group(1).replace('t', 'batch'), expr)

        # Replace standalone 't' with 'batch'
        expr = re.sub(r'(?<!\w)t(?!\w)', 'batch', expr)

        functions = ['sin', 'cos', 'tan', 'log']  # Add more functions as needed
        for func in functions:
            expr = re.sub(r'(?<!\w)' + func + r'(?=\()', 'torch.' + func, expr)

        return expr
    
    def replace_primes(self,expr):
        def replacement(match):
            var_name = match.group(1)  # Variable name (e.g., 'y')
            num_primes = len(match.group(2))  # Number of primes
            return 'd' * num_primes + var_name  # Construct replacement string

        # Replace patterns like "y'", "y''", "y'''" with "dy", "ddy", "dddy"
        return re.sub(r"(\w)('*)", replacement, expr)

    
    def generate_torch_eq(self):
        expr = self.replace_funcs(self.expr)
        expr = self.replace_primes(expr)
        return expr
        
        
        









