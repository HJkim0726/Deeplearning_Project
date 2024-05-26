import torch
import torch.nn as nn
import matplotlib.pyplot as plt


###############  point_sampler ###############
# Makes the collocation points tensor (randomly scattered points)
# The dimension, range of points, boundary condition points are determined by user input
# Plots the sampled points
# Result is dictionaries of tensors
# Three dictionaries : 1) For Train, 2) For Test 3) For additional method(Training)
# Dictionary keys : 'collocation', 'initial', 'boundary'(if input of boundary condition is not 0)
def point_sampler():
  points_dict = {}
  points_dict_cha = {}
  dim = int(input('PDE Dimension(excluding t) : '))
  col_num = int(input('Number of collocation points : '))

  if dim == 1:
    x_min = float(input('x_min : '))
    x_max = float(input('x_max : '))
    t_max = float(input('t_max : '))

    col_x = torch.linspace(x_min, x_max, col_num)[torch.randperm(col_num)].unsqueeze(dim = -1)
    col_t = torch.linspace(0,t_max, col_num)[torch.randperm(col_num)].unsqueeze(dim = -1)

    bc_num = int(input('Number of boundary conditions : '))
    bc_points = []

    for i in range(bc_num):
      bc_points.append(float(input(f'Boundary position{i+1} (x) : ')))

    bc_x = torch.zeros((col_num//10 * bc_num, 1))
    indice = bc_x.shape[0] // bc_num
    bc_t = torch.linspace(0,t_max,indice).unsqueeze(dim = -1).repeat(bc_num,1)

    for i, bc_point in enumerate(bc_points):
      bc_x[indice*i: indice * (i+1)] = torch.tensor([bc_point]).unsqueeze(dim = -1).repeat(indice,1)


    ic_x = torch.linspace(x_min,x_max,col_num//10).unsqueeze(dim = -1)
    ic_t = torch.zeros_like(ic_x)

    plt.scatter(col_x, col_t, c = 'k', s = 0.5)
    plt.scatter(bc_x, bc_t, c = 'r', s = 1)
    plt.scatter(ic_x, ic_t, c = 'g', s = 1)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend(['collocation points','boundary points','initial points'])
    plt.title('Sampled points')
    plt.show()

    col_points = torch.cat((col_x, col_t), dim = -1)
    boundary_points = torch.cat((bc_x, bc_t), dim = -1)
    initial_points = torch.cat((ic_x,ic_t), dim = -1)

    col_points = torch.cat((col_points, boundary_points, initial_points), dim = 0)

    print(f'Total sampled points : {col_points.shape[0]}')
    points_dict['collocation'] = col_points
    points_dict['boundary'] = boundary_points
    points_dict['initial'] = initial_points

    return points_dict

  if dim == 2:
    x_min = float(input('x_min : '))
    x_max = float(input('x_max : '))
    test_x_num = int(input('test_x_grid_num : '))
    y_min = float(input('y_min : '))
    y_max = float(input('y_max : '))
    test_y_num = int(input('test_y_grid_num : '))
    t_min = 0
    t_max = float(input('t_max : '))
    test_t_num = int(input('test_t_grid_num : '))
    cha_num = int(input('cha_num: '))

    info = [[x_min,x_max,test_x_num],[y_min,y_max,test_y_num],[t_min,t_max,test_t_num]]
    x_grad = torch.linspace(info[0][0],info[0][1], info[0][2]).unsqueeze(dim = -1) # Decreasing sampling points over epochs
    y_grad = torch.linspace(info[1][0],info[1][1], info[1][2]).unsqueeze(dim = -1)
    t_grad = torch.linspace(info[2][0],info[2][1], info[2][2]).unsqueeze(dim = -1)

    col_x_test = x_grad.repeat(info[1][2]*info[2][2],1)
    col_y_test = y_grad.repeat(1,info[0][2]).view(-1).unsqueeze(dim = -1).repeat(info[2][2],1)
    col_t_test = t_grad.repeat(1,info[0][2]*info[1][2]).view(-1).unsqueeze(dim = -1)

    col_test = torch.cat((col_x_test, col_y_test, col_t_test), dim = -1)
    col_test_re = col_test.unsqueeze(dim = -1).view(info[2][2],info[1][2],info[0][2],3)
    bc_x_test_1 = col_test_re[:,:,0,:].reshape(-1,3)
    bc_x_test_2 = col_test_re[:,:,-1,:].reshape(-1,3)
    bc_y_test_1 = col_test_re[:,0,:,:].reshape(-1,3)
    bc_y_test_2 = col_test_re[:,-1,:,:].reshape(-1,3)
    bc_t_test_1 = col_test_re[0,:,:,:].reshape(-1,3)
    bc_t_test_2 = col_test_re[-1,:,:,:].reshape(-1,3)
    ini_test = col_test[:info[0][2]*info[1][2]]

    test_dict = dict()
    test_dict['initial'] = ini_test
    test_dict['collocation'] = col_test
    test_dict['t_num'] = info[2][2]

    col_x = torch.linspace(x_min, x_max, col_num)[torch.randperm(col_num)].unsqueeze(dim = -1)
    col_y = torch.linspace(y_min, y_max, col_num)[torch.randperm(col_num)].unsqueeze(dim = -1)
    col_t = torch.linspace(0,t_max, col_num)[torch.randperm(col_num)].unsqueeze(dim = -1)

    col_x_cha = torch.linspace(x_min, x_max, cha_num)[torch.randperm(cha_num)].unsqueeze(dim = -1).repeat(1,50).view(-1).unsqueeze(dim = -1)
    col_y_cha = torch.linspace(y_min, y_max, cha_num)[torch.randperm(cha_num)].unsqueeze(dim = -1).repeat(1,50).view(-1).unsqueeze(dim = -1)
    col_t_cha = torch.linspace(0,t_max, cha_num)[torch.randperm(cha_num)].unsqueeze(dim = -1).repeat(1,50).view(-1).unsqueeze(dim = -1)



    bc_num = int(input('Number of boundary conditions: '))
    bc_points = []


    for i in range(bc_num):
      print(f'Boundary {i+1}')
      x_or_y = input('Boundary for x or y? (x/y) : ')
      value = float(input(f'Boundary{i+1} : {x_or_y} = '))
      bc_points.append((x_or_y,value))

    ratio = int(input('boundary/initial points rato (%) : '))/100


    if bc_num != 0:

      test_bc_dict = {}


      test_bc_dict['x_1']=bc_x_test_1
      test_bc_dict['x_2']=bc_x_test_2
      test_bc_dict['y_1']=bc_y_test_1
      test_bc_dict['y_2']=bc_y_test_2
      test_bc_dict['t_1']=bc_t_test_1
      test_bc_dict['t_2']=bc_t_test_2

      keys = [*(test_bc_dict.keys())]
      aligned_boundary_points = test_bc_dict[keys[0]]
      for i in range(1,len(test_bc_dict)):
          aligned_boundary_points = torch.cat((aligned_boundary_points, test_bc_dict[keys[i]]), dim = 0)
      test_dict['boundary'] = aligned_boundary_points
    
    if bc_num != 0:

      bc_col_x = torch.linspace(x_min,x_max,int(col_num*ratio))[torch.randperm(int(col_num*ratio))].unsqueeze(dim = -1)
      bc_col_y = torch.linspace(y_min,y_max,int(col_num*ratio))[torch.randperm(int(col_num*ratio))].unsqueeze(dim = -1)
      bc_col_t = torch.linspace(0,t_max,int(col_num*ratio // bc_num))[torch.randperm(int(col_num*ratio//bc_num))].unsqueeze(dim = -1).repeat(bc_num,1)

      bc_dict = {}

      for i, (axis, value) in enumerate(bc_points):
        if axis == 'x':
          bc_x = torch.tensor([value]).unsqueeze(dim = -1).repeat(int(col_num*ratio),1)
          bc_dict[f'boundary{i+1}'] = torch.cat((bc_x, bc_col_y, bc_col_t), dim = -1)
        if axis == 'y':
          bc_y = torch.tensor([value]).unsqueeze(dim = -1).repeat(int(col_num*ratio),1)
          bc_dict[f'boundary{i+1}'] = torch.cat((bc_col_x, bc_y, bc_col_t), dim = -1)

      print(bc_dict['boundary1'].shape)
    ic_x = torch.linspace(x_min,x_max,int(col_num*ratio)).unsqueeze(dim = -1)[torch.randperm(int(col_num*ratio))]
    ic_y = torch.linspace(y_min,y_max,int(col_num*ratio)).unsqueeze(dim = -1)[torch.randperm(int(col_num*ratio))]
    ic_t = torch.zeros_like(ic_x)

    ic_x_cha = torch.linspace(x_min,x_max,int(cha_num)).unsqueeze(dim = -1)[torch.randperm(int(cha_num))].repeat(1,50).view(-1).unsqueeze(dim = -1)
    ic_y_cha = torch.linspace(y_min,y_max,int(cha_num)).unsqueeze(dim = -1)[torch.randperm(int(cha_num))].repeat(1,50).view(-1).unsqueeze(dim = -1)
    ic_t_cha = torch.zeros_like(ic_x_cha)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(col_x,col_y,col_t, c = 'k', s = 0.3)
    if bc_num != 0:
      for i in range(len(bc_dict)):
        ax.scatter(bc_dict[f'boundary{i+1}'][:,0],bc_dict[f'boundary{i+1}'][:,1],bc_dict[f'boundary{i+1}'][:,2] ,c = 'r', s = 0.3, alpha = 0.2)
    ax.scatter(ic_x, ic_y, ic_t, c = 'g', s = 0.8)
    # Set labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t-axis')
    ax.set_title('Sampled points')

    col_points = torch.cat((col_x, col_y, col_t), dim = -1)
    col_points_cha = torch.cat((col_x_cha, col_y_cha, col_t_cha), dim = -1)
    initial_points = torch.cat((ic_x, ic_y, ic_t), dim = -1)
    initial_points_cha = torch.cat((ic_x_cha, ic_y_cha, ic_t_cha), dim = -1)

    points_dict['initial'] = initial_points
    points_dict_cha['initial']=initial_points_cha

    aligned_boundary_points = None

    if bc_num != 0:
      keys = [*(bc_dict.keys())]
      aligned_boundary_points = bc_dict[keys[0]]
      for i in range(1,len(bc_dict)):
        aligned_boundary_points = torch.cat((aligned_boundary_points, bc_dict[keys[i]]), dim = 0)
      points_dict['boundary'] = aligned_boundary_points
      
      col_points = torch.cat((col_points, aligned_boundary_points, initial_points), dim = 0)
    else : 
      col_points = torch.cat((col_points, initial_points), dim = 0)
    print(f'Total sampled points : {col_points.shape[0]}')
    points_dict['collocation'] = col_points
    points_dict_cha['collocation'] = col_points_cha


    # Show the plot
    plt.show()

    return points_dict, test_dict, points_dict_cha


############### visualize_3d ###############
# Simply plots scattered points in 3d
# Input : (x,y,t) tensor
def visualize_3d(tensor):
  tensor = tensor.to('cpu')
  x = tensor[:,0]
  y = tensor[:,1]
  t = tensor[:,2]

  fig = plt.figure()
  ax = fig.add_subplot(111, projection = '3d')
  ax.scatter(x,y,t,c = 'g', s = 0.5)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('t')
  ax.set_title('New sampled points')

  plt.show()


############### validate_visualization ###############
# Validate the Neural Network model by plotting
# Makes collocation points of given x,y,t range
# Plot the result of model and analytical solution(if exists) in same figure
# Repeatidly shows graph at each observation time
def validate_visualization(model, device, condition = None, observe_t = list, x_range = list, y_range = None, axis_range_ratio = 1.1):
  with torch.no_grad():
    model = model.to('cpu')
    valid_points_number = 10000
    x = torch.linspace(x_range[0], x_range[-1], valid_points_number)[torch.randperm(valid_points_number)].unsqueeze(dim = -1)
    y = torch.linspace(y_range[0], y_range[-1], valid_points_number)[torch.randperm(valid_points_number)].unsqueeze(dim = -1)

    for time in observe_t:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection = '3d')

      time_point = torch.tensor([time]).unsqueeze(dim = -1).repeat(x.shape[0],1)
      batch = torch.cat((x,y,time_point), dim = -1)

      out = model(batch).detach()

      if condition != None:
        truth = condition.analytic_solution(batch)
        truth_u = truth[:,0]
        ax.set_zlim(torch.min(truth_u)*(2-axis_range_ratio),torch.max(truth_u)*axis_range_ratio)
        ax.scatter(x,y,out[:,0], c = 'r', s = 0.5)
        ax.scatter(x,y,truth_u, c = 'b', s = 2, alpha = 0.2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title(f't = {time}')
        ax.legend(['predict','analytical solution'])
      else : 
        ax.set_zlim(torch.min(out[:,0])*(2-axis_range_ratio),torch.max(out[:,0])*axis_range_ratio)
        ax.scatter(x,y,out[:,0], c = 'r', s = 0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title(f't = {time}')
        ax.legend(['predict'])

      plt.show()
  model = model.to(device)



############### test ###############
# Points from test_dict which is from point_sampler are grid points (not randomly scattered points of given range)
# Returns loss value by using loss_fn function which is defined in main.ipynb
def test(test_dict, model,loss_ratio, loss_fn, condition, device):
  model.eval()
  _,_,_, total_loss = loss_fn(test_dict['collocation'], model, test_dict, condition, device, loss_ratio)
  return total_loss


############### real_test ###############
# Returns MSE between model predicted value and result of analytical solution
# Works only if analytical solution exists
def real_test(test_dict, model, loss_ratio, loss_fn, condition, device):
  model.eval()
  mse = nn.MSELoss().to(device)
  result = condition.analytic_solution(test_dict['collocation']).to(device)
  predict = model(test_dict['collocation'].to(device))
  col_error = mse(result, predict)

  _,initial_error,_,_ = loss_fn(test_dict['collocation'], model, test_dict, condition, device, loss_ratio)
  total_error = col_error + initial_error
  print(col_error)
  print(initial_error)
  return total_error

