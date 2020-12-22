import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import Laplace3d_1Activation

R={}
# -------------------------------------- CPU or GPU 选择 -----------------------------------------------
R['gpuNo'] = 0
# 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1代表使用 CPU 模式
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为第 0 块GPU, 设备名称为'/gpu:0'
if platform.system() == 'Windows':
    os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
else:
    print('-------------------------------------- linux -----------------------------------------------')
    # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
    matplotlib.use('Agg')

    if tf.test.is_gpu_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 文件保存路径设置
store_file = 'laplace3d'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
OUT_DIR = os.path.join(BASE_DIR, store_file)
if not os.path.exists(OUT_DIR):
    print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
    os.mkdir(OUT_DIR)

R['seed'] = np.random.randint(1e5)
seed_str = str(R['seed'])                     # int 型转为字符串型
FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
R['FolderName'] = FolderName
if not os.path.exists(FolderName):
    print('--------------------- FolderName -----------------:', FolderName)
    os.mkdir(FolderName)

# ----------------------------------------  复制并保存当前文件 -----------------------------------------
if platform.system() == 'Windows':
    tf.compat.v1.reset_default_graph()
    shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
else:
    shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

# if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
R['activate_stop'] = int(step_stop_flag)
# if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
R['max_epoch'] = 200000
if 0 != R['activate_stop']:
    epoch_stop = input('please input a stop epoch:')
    R['max_epoch'] = int(epoch_stop)

# ---------------------------- Setup of multi-scale problem-------------------------------
R['input_dim'] = 3  # 输入维数，即问题的维数(几元问题)
R['output_dim'] = 1  # 输出维数

# R['laplace_opt'] = 'general_laplace'
# R['equa_name'] = 'PDE1'
# R['equa_name'] = 'PDE2'
# R['equa_name'] = 'PDE3'
# R['equa_name'] = 'PDE4'
# R['equa_name'] = 'PDE5'
# R['equa_name'] = 'PDE6'
# R['equa_name'] = 'PDE7'

R['laplace_opt'] = 'p_laplace2multi_scale'
# R['equa_name'] = 'multi_scale3D_1'
R['equa_name'] = 'multi_scale3D_2'

if R['laplace_opt'] == 'general_laplace':
    R['mesh_number'] = 1
    R['epsilon'] = 0.1
    R['order2laplace'] = 2
    R['batch_size2interior'] = 5000  # 内部训练数据的批大小
    R['batch_size2boundary'] = 1000
elif R['laplace_opt'] == 'p_laplace2multi_scale':
    R['mesh_number'] = 1
    R['epsilon'] = 0.1
    R['order2laplace'] = 2
    R['batch_size2interior'] = 5000  # 内部训练数据的批大小
    R['batch_size2boundary'] = 1000

# ---------------------------- Setup of DNN -------------------------------
# 装载测试数据模式和画图模式
R['hot_power'] = 1
R['testData_model'] = 'loadData'

R['variational_loss'] = 1  # PDE变分

R['optimizer_name'] = 'Adam'  # 优化器
R['learning_rate'] = 2e-4  # 学习率
R['learning_rate_decay'] = 5e-5  # 学习率 decay
R['train_group'] = 0

R['weight_biases_model'] = 'general_model'              # 权重和偏置生成模式
# 正则化权重和偏置的模式
R['regular_weight_model'] = 'L0'
# R['regular_weight_model'] = 'L1'
# R['regular_weight_model'] = 'L2'
R['regular_weight_biases'] = 0.000                      # Regularization parameter for weights
# R['regular_weight_biases'] = 0.001                    # Regularization parameter for weights
# R['regular_weight_biases'] = 0.0025                   # Regularization parameter for weights

# 边界的惩罚处理方式,以及边界的惩罚因子
R['activate_penalty2bd_increase'] = 1
# R['init_boundary_penalty'] = 1000  # Regularization parameter for boundary conditions
R['init_boundary_penalty'] = 100  # Regularization parameter for boundary conditions

# 网络的频率范围设置
R['freqs'] = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)

# &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
# R['hidden_layers'] = (100, 10, 8, 6, 4)  # 测试
# R['hidden_layers'] = (100, 80, 60, 60, 40, 40, 20)
# R['hidden_layers'] = (200, 100, 80, 50, 30)
# R['hidden_layers'] = (300, 200, 150, 100, 100, 50, 50)
R['hidden_layers'] = (500, 400, 400, 200, 200)                                     # 待选择的网络规模
# R['hidden_layers'] = (500, 400, 300, 300, 200, 100)
# R['hidden_layers'] = (500, 400, 300, 200, 200, 100)
# R['hidden_layers'] = (500, 400, 300, 300, 200, 100, 100)
# R['hidden_layers'] = (500, 300, 200, 200, 100, 100, 50)
# R['hidden_layers'] = (1000, 800, 600, 400, 200)
# R['hidden_layers'] = (1000, 500, 400, 300, 300, 200, 100, 100)
# R['hidden_layers'] = (2000, 1500, 1000, 500, 250)

# &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
# R['model'] = 'PDE_DNN'
# R['model'] = 'PDE_DNN_BN'
# R['model'] = 'PDE_DNN_scale'
# R['model'] = 'PDE_DNN_adapt_scale'
R['model'] = 'PDE_DNN_FourierBase'

# &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# R['activate_func'] = 'relu'
# R['activate_func'] = 'tanh'
# R['activate_func'] = 'sintanh'
# R['activate_func']' = leaky_relu'
# R['activate_func'] = 'srelu'
R['activate_func'] = 's2relu'
# R['activate_func'] = 's3relu'
# R['activate_func'] = 'csrelu'
# R['activate_func'] = 'gauss'
# R['activate_func'] = 'singauss'
# R['activate_func'] = 'metican'
# R['activate_func'] = 'modify_mexican'
# R['activate_func'] = 'leaklysrelu'
# R['activate_func'] = 'slrelu'
# R['activate_func'] = 'elu'
# R['activate_func'] = 'selu'
# R['activate_func'] = 'phi'
# R['activate_func'] = 'sin_modify_mexican'

Laplace3d_1Activation.solve_laplace(R)
