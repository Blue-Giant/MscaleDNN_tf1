"""
@author: LXA
 Date: 2020 年 5 月 31 日
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import time
import DNN_base
import DNN_tools
import DNN_data
import MS_LaplaceEqs
import MS_BoltzmannEqs
import MS_ConvectionEqs
import General_Laplace
import matData2Laplace
import matData2pLaplace
import matData2Boltzmann
import saveData
import plotData
import DNN_Log_Print


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)

    if R_dic['PDE_type'] != 'general_Laplace':
        DNN_tools.log_string('epsilon: %f\n' % (R_dic['epsilon']), log_fileout)
    if R_dic['PDE_type'] == 'pLaplace' or R_dic['PDE_type'] == 'pLaplace_implicit' or \
            R_dic['PDE_type'] == 'pLaplace_implicit':
        DNN_tools.log_string('The order to pLaplace: %s\n' % (R_dic['order2pLaplace_operator']), log_fileout)

    if R_dic['PDE_type'] == 'pLaplace_implicit':
        DNN_tools.log_string('The mesh_number: %f\n' % (R_dic['mesh_number']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    if R_dic['loss_type'] == 'variational_loss':
        DNN_tools.log_string('Loss function: variational loss\n', log_fileout)
    elif R_dic['loss_type'] == 'lncosh_loss2Ritz':
        DNN_tools.log_string('Loss function: lncosh_ritz loss\n', log_fileout)
        DNN_tools.log_string('The penalty for  lncosh_ritz:%s \n' % str(R_dic['lambda2lncosh']), log_fileout)
    else:
        DNN_tools.log_string('Loss function: L2 loss\n', log_fileout)

    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    if R_dic['model'] != 'DNN':
        DNN_tools.log_string('The frequency flags to Network: %s\n' % (R_dic['freqs']), log_fileout)
    DNN_tools.log_string('Activate function for network: %s\n' % str(R_dic['activate_func']), log_fileout)

    if R_dic['model'] == 'DNN_FourierBase' and R_dic['activate_func'] == 'tanh':
        DNN_tools.log_string('The scale-factor to fourier basis: %s\n' % (R_dic['sfourier']), log_fileout)

    DNN_tools.log_string('hidden layer:%s\n' % str(R_dic['hidden_layers']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    DNN_tools.log_string('Batch-size 2 interior: %s\n' % str(R_dic['batch_size2interior']), log_fileout)
    DNN_tools.log_string('Batch-size 2 boundary: %s\n' % str(R_dic['batch_size2boundary']), log_fileout)

    DNN_tools.log_string('Initial boundary penalty: %s\n' % str(R_dic['init_boundary_penalty']), log_fileout)
    if R_dic['activate_penalty2bd_increase'] == 1:
        DNN_tools.log_string('The penalty of boundary will increase with training going on.\n', log_fileout)
    elif R_dic['activate_penalty2bd_increase'] == 2:
        DNN_tools.log_string('The penalty of boundary will decrease with training going on.\n', log_fileout)
    else:
        DNN_tools.log_string('The penalty of boundary will keep unchanged with training going on.\n', log_fileout)


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['activate_func'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    # 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['init_boundary_penalty']           # Regularization parameter for boundary conditions
    penalty2WB = R['penalty2weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    hidden_layers = R['hidden_layers']
    act_func = R['activate_func']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    lambda2lncosh = R['lambda2lncosh']

    # pLaplace 算子需要的额外设置, 先预设一下
    p_index = 2
    epsilon = 0.1
    mesh_number = 2
    region_lb = 0.0
    region_rt = 1.0

    if R['PDE_type'] == 'general_Laplace':
        # -laplace u = f
        region_lb = 0.0
        region_rt = 1.0
        f, u_true, u_left, u_right, u_bottom, u_top = General_Laplace.get_infos2Laplace_2D(
            input_dim=input_dim, out_dim=out_dim, left_bottom=region_lb, right_top=region_rt, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace_implicit':
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        if R['equa_name'] == 'multi_scale2D_5':
            region_lb = 0.0
            region_rt = 1.0
        else:
            region_lb = -1.0
            region_rt = 1.0
        u_true, f, A_eps, u_left, u_right, u_bottom, u_top = MS_LaplaceEqs.get_infos2pLaplace_2D(
            input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], intervalL=0.0, intervalR=1.0,
            equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace_explicit':
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        if R['equa_name'] == 'multi_scale2D_7':
            region_lb = 0.0
            region_rt = 1.0
            u_true = MS_LaplaceEqs.true_solution2E7(input_dim, out_dim, eps=epsilon)
            u_left, u_right, u_bottom, u_top = MS_LaplaceEqs.boundary2E7(
                input_dim, out_dim, region_lb, region_rt, eps=epsilon)
            A_eps = MS_LaplaceEqs.elliptic_coef2E7(input_dim, out_dim, eps=epsilon)
    elif R['PDE_type'] == 'Possion_Boltzmann':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) | + Ku_eps =f(x), x \in R^n
        #       dx     ****         dx        ****
        # region_lb = -1.0
        region_lb = 0.0
        region_rt = 1.0
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        A_eps, kappa, u_true, u_left, u_right, u_top, u_bottom, f = MS_BoltzmannEqs.get_infos2Boltzmann_2D(
            equa_name=R['equa_name'], intervalL=region_lb, intervalR=region_rt)
    elif R['PDE_type'] == 'Convection_diffusion':
        region_lb = -1.0
        region_rt = 1.0
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        A_eps, Bx, By, u_true, u_left, u_right, u_top, u_bottom, f = MS_ConvectionEqs.get_infos2Convection_2D(
            equa_name=R['equa_name'], eps=epsilon, region_lb=0.0, region_rt=1.0)

    # 初始化权重和和偏置
    flag1 = 'WB'
    if R['model'] == 'DNN_FourierBase' or R['model'] == 'DNN_FourierBase2':
        W2NN, B2NN = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_layers, flag1)
    else:
        W2NN, B2NN = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_layers, flag1)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XY_it = tf.placeholder(tf.float32, name='X_it', shape=[None, input_dim])                # * 行 2 列
            XY_left_bd = tf.placeholder(tf.float32, name='X_left_bd', shape=[None, input_dim])      # * 行 2 列
            XY_right_bd = tf.placeholder(tf.float32, name='X_right_bd', shape=[None, input_dim])    # * 行 2 列
            XY_bottom_bd = tf.placeholder(tf.float32, name='Y_bottom_bd', shape=[None, input_dim])  # * 行 2 列
            XY_top_bd = tf.placeholder(tf.float32, name='Y_top_bd', shape=[None, input_dim])        # * 行 2 列
            boundary_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')

            # 供选择的网络模式
            if R['model'] == 'DNN':
                UNN = DNN_base.DNN(XY_it, W2NN, B2NN, hidden_layers, activate_name=act_func)
                UNN_left = DNN_base.DNN(XY_left_bd, W2NN, B2NN, hidden_layers, activate_name=act_func)
                UNN_right = DNN_base.DNN(XY_right_bd, W2NN, B2NN, hidden_layers, activate_name=act_func)
                UNN_bottom = DNN_base.DNN(XY_bottom_bd, W2NN, B2NN, hidden_layers, activate_name=act_func)
                UNN_top = DNN_base.DNN(XY_top_bd, W2NN, B2NN, hidden_layers, activate_name=act_func)
            elif R['model'] == 'DNN_scale':
                freq = R['freqs']
                UNN = DNN_base.DNN_scale(XY_it, W2NN, B2NN, hidden_layers, freq, activate_name=act_func)
                UNN_left = DNN_base.DNN_scale(XY_left_bd, W2NN, B2NN, hidden_layers, freq, activate_name=act_func)
                UNN_right = DNN_base.DNN_scale(XY_right_bd, W2NN, B2NN, hidden_layers, freq, activate_name=act_func)
                UNN_bottom = DNN_base.DNN_scale(XY_bottom_bd, W2NN, B2NN, hidden_layers, freq, activate_name=act_func)
                UNN_top = DNN_base.DNN_scale(XY_top_bd, W2NN, B2NN, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'DNN_adapt_scale':
                freqs = R['freqs']
                UNN = DNN_base.DNN_adapt_scale(XY_it, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                UNN_left = DNN_base.DNN_adapt_scale(XY_left_bd, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                UNN_right = DNN_base.DNN_adapt_scale(XY_right_bd, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                UNN_bottom = DNN_base.DNN_adapt_scale(XY_bottom_bd, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                UNN_top = DNN_base.DNN_adapt_scale(XY_top_bd, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
            elif R['model'] == 'DNN_FourierBase':
                freqs = R['freqs']
                UNN = DNN_base.DNN_FourierBase(XY_it, W2NN, B2NN, hidden_layers, freqs,
                                               activate_name=act_func, sFourier=R['sfourier'])
                UNN_left = DNN_base.DNN_FourierBase(XY_left_bd, W2NN, B2NN, hidden_layers, freqs,
                                                    activate_name=act_func, sFourier=R['sfourier'])
                UNN_right = DNN_base.DNN_FourierBase(XY_right_bd, W2NN, B2NN, hidden_layers, freqs,
                                                     activate_name=act_func, sFourier=R['sfourier'])
                UNN_bottom = DNN_base.DNN_FourierBase(XY_bottom_bd, W2NN, B2NN, hidden_layers, freqs,
                                                      activate_name=act_func, sFourier=R['sfourier'])
                UNN_top = DNN_base.DNN_FourierBase(XY_top_bd, W2NN, B2NN, hidden_layers, freqs,
                                                   activate_name=act_func, sFourier=R['sfourier'])
            elif R['model'] == 'DNN_FourierBase2':
                freqs = R['freqs']
                freqs2 = np.arange(1, 26)
                UNN = DNN_base.DNN_FourierBase2(XY_it, W2NN, B2NN, hidden_layers, freqs, freqs2, activate_name=act_func)
                UNN_left = DNN_base.DNN_FourierBase2(XY_left_bd, W2NN, B2NN, hidden_layers, freqs, freqs2, activate_name=act_func)
                UNN_right = DNN_base.DNN_FourierBase2(XY_right_bd, W2NN, B2NN, hidden_layers, freqs, freqs2, activate_name=act_func)
                UNN_bottom = DNN_base.DNN_FourierBase2(XY_bottom_bd, W2NN, B2NN, hidden_layers, freqs, freqs2, activate_name=act_func)
                UNN_top = DNN_base.DNN_FourierBase2(XY_top_bd, W2NN, B2NN, hidden_layers, freqs, freqs2, activate_name=act_func)
            elif R['model'] == 'DNN_Sin+Cos_Base':
                freqs = R['freqs']
                UNN = DNN_base.DNN_SinAddCos(XY_it, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                UNN_left = DNN_base.DNN_SinAddCos(XY_left_bd, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                UNN_right = DNN_base.DNN_SinAddCos(XY_right_bd, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                UNN_bottom = DNN_base.DNN_SinAddCos(XY_bottom_bd, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                UNN_top = DNN_base.DNN_SinAddCos(XY_top_bd, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)

            X_it = tf.reshape(XY_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XY_it[:, 1], shape=[-1, 1])
            # 变分形式的loss of interior，训练得到的 UNN 是 * 行 1 列
            if R['loss_type'] == 'variational_loss':
                dUNN = tf.gradients(UNN, XY_it)[0]                                                       # * 行 2 列
                dUNN_Norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])  # 按行求和
                if R['PDE_type'] == 'general_Laplace':
                    dUNN_2Norm = tf.square(dUNN_Norm)
                    loss_it_variational = (1.0 / 2) * dUNN_2Norm - \
                                          tf.multiply(tf.reshape(f(X_it, Y_it), shape=[-1, 1]), UNN)
                elif R['PDE_type'] == 'pLaplace_explicit' or R['PDE_type'] == 'pLaplace_implicit':
                    a_eps = A_eps(X_it, Y_it)                                     # * 行 1 列
                    AdUNN_pNorm = tf.multiply(a_eps, tf.pow(dUNN_Norm, p_index))
                    if R['equa_name'] == 'multi_scale2D_7':
                        fxy = MS_LaplaceEqs.get_force_side2MS_E7(x=X_it, y=Y_it)
                        loss_it_variational = (1.0 / p_index) * AdUNN_pNorm - \
                                              tf.multiply(tf.reshape(fxy, shape=[-1, 1]), UNN)
                    else:
                        loss_it_variational = (1.0 / p_index) * AdUNN_pNorm - \
                                              tf.multiply(tf.reshape(f(X_it, Y_it), shape=[-1, 1]), UNN)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it, Y_it)                          # * 行 1 列
                    Kappa = kappa(X_it, Y_it)                          # * 行 1 列
                    AdUNN_pNorm = a_eps * tf.pow(dUNN_Norm, p_index)   # * 行 1 列
                    if R['equa_name'] == 'Boltzmann3' or R['equa_name'] == 'Boltzmann4' or\
                            R['equa_name'] == 'Boltzmann5' or R['equa_name'] == 'Boltzmann6':
                        fxy = MS_BoltzmannEqs.get_foreside2Boltzmann2D(x=X_it, y=Y_it)
                        loss_it_variational = (1.0 / p_index) * (AdUNN_pNorm + Kappa*UNN*UNN) - \
                                          tf.multiply(tf.reshape(fxy, shape=[-1, 1]), UNN)
                    else:
                        loss_it_variational = (1.0 / p_index) * (AdUNN_pNorm + Kappa * UNN * UNN) - \
                                              tf.multiply(tf.reshape(f(X_it, Y_it), shape=[-1, 1]), UNN)
                elif R['PDE_type'] == 'Convection_diffusion':
                    a_eps = A_eps(X_it, Y_it)                         # * 行 1 列
                    bx = Bx(X_it, Y_it)                               # * 行 1 列
                    by = By(X_it, Y_it)                               # * 行 1 列
                    dUNNx = tf.gather(dUNN, [0], axis=-1)             # * 行 1 列
                    dUNNy = tf.gather(dUNN, [1], axis=-1)             # * 行 1 列
                    bdUNN = bx*dUNNx + by*dUNNy                       # * 行 1 列
                    dAdU_pNorm = a_eps * tf.pow(dUNN_Norm, p_index)   # * 行 1 列
                    loss_it_variational = (dAdU_pNorm + bdUNN*UNN) - tf.multiply(f(X_it, Y_it), UNN)
                    # loss_it_variational = 0.5*(dAdU_pNorm + bdUNN * UNN) - tf.multiply(f(X_it, Y_it), UNN)
                    # loss_it_variational = tf.square((dAdU_pNorm + bdUNN * UNN) - tf.multiply(f(X_it, Y_it), UNN))
                loss_it = tf.reduce_mean(loss_it_variational)
            elif R['loss_type'] == 'lncosh_loss2Ritz':
                dUNN = tf.gradients(UNN, XY_it)[0]  # * 行 2 列
                dUNN_Norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])  # 按行求和
                if R['PDE_type'] == 'general_Laplace':
                    dUNN_2Norm = tf.square(dUNN_Norm)
                    loss_it_variational = (1.0 / 2) * dUNN_2Norm - \
                                          tf.multiply(tf.reshape(f(X_it, Y_it), shape=[-1, 1]), UNN)
                    cosh_loss_it = tf.cosh(lambda2lncosh * loss_it_variational)
                    loss_lncosh_it = (1.0 / lambda2lncosh)*tf.log(cosh_loss_it)
                elif R['PDE_type'] == 'pLaplace_explicit' or R['PDE_type'] == 'pLaplace_implicit':
                    a_eps = A_eps(X_it, Y_it)  # * 行 1 列
                    AdUNN_pNorm = tf.multiply(a_eps, tf.pow(dUNN_Norm, p_index))
                    if R['equa_name'] == 'multi_scale2D_7':
                        fxy = MS_LaplaceEqs.get_force_side2MS_E7(x=X_it, y=Y_it)
                        loss_it_variational = (1.0 / p_index) * AdUNN_pNorm - \
                                              tf.multiply(tf.reshape(fxy, shape=[-1, 1]), UNN)
                    else:
                        loss_it_variational = (1.0 / p_index) * AdUNN_pNorm - \
                                              tf.multiply(tf.reshape(f(X_it, Y_it), shape=[-1, 1]), UNN)
                    cosh_loss_it = tf.cosh(lambda2lncosh * loss_it_variational)
                    loss_lncosh_it = (1.0 / lambda2lncosh)*tf.log(cosh_loss_it)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it, Y_it)  # * 行 1 列
                    Kappa = kappa(X_it, Y_it)  # * 行 1 列
                    AdUNN_pNorm = a_eps * tf.pow(dUNN_Norm, p_index)  # * 行 1 列
                    if R['equa_name'] == 'Boltzmann3' or R['equa_name'] == 'Boltzmann4' or \
                            R['equa_name'] == 'Boltzmann5' or R['equa_name'] == 'Boltzmann6':
                        fxy = MS_BoltzmannEqs.get_foreside2Boltzmann2D(x=X_it, y=Y_it)
                        loss_it_variational = (1.0 / p_index) * (AdUNN_pNorm + Kappa * UNN * UNN) - \
                                              tf.multiply(tf.reshape(fxy, shape=[-1, 1]), UNN)
                    else:
                        loss_it_variational = (1.0 / p_index) * (AdUNN_pNorm + Kappa * UNN * UNN) - \
                                              tf.multiply(tf.reshape(f(X_it, Y_it), shape=[-1, 1]), UNN)
                    cosh_loss_it = tf.cosh(lambda2lncosh * loss_it_variational)
                    loss_lncosh_it = (1.0 / lambda2lncosh)*tf.log(cosh_loss_it)
                loss_it = tf.reduce_mean(loss_lncosh_it)
            elif R['loss_type'] == 'L2_loss':
                dUNN = tf.gradients(UNN, XY_it)[0]  # * 行 2 列
                if R['PDE_type'] == 'general_Laplace':
                    dUNNx = tf.gather(dUNN, [0], axis=-1)
                    dUNNy = tf.gather(dUNN, [1], axis=-1)
                    dUNNxxy = tf.gradients(dUNNx, XY_it)[0]
                    dUNNyxy = tf.gradients(dUNNy, XY_it)[0]
                    dUNNxx = tf.gather(dUNNxxy, [0], axis=-1)
                    dUNNyy = tf.gather(dUNNyxy, [1], axis=-1)
                    # -Laplace U=f --> -Laplace U - f --> -(Laplace U + f)
                    loss_it_L2 = tf.add(dUNNxx, dUNNyy) + tf.reshape(f(X_it, Y_it), shape=[-1, 1])
                    square_loss_it = tf.square(loss_it_L2)
                elif R['PDE_type'] == 'Convection_diffusion':
                    a_eps = A_eps(X_it, Y_it)  # * 行 1 列
                    bx = Bx(X_it, Y_it)
                    by = By(X_it, Y_it)
                    dUNNx = tf.gather(dUNN, [0], axis=-1)
                    dUNNy = tf.gather(dUNN, [1], axis=-1)

                    dUNNxxy = tf.gradients(dUNNx, XY_it)[0]
                    dUNNyxy = tf.gradients(dUNNy, XY_it)[0]
                    dUNNxx = tf.gather(dUNNxxy, [0], axis=-1)
                    dUNNyy = tf.gather(dUNNyxy, [1], axis=-1)
                    ddUNN = tf.add(dUNNxx, dUNNyy)
                    bdUNN = bx * dUNNx + by * dUNNy
                    loss_it_L2 = -a_eps*ddUNN + bdUNN - f(X_it, Y_it)
                    square_loss_it = tf.square(loss_it_L2)
                elif R['PDE_type'] == 'pLaplace_explicit' or R['PDE_type'] == 'pLaplace_implicit':
                    a_eps = A_eps(X_it, Y_it)
                    dUNNxy_norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])  # 按行求和
                    dUNNx = tf.gather(dUNN, [0], axis=-1)
                    dUNNy = tf.gather(dUNN, [1], axis=-1)
                    if p_index == 2:
                        dUxdUnorm = dUNNx
                        dUydUnorm = dUNNy
                    else:
                        dUxdUnorm = tf.multiply(tf.pow(dUNNxy_norm, p_index - 2), dUNNx)
                        dUydUnorm = tf.multiply(tf.pow(dUNNxy_norm, p_index - 2), dUNNy)

                    AdUxdUnorm = tf.multiply(a_eps, dUxdUnorm)
                    AdUydUnorm = tf.multiply(a_eps, dUydUnorm)

                    dAdUxdUnorm_xy = tf.gradients(AdUxdUnorm, XY_it)[0]
                    dAdUydUnorm_xy = tf.gradients(AdUydUnorm, XY_it)[0]

                    dAdUxdUnorm_x = tf.gather(dAdUxdUnorm_xy, [0], axis=-1)
                    dAdUydUnorm_y = tf.gather(dAdUydUnorm_xy, [1], axis=-1)

                    div_AdUdUnorm = tf.add(dAdUxdUnorm_x, dAdUydUnorm_y)
                    loss_it_L2 = -div_AdUdUnorm - f(X_it, Y_it)
                    square_loss_it = tf.square(loss_it_L2)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it, Y_it)
                    Kappa = kappa(X_it, Y_it)  # * 行 1 列
                    dUNNxy_norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])  # 按行求和
                    dUNNx = tf.gather(dUNN, [0], axis=-1)
                    dUNNy = tf.gather(dUNN, [1], axis=-1)
                    if p_index == 2:
                        dUxdUnorm = dUNNx
                        dUydUnorm = dUNNy
                    else:
                        dUxdUnorm = tf.multiply(tf.pow(dUNNxy_norm, p_index - 2), dUNNx)
                        dUydUnorm = tf.multiply(tf.pow(dUNNxy_norm, p_index - 2), dUNNy)

                    AdUxdUnorm = tf.multiply(a_eps, dUxdUnorm)
                    AdUydUnorm = tf.multiply(a_eps, dUydUnorm)

                    dAdUxdUnorm_xy = tf.gradients(AdUxdUnorm, XY_it)[0]
                    dAdUydUnorm_xy = tf.gradients(AdUydUnorm, XY_it)[0]

                    dAdUxdUnorm_x = tf.gather(dAdUxdUnorm_xy, [0], axis=-1)
                    dAdUydUnorm_y = tf.gather(dAdUydUnorm_xy, [1], axis=-1)

                    div_AdUdUnorm = tf.add(dAdUxdUnorm_x, dAdUydUnorm_y)
                    KU = tf.multiply(Kappa, UNN)
                    loss_it_L2 = -div_AdUdUnorm + KU - f(X_it, Y_it)
                    square_loss_it = tf.square(loss_it_L2)
                # loss_it = tf.reduce_mean(square_loss_it) * (region_rt - region_lb) * (region_rt - region_lb)
                loss_it = tf.reduce_mean(square_loss_it)

            U_left = u_left(tf.reshape(XY_left_bd[:, 0], shape=[-1, 1]), tf.reshape(XY_left_bd[:, 1], shape=[-1, 1]))
            U_right = u_right(tf.reshape(XY_right_bd[:, 0], shape=[-1, 1]), tf.reshape(XY_right_bd[:, 1], shape=[-1, 1]))
            U_bottom = u_bottom(tf.reshape(XY_bottom_bd[:, 0], shape=[-1, 1]), tf.reshape(XY_bottom_bd[:, 1], shape=[-1, 1]))
            U_top = u_top(tf.reshape(XY_top_bd[:, 0], shape=[-1, 1]), tf.reshape(XY_top_bd[:, 1], shape=[-1, 1]))
            if R['loss_type'] == 'lncosh_loss2Ritz':
                cosh_bd = tf.cosh(lambda2lncosh*(UNN_left-U_left))+tf.cosh(lambda2lncosh*(UNN_right-U_right)) + \
                          tf.cosh(lambda2lncosh * (UNN_bottom - U_bottom)) + tf.cosh(lambda2lncosh * (UNN_top - U_top))
                loss_cosh_bd = (1.0 / lambda2lncosh) * tf.log(cosh_bd)
                loss_bd = tf.reduce_mean(loss_cosh_bd)
            else:
                cosh_bd = tf.cosh(lambda2lncosh * (UNN_left - U_left)) + tf.cosh(
                    lambda2lncosh * (UNN_right - U_right)) + \
                          tf.cosh(lambda2lncosh * (UNN_bottom - U_bottom)) + tf.cosh(lambda2lncosh * (UNN_top - U_top))
                loss_cosh_bd = (1.0 / lambda2lncosh) * tf.log(cosh_bd)
                loss_bd = tf.reduce_mean(loss_cosh_bd)

                # loss_bd_square = tf.square(UNN_left - U_left) + tf.square(UNN_right - U_right) + \
                #                  tf.square(UNN_bottom - U_bottom) + tf.square(UNN_top - U_top)
                # loss_bd = tf.reduce_mean(loss_bd_square)

            if R['regular_wb_model'] == 'L1':
                regularSum2WB = DNN_base.regular_weights_biases_L1(W2NN, B2NN)    # 正则化权重和偏置 L1正则化
            elif R['regular_wb_model'] == 'L2':
                regularSum2WB = DNN_base.regular_weights_biases_L2(W2NN, B2NN)    # 正则化权重和偏置 L2正则化
            else:
                regularSum2WB = tf.constant(0.0)                                  # 无正则化权重参数

            PWB = penalty2WB * regularSum2WB
            loss = loss_it + boundary_penalty * loss_bd + PWB                     # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            if R['train_model'] == 'group3_training':
                train_op1 = my_optimizer.minimize(loss_it, global_step=global_steps)
                train_op2 = my_optimizer.minimize(loss_bd, global_step=global_steps)
                train_op3 = my_optimizer.minimize(loss, global_step=global_steps)
                train_my_loss = tf.group(train_op1, train_op2, train_op3)
            elif R['train_model'] == 'group2_training':
                train_op2bd = my_optimizer.minimize(loss_bd, global_step=global_steps)
                train_op2union = my_optimizer.minimize(loss, global_step=global_steps)
                train_my_loss = tf.gruop(train_op2union, train_op2bd)
            elif R['train_model'] == 'union_training':
                train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

            # 训练上的真解值和训练结果的误差
            if R['PDE_type'] == 'general_Laplace' or R['PDE_type'] == 'pLaplace_explicit' \
                        or R['PDE_type'] == 'Possion_Boltzmann' or R['PDE_type'] == 'Convection_diffusion':
                U_true = u_true(X_it, Y_it)
                train_mse = tf.reduce_mean(tf.square(U_true - UNN))
                train_rel = train_mse / tf.reduce_mean(tf.square(U_true))
                loss_it2Utrue = tf.square(0.0)
            else:
                train_mse = tf.constant(0.0)
                train_rel = tf.constant(0.0)
                loss_it2Utrue = tf.square(0.0)

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        test_bach_size = 1600
        size2test = 40
        # test_bach_size = 4900
        # size2test = 70
        # test_bach_size = 10000
        # size2test = 100
        # test_bach_size = 40000
        # size2test = 200
        # test_bach_size = 250000
        # size2test = 500
        # test_bach_size = 1000000
        # size2test = 1000
        test_xy_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
        saveData.save_testData_or_solus2mat(test_xy_bach, dataName='testXY', outPath=R['FolderName'])
    else:
        if R['PDE_type'] == 'pLaplace_implicit' or R['PDE_type'] == 'pLaplace_explicit':
            test_xy_bach = matData2pLaplace.get_data2pLaplace(equation_name=R['equa_name'], mesh_number=mesh_number)
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))
            saveData.save_meshData2mat(test_xy_bach, dataName='meshXY', mesh_number=mesh_number, outPath=R['FolderName'])
        elif R['PDE_type'] == 'Possion_Boltzmann':
            if region_lb == (-1.0) and region_rt == 1.0:
                name2data_file = '11'
            else:
                name2data_file = '01'
            test_xy_bach = matData2Boltzmann.get_meshData2Boltzmann(domain_lr=name2data_file, mesh_number=mesh_number)
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))
            saveData.save_meshData2mat(test_xy_bach, dataName='meshXY', mesh_number=mesh_number,
                                       outPath=R['FolderName'])
        elif R['PDE_type'] == 'Convection_diffusion':
            if region_lb == (-1.0) and region_rt == 1.0:
                name2data_file = '11'
            else:
                name2data_file = '01'
            test_xy_bach = matData2Boltzmann.get_meshData2Boltzmann(domain_lr=name2data_file, mesh_number=mesh_number)
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))
            saveData.save_meshData2mat(test_xy_bach, dataName='meshXY', mesh_number=mesh_number,
                                       outPath=R['FolderName'])
        else:
            test_xy_bach = matData2Laplace.get_randData2Laplace(dim=input_dim, data_path='dataMat_highDim')
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        train_option = True
        for i_epoch in range(R['max_epoch'] + 1):
            xy_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            xl_bd_batch, xr_bd_batch, yb_bd_batch, yt_bd_batch = DNN_data.rand_bd_2D(batchsize_bd, input_dim,
                                                                                     region_a=region_lb, region_b=region_rt)
            tmp_lr = tmp_lr * (1 - lr_decay)
            if R['activate_penalty2bd_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 50 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 100 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 200 * bd_penalty_init
                else:
                    temp_penalty_bd = 500 * bd_penalty_init
            elif R['activate_penalty2bd_increase'] == 2:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = 5*bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 1 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 0.5 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 0.1 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 0.05 * bd_penalty_init
                else:
                    temp_penalty_bd = 0.02 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init

            _, loss_it_tmp, loss_bd_tmp, loss_tmp, train_mse_tmp, train_rel_tmp, pwb = sess.run(
                [train_my_loss, loss_it, loss_bd, loss, train_mse, train_rel, PWB],
                feed_dict={XY_it: xy_it_batch, XY_left_bd: xl_bd_batch, XY_right_bd: xr_bd_batch,
                           XY_bottom_bd: yb_bd_batch, XY_top_bd: yt_bd_batch, in_learning_rate: tmp_lr,
                           boundary_penalty: temp_penalty_bd, train_opt: train_option})

            loss_it_all.append(loss_it_tmp)
            loss_bd_all.append(loss_bd_tmp)
            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_rel_tmp)

            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                DNN_tools.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, pwb, loss_it_tmp, loss_bd_tmp, loss_tmp, train_mse_tmp,
                    train_rel_tmp, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                if R['PDE_type'] == 'general_Laplace' or R['PDE_type'] == 'pLaplace_explicit' or \
                        R['PDE_type'] == 'Possion_Boltzmann' or R['PDE_type'] == 'Convection_diffusion':
                    u_true2test, unn2test = sess.run([U_true, UNN], feed_dict={XY_it: test_xy_bach,
                                                                               train_opt: train_option})
                else:
                    u_true2test = u_true
                    unn2test = sess.run(UNN,  feed_dict={XY_it: test_xy_bach, train_opt: train_option})

                point_square_error = np.square(u_true2test - unn2test)
                mse2test = np.mean(point_square_error)
                test_mse_all.append(mse2test)
                res2test = mse2test / np.mean(np.square(u_true2test))
                test_rel_all.append(res2test)

                DNN_tools.print_and_log_test_one_epoch(mse2test, res2test, log_out=log_fileout)

        # ------------------- save the testing results into mat file and plot them -------------------------
        saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act_func,
                                             outPath=R['FolderName'])
        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act_func, seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)

        # ----------------------  save testing results to mat files, then plot them --------------------------------
        saveData.save_2testSolus2mat(u_true2test, unn2test, actName='utrue', actName1=act_func,
                                     outPath=R['FolderName'])

        plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue', seedNo=R['seed'],
                                        outPath=R['FolderName'])
        plotData.plot_Hot_solution2test(unn2test, size_vec2mat=size2test, actName=act_func, seedNo=R['seed'],
                                        outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act_func,
                                  seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

        saveData.save_test_point_wise_err2mat(point_square_error, actName=act_func, outPath=R['FolderName'])

        plotData.plot_Hot_point_wise_err(point_square_error, size_vec2mat=size2test, actName=act_func,
                                         seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0
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
    # store_file = 'Laplace2D'
    store_file = 'pLaplace2D'
    # store_file = 'Boltzmann2D'
    # store_file = 'Convection2D'
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

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of multi-scale problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R['input_dim'] = 2                                  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                                 # 输出维数

    if store_file == 'Laplace2D':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace2D':
        R['PDE_type'] = 'pLaplace_implicit'
        # R['equa_name'] = 'multi_scale2D_1'      # p=2 区域为 [-1,1]X[-1,1]
        # R['equa_name'] = 'multi_scale2D_2'      # p=2 区域为 [-1,1]X[-1,1]
        # R['equa_name'] = 'multi_scale2D_3'      # p=2 区域为 [-1,1]X[-1,1] 论文中的例子
        R['equa_name'] = 'multi_scale2D_4'      # p=2 区域为 [-1,1]X[-1,1] 论文中的例子
        # R['equa_name'] = 'multi_scale2D_5'      # p=3 区域为 [0,1]X[0,1]   和例三的系数A一样
        # R['equa_name'] = 'multi_scale2D_6'      # p=3 区域为 [-1,1]X[-1,1] 和例三的系数A一样

        # R['PDE_type'] = 'pLaplace_explicit'
        # R['equa_name'] = 'multi_scale2D_7'      # p=2 区域为 [0,1]X[0,1]
    elif store_file == 'Boltzmann2D':
        R['PDE_type'] = 'Possion_Boltzmann'
        # R['equa_name'] = 'Boltzmann1'           # p=2 区域为 [-1,1]X[-1,1]
        # R['equa_name'] = 'Boltzmann2'             # p=2 区域为 [-1,1]X[-1,1]
        # R['equa_name'] = 'Boltzmann3'
        # R['equa_name'] = 'Boltzmann4'
        R['equa_name'] = 'Boltzmann5'
    elif store_file == 'Convection2D':
        R['PDE_type'] = 'Convection_diffusion'
        # R['equa_name'] = 'Convection1'
        R['equa_name'] = 'Convection2'

    if R['PDE_type'] == 'general_Laplace':
        R['mesh_number'] = 6
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
    else:
        epsilon = 0.1                  # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)  # 字符串转为浮点

    if R['PDE_type'] == 'pLaplace_explicit' or R['PDE_type'] == 'pLaplace_implicit':
        order2pLaplace = input('please input the order(a int number) to pLaplace:')
        order = float(order2pLaplace)
        R['order2pLaplace_operator'] = order

    if R['PDE_type'] == 'pLaplace_implicit':
        # 网格大小设置
        mesh_number = input('please input mesh_number =')     # 由终端输入的会记录为字符串形式
        R['mesh_number'] = int(mesh_number)                   # 字符串转为浮点
    elif R['PDE_type'] == 'Possion_Boltzmann' or R['PDE_type'] == 'pLaplace_explicit'\
            or R['PDE_type'] == 'Convection_diffusion':
        R['mesh_number'] = int(6)
        R['order2pLaplace_operator'] = float(2)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    if R['PDE_type'] == 'pLaplace_implicit':
        R['batch_size2interior'] = 3000      # 内部训练数据的批大小
        # R['batch_size2interior'] = 10000   # 内部训练数据的批大小
        if R['mesh_number'] == 2:
            R['batch_size2boundary'] = 25    # 边界训练数据的批大小
        elif R['mesh_number'] == 3:
            R['batch_size2boundary'] = 100   # 边界训练数据的批大小
        elif R['mesh_number'] == 4:
            R['batch_size2boundary'] = 200   # 边界训练数据的批大小
        elif R['mesh_number'] == 5:
            R['batch_size2boundary'] = 300   # 边界训练数据的批大小
        elif R['mesh_number'] == 6:
            R['batch_size2boundary'] = 500   # 边界训练数据的批大小
    else:
        R['batch_size2interior'] = 3000      # 内部训练数据的批大小
        R['batch_size2boundary'] = 500       # 边界训练数据的批大小

    # 装载测试数据模式
    R['testData_model'] = 'loadData'
    # R['testData_model'] = 'random_generate'

    # R['loss_type'] = 'L2_loss'                             # loss类型:L2 loss
    R['loss_type'] = 'variational_loss'                      # loss类型:PDE变分
    # R['loss_type'] = 'lncosh_loss2Ritz'
    R['lambda2lncosh'] = 50.0

    R['optimizer_name'] = 'Adam'                          # 优化器
    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['train_model'] = 'union_training'
    # R['train_model'] = 'group2_training'
    # R['train_model'] = 'group3_training'

    # 正则化权重和偏置的模式
    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    R['penalty2weight_biases'] = 0.000                    # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                 # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2bd_increase'] = 1
    # R['init_boundary_penalty'] = 1000                     # Regularization parameter for boundary conditions
    R['init_boundary_penalty'] = 100                     # Regularization parameter for boundary conditions

    # 网络的频率范围设置
    R['freqs'] = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model'] = 'DNN'
    # R['model'] = 'DNN_scale'
    # R['model'] = 'DNN_adapt_scale'
    R['model'] = 'DNN_FourierBase'
    # R['model'] = 'DNN_FourierBase2'
    # R['model'] = 'DNN_Sin+Cos_Base'
    # R['model'] = 'DNN_WaveletBase'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model'] == 'DNN_FourierBase' or R['model'] == 'DNN_FourierBase2':
        R['hidden_layers'] = (125, 200, 200, 100, 100, 80)  # 1*125+250*200+200*200+200*100+100*100+100*50+50*1=128205
    else:
        # R['hidden_layers'] = (100, 80, 80, 60, 40, 40)
        # R['hidden_layers'] = (200, 100, 80, 50, 30)
        R['hidden_layers'] = (250, 200, 200, 100, 100, 80)  # 1*250+250*200+200*200+200*100+100*100+100*50+50*1=128330
        # R['hidden_layers'] = (500, 400, 300, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['activate_func'] = 'relu'
    # R['activate_func'] = 'tanh'
    # R['activate_func']' = leaky_relu'
    # R['activate_func'] = 'srelu'
    R['activate_func'] = 's2relu'
    # R['activate_func'] = 'scsrelu'
    # R['activate_func'] = 'sin'
    # R['activate_func'] = 'sinAddcos'
    # R['activate_func'] = 'elu'
    # R['activate_func'] = 'phi'

    if R['model'] == 'DNN_FourierBase' and R['activate_func'] == 'tanh':
        R['sfourier'] = 0.5
        # R['sfourier'] = 1.0
    elif R['model'] == 'DNN_FourierBase' and R['activate_func'] == 's2relu':
        R['sfourier'] = 0.5
        # R['sfourier'] = 1.0
    elif R['model'] == 'DNN_FourierBase' and R['activate_func'] == 'sinAddcos':
        R['sfourier'] = 0.5
        # R['sfourier'] = 1.0
    elif R['model'] == 'DNN_FourierBase' and R['activate_func'] == 'sin':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    elif R['model'] == 'DNN_FourierBase' and R['activate_func'] == 'scsrelu':
        R['sfourier'] = 0.5
        # R['sfourier'] = 1.0

    solve_Multiscale_PDE(R)

