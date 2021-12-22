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
import DNN_data
import time
import DNN_base
import General_Laplace
import MS_LaplaceEqs
import MS_BoltzmannEqs
import MS_ConvectionEqs
import random2pLaplace
import plotData
import saveData
import DNN_Log_Print


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']                                     # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):                               # 判断路径是否已经存在
        os.mkdir(log_out_path)                                         # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('logTrain', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary_out2file(R, log_fileout)

    # 一般 laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['init_boundary_penalty']                # Regularization parameter for boundary conditions
    penalty2WB = R['penalty2weight_biases']                     # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    hidden_layers = R['hidden_layers']
    act_func = R['name2act_hidden']

    # ------- set the problem ---------
    input_dim = R['input_dim']
    out_dim = R['output_dim']

    region_l = 0.0
    region_r = 1.0
    if R['PDE_type'] == 'general_Laplace':
        # -laplace u = f
        region_l = 0.0
        region_r = 1.0
        f, u_true, u_left, u_right = General_Laplace.get_infos2Laplace_1D(
            input_dim=input_dim, out_dim=out_dim, intervalL=region_l, intervalR=region_r, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
        #       dx     ****         dx        ****
        # 问题区域，每个方向设置为一样的长度。等网格划分，对于二维是方形区域
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        region_l = 0.0
        region_r = 1.0
        if R['equa_name'] == 'multi_scale':
            u_true, f, A_eps, u_left, u_right = MS_LaplaceEqs.get_infos2pLaplace_1D(
                in_dim=input_dim, out_dim=out_dim, intervalL=region_l, intervalR=region_r, index2p=p_index, eps=epsilon)
        elif R['equa_name'] == '3scale2':
            epsilon2 = 0.01
            u_true, A_eps, u_left, u_right = MS_LaplaceEqs.get_infos2pLaplace_1D_3scale2(
                in_dim=input_dim, out_dim=out_dim, intervalL=region_l, intervalR=region_r, index2p=p_index, eps1=epsilon,
                eps2=epsilon2)
        elif R['equa_name'] == 'rand_ceof':
            num2sun_term = 2
            Alpha = 1.2
            Xi1 = [-0.25, 0.25]
            Xi2 = [-0.3, 0.3]
            # Xi1 = np.random.uniform(-0.5, 0.5, num2sun_term)
            print('Xi1:', Xi1)
            print('\n')
            # Xi2 = np.random.uniform(-0.5, 0.5, num2sun_term)
            print('Xi2:', Xi2)
            print('\n')
            u_left, u_right = random2pLaplace.random_boundary()
        elif R['equa_name'] == 'rand_sin_ceof':
            Xi1=-0.25
            Xi2=0.25
            u_true, f, A_eps = random2pLaplace.random_equa2()
    elif R['PDE_type'] == 'Possion_Boltzmann':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) | + K(x)u_eps(x) =f(x), x \in R^n
        #       dx     ****         dx        ****
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        region_l = 0.0
        region_r = 1.0
        A_eps, kappa, u_true, u_left, u_right, f = MS_BoltzmannEqs.get_infos2Boltzmann_1D(
            in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, index2p=p_index, eps=epsilon,
            eqs_name=R['equa_name'])

    # 初始化权重和和偏置
    flag1 = 'WB'
    if R['model2NN'] == 'DNN_FourierBase':
        Weights, Biases = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_layers, flag1)
    else:
        Weights, Biases = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_layers, flag1)

    global_steps = tf.compat.v1.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            X_it = tf.compat.v1.placeholder(tf.float32, name='X_it', shape=[None, input_dim])          # * 行 1 列
            X_left = tf.compat.v1.placeholder(tf.float32, name='X_left', shape=[None, input_dim])      # * 行 1 列
            X_right = tf.compat.v1.placeholder(tf.float32, name='X_right', shape=[None, input_dim])    # * 行 1 列
            bd_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            # 供选择的网络模式
            if R['model2NN'] == 'DNN':
                UNN = DNN_base.DNN(X_it, Weights, Biases, hidden_layers, activateIn_name=R['name2act_in'],
                                   activate_name=R['name2act_hidden'], activateOut_name=R['activateOut_func'])
                UNN_left = DNN_base.DNN(X_left, Weights, Biases, hidden_layers, activateIn_name=R['name2act_in'],
                                        activate_name=R['name2act_hidden'], activateOut_name=R['name2act_out'])
                UNN_right = DNN_base.DNN(X_right, Weights, Biases, hidden_layers, activateIn_name=R['name2act_in'],
                                         activate_name=R['name2act_hidden'], activateOut_name=R['name2act_out'])
            elif R['model2NN'] == 'DNN_scale':
                freqs = R['freq']
                UNN = DNN_base.DNN_scale(X_it, Weights, Biases, hidden_layers, freqs, activateIn_name=R['name2act_in'],
                                         activate_name=act_func, activateOut_name=R['name2act_out'])
                UNN_left = DNN_base.DNN_scale(X_left, Weights, Biases, hidden_layers, freqs,
                                              activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'],
                                              activateOut_name=R['name2act_out'])
                UNN_right = DNN_base.DNN_scale(X_right, Weights, Biases, hidden_layers, freqs,
                                               activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'],
                                               activateOut_name=R['name2act_out'])
            elif R['model2NN'] == 'DNN_adapt_scale':
                freqs = R['freq']
                UNN = DNN_base.DNN_adapt_scale(X_it, Weights, Biases, hidden_layers, freqs,
                                               activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'],
                                               activateOut_name=R['name2act_out'])
                UNN_left = DNN_base.DNN_adapt_scale(X_left, Weights, Biases, hidden_layers, freqs,
                                                    activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'],
                                                    activateOut_name=R['name2act_out'])
                UNN_right = DNN_base.DNN_adapt_scale(X_right, Weights, Biases, hidden_layers, freqs,
                                                     activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'],
                                                     activateOut_name=R['name2act_out'])
            elif R['model2NN'] == 'DNN_FourierBase':
                freqs = R['freq']
                UNN = DNN_base.DNN_FourierBase(X_it, Weights, Biases, hidden_layers, freqs,
                                               activate_name=act_func, activateOut_name=R['name2act_out'],
                                               sFourier=R['sfourier'])
                UNN_left = DNN_base.DNN_FourierBase(X_left, Weights, Biases, hidden_layers, freqs,
                                                    activate_name=act_func, activateOut_name=R['name2act_out'],
                                                    sFourier=R['sfourier'])
                UNN_right = DNN_base.DNN_FourierBase(X_right, Weights, Biases, hidden_layers, freqs,
                                                     activate_name=act_func, activateOut_name=R['name2act_out'],
                                                     sFourier=R['sfourier'])

            # 变分形式的loss of interior，训练得到的 UNN 是 * 行 1 列
            if R['loss_type'] == 'variational_loss':
                dUNN = tf.gradients(UNN, X_it)
                if R['PDE_type'] == 'general_Laplace':
                    dUNN_Norm = tf.reduce_sum(tf.square(dUNN), axis=-1)
                    loss_it_variational = (1.0 / 2) * tf.reshape(dUNN_Norm, shape=[-1, 1]) - \
                                           tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), UNN)
                elif R['PDE_type'] == 'pLaplace':
                    if R['equa_name'] == '3scale2':
                        a_eps = A_eps(X_it)                          # * 行 1 列
                        AdUNN_pNorm = tf.reduce_sum(a_eps * tf.pow(tf.abs(dUNN), p_index), axis=-1)
                        fx = MS_LaplaceEqs.force_sice_3scale2(X_it, eps1=R['epsilon'], eps2=0.01)
                        loss_it_variational = (1.0 / p_index) * tf.reshape(AdUNN_pNorm, shape=[-1, 1]) - \
                                              tf.multiply(tf.reshape(fx, shape=[-1, 1]), UNN)
                    elif R['equa_name'] == 'rand_ceof':
                        a_eps = random2pLaplace.rangdom_ceof(x=X_it, xi1=Xi1, xi2=Xi2, K=num2sun_term, alpha=Alpha)
                        # fx = random2pLaplace.rangdom_force(x=X_it, xi1=Xi1, xi2=Xi2, K=num2sun_term, alpha=Alpha)
                        # duf = tf.gradients(force_side, X_it)
                        fx = random2pLaplace.rangdom_diff_force2x(x=X_it, xi1=Xi1, xi2=Xi2, K=num2sun_term, alpha=Alpha)
                        AdUNN_pNorm = tf.reduce_sum(a_eps * tf.pow(tf.abs(dUNN), p_index), axis=-1)
                        loss_it_variational = (1.0 / p_index) * tf.reshape(AdUNN_pNorm, shape=[-1, 1]) - \
                                              tf.multiply(tf.reshape(fx, shape=[-1, 1]), UNN)
                    elif R['equa_name'] == 'rand_sin_ceof':
                        a_eps = 1.0
                        fx = random2pLaplace.random_sin_f(x=X_it, xi1=Xi1, xi2=Xi2, K=2, alpha=1.0)
                        AdUNN_pNorm = tf.reduce_sum(a_eps * tf.pow(tf.abs(dUNN), p_index), axis=-1)
                        loss_it_variational = (1.0 / p_index) * tf.reshape(AdUNN_pNorm, shape=[-1, 1]) - \
                                              tf.multiply(tf.reshape(fx, shape=[-1, 1]), UNN)
                    else:
                        # a_eps = A_eps(X_it)                          # * 行 1 列
                        a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                        AdUNN_pNorm = tf.reduce_sum(a_eps * tf.pow(tf.abs(dUNN), p_index), axis=-1)
                        loss_it_variational = (1.0 / p_index) * tf.reshape(AdUNN_pNorm, shape=[-1, 1]) - \
                                               tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), UNN)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    # a_eps = A_eps(X_it)                          # * 行 1 列
                    a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    Kappa = kappa(X_it)
                    AdUNN_pNorm = tf.reduce_sum(a_eps * tf.pow(tf.abs(dUNN), p_index), axis=-1)
                    if R['equa_name'] == 'Boltzmann2':
                        fside = MS_BoltzmannEqs.get_force_side2Boltzmann_1D(X_it, index2p=p_index, eps=epsilon)
                    else:
                        fside = tf.reshape(f(X_it), shape=[-1, 1])
                    if p_index == 1:
                        loss_it_variational = (1.0 / 2) * (tf.reshape(AdUNN_pNorm, shape=[-1, 1]) +
                                               Kappa*UNN*UNN) - tf.multiply(fside, UNN)
                    elif p_index == 2:
                        loss_it_variational = (1.0 / 2) * (tf.reshape(AdUNN_pNorm, shape=[-1, 1]) +
                                               Kappa*UNN*UNN*UNN) - tf.multiply(fside, UNN)

                loss_it = tf.reduce_mean(loss_it_variational)
            elif R['loss_type'] == 'L2_loss':
                dUNN = tf.gradients(UNN, X_it)
                if R['PDE_type'] == 'general_Laplace':
                    ddUNN = tf.gradients(dUNN, X_it)
                    loss_it_L2 = -1.0*ddUNN - tf.reshape(f(X_it), shape=[-1, 1])
                    square_loss_it = tf.square(loss_it_L2)
                elif R['PDE_type'] == 'pLaplace':
                    a_eps = A_eps(X_it)
                    # a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    if p_index == 2.0:
                        AdUNNpNorm = 1.0*a_eps
                    elif p_index == 3.0:
                        dUNNp_2Nrom = tf.abs(dUNN)
                        AdUNNpNorm = tf.multiply(a_eps, dUNNp_2Nrom)
                    else:
                        dUNNp_2Nrom = tf.pow(tf.abs(dUNN), p_index-2.0)
                        AdUNNpNorm = tf.multiply(a_eps, dUNNp_2Nrom)
                    AdUNNpNorm_dUNN = tf.multiply(AdUNNpNorm, dUNN)
                    dAdUNNpNorm_dUNN = tf.gradients(AdUNNpNorm_dUNN, X_it)
                    if R['equa_name'] == '3scale2':
                        fx = MS_LaplaceEqs.force_sice_3scale2(X_it, eps1=R['epsilon'], eps2=0.01)
                        loss_it_L2 = dAdUNNpNorm_dUNN + tf.reshape(fx, shape=[-1, 1])
                    else:
                        loss_it_L2 = dAdUNNpNorm_dUNN + tf.reshape(f(X_it), shape=[-1, 1])
                    square_loss_it = tf.square(loss_it_L2)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it)
                    # a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    Kappa = kappa(X_it)
                    if p_index == 2.0:
                        AdUNNpNorm = 1.0*a_eps
                    elif p_index == 3.0:
                        dUNNp_2Nrom = tf.abs(dUNN)
                        AdUNNpNorm = tf.multiply(a_eps, dUNNp_2Nrom)
                    else:
                        dUNNp_2Nrom = tf.pow(tf.abs(dUNN), p_index-2.0)
                        AdUNNpNorm = tf.multiply(a_eps, dUNNp_2Nrom)
                    AdUNNpNorm_dUNN = tf.multiply(AdUNNpNorm, dUNN)
                    dAdUNNpNorm_dUNN = tf.gradients(AdUNNpNorm_dUNN, X_it)
                    loss_it_L2 = -1.0*dAdUNNpNorm_dUNN + Kappa*UNN - tf.reshape(f(X_it), shape=[-1, 1])

                    square_loss_it = tf.square(loss_it_L2)

                # loss_it = tf.reduce_mean(loss_it_L2)*(region_r-region_l)
                loss_it = tf.reduce_mean(square_loss_it)

            U_left = u_left(X_left)
            U_right = u_right(X_right)
            loss_bd_square = tf.square(UNN_left - U_left) + tf.square(UNN_right - U_right)
            loss_bd = tf.reduce_mean(loss_bd_square)

            if R['regular_wb_model'] == 'L1':
                regularSum2WB = DNN_base.regular_weights_biases_L1(Weights, Biases)    # 正则化权重和偏置 L1正则化
            elif R['regular_wb_model'] == 'L2':
                regularSum2WB = DNN_base.regular_weights_biases_L2(Weights, Biases)    # 正则化权重和偏置 L2正则化
            else:
                regularSum2WB = tf.constant(0.0)                                       # 无正则化权重参数

            PWB = penalty2WB * regularSum2WB
            loss = loss_it + bd_penalty * loss_bd + PWB         # 要优化的loss function

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            if R['train_model'] == 'group3_training':
                train_op1 = my_optimizer.minimize(loss_it, global_step=global_steps)
                train_op2 = my_optimizer.minimize(loss_bd, global_step=global_steps)
                train_op3 = my_optimizer.minimize(loss, global_step=global_steps)
                train_my_loss = tf.group(train_op1, train_op2, train_op3)
            elif R['train_model'] == 'group2_training':
                train_op2 = my_optimizer.minimize(loss_bd, global_step=global_steps)
                train_op3 = my_optimizer.minimize(loss, global_step=global_steps)
                train_my_loss = tf.group(train_op2, train_op3)
            elif R['train_model'] == 'union_training':
                train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

            # 训练上的真解值和训练结果的误差
            if R['equa_name'] == 'rand_ceof':
                U_true = random2pLaplace.rangdom_exact_solution_1(x=X_it, xi1=Xi1, xi2=Xi2, K=num2sun_term, alpha=Alpha)
                mean_square_error = tf.reduce_mean(tf.square(U_true - dUNN))
                residual_error = mean_square_error / tf.reduce_mean(tf.square(U_true))
            else:
                U_true = u_true(X_it)
                mean_square_error = tf.reduce_mean(tf.square(U_true - UNN))
                residual_error = mean_square_error / tf.reduce_mean(tf.square(U_true))

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    testing_epoch = []

    test_batch_size = 1000
    test_x_bach = np.reshape(np.linspace(region_l, region_r, num=test_batch_size), [-1, 1])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            x_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_l, region_b=region_r)
            xl_bd_batch, xr_bd_batch = DNN_data.rand_bd_1D(batchsize_bd, input_dim, region_a=region_l, region_b=region_r)
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
            else:
                temp_penalty_bd = bd_penalty_init

            _, loss_it_tmp, loss_bd_tmp, loss_tmp, train_mse_tmp, train_res_tmp, pwb = sess.run(
                [train_my_loss, loss_it, loss_bd, loss, mean_square_error, residual_error, PWB],
                feed_dict={X_it: x_it_batch, X_left: xl_bd_batch, X_right: xr_bd_batch,
                           in_learning_rate: tmp_lr, bd_penalty: temp_penalty_bd})

            loss_it_all.append(loss_it_tmp)
            loss_bd_all.append(loss_bd_tmp)
            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_res_tmp)
            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                DNN_Log_Print.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, pwb, loss_it_tmp, loss_bd_tmp, loss_tmp, train_mse_tmp,
                    train_res_tmp, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                testing_epoch.append(i_epoch / 1000)
                if R['equa_name'] == 'rand_ceof':
                    u_true2test, unn2test = sess.run([U_true, dUNN], feed_dict={X_it: test_x_bach})
                else:
                    u_true2test, unn2test = sess.run([U_true, UNN], feed_dict={X_it: test_x_bach})
                mse2test = np.mean(np.square(u_true2test - unn2test))
                test_mse_all.append(mse2test)
                res2test = mse2test / np.mean(np.square(u_true2test))
                test_rel_all.append(res2test)

                DNN_Log_Print.print_and_log_test_one_epoch(mse2test, res2test, log_out=log_fileout)

    # -----------------------  save training results to mat files, then plot them ---------------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act_func, outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])
    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act_func, seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    saveData.save_2testSolus2mat(u_true2test, unn2test, actName='utrue', actName1=act_func, outPath=R['FolderName'])
    plotData.plot_2solutions2test(u_true2test, unn2test, coord_points2test=test_x_bach,
                                  batch_size2test=test_batch_size, seedNo=R['seed'], outPath=R['FolderName'],
                                  subfig_type=R['subfig_type'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func, outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, testing_epoch, actName=act_func,
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 1
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）
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

    # ------------------------------------------- 文件保存路径设置 ----------------------------------------
    # store_file = 'Laplace1D'
    store_file = 'pLaplace1D'
    # store_file = 'Boltzmann1D'
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

    # ---------------------------- Setup of laplace equation ------------------------------
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    if store_file == 'Laplace1D':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace1D':
        R['PDE_type'] = 'pLaplace'
        R['equa_name'] = 'multi_scale'
        # R['equa_name'] = '3scale2'
        # R['equa_name'] = 'rand_ceof'
        # R['equa_name'] = 'rand_sin_ceof'
    elif store_file == 'Boltzmann1D':
        R['PDE_type'] = 'Possion_Boltzmann'
        # R['equa_name'] = 'Boltzmann1'
        R['equa_name'] = 'Boltzmann2'

    if R['PDE_type'] == 'general_Laplace':
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
    elif R['PDE_type'] == 'pLaplace'or R['PDE_type'] == 'Possion_Boltzmann':
        # 频率设置
        epsilon = input('please input epsilon =')         # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)                     # 字符串转为浮点

        # 问题幂次
        order2pLaplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2pLaplace)
        R['order2pLaplace_operator'] = order

    R['input_dim'] = 1                                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                                   # 输出维数

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    R['batch_size2interior'] = 3000                       # 内部训练数据的批大小
    R['batch_size2boundary'] = 500                        # 边界训练数据大小

    # 装载测试数据模式和画图
    R['plot_ongoing'] = 0
    R['subfig_type'] = 1
    R['testData_model'] = 'loadData'
    # R['testData_model'] = 'random_generate'

    # R['loss_type'] = 'L2_loss'                            # PDE变分
    R['loss_type'] = 'variational_loss'                   # PDE变分

    if R['loss_type'] == 'L2_loss':
        R['batch_size2interior'] = 15000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 2500   # 边界训练数据大小

    R['optimizer_name'] = 'Adam'                          # 优化器
    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['train_model'] = 'union_training'
    # R['train_model'] = 'group2_training'
    # R['train_model'] = 'group3_training'

    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    R['penalty2weight_biases'] = 0.000                    # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                 # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2bd_increase'] = 1
    # R['init_boundary_penalty'] = 1000                   # Regularization parameter for boundary conditions
    R['init_boundary_penalty'] = 100                      # Regularization parameter for boundary conditions

    # 网络的频率范围设置
    R['freq'] = np.arange(1, 121)

    # R['freq'] = np.concatenate((np.random.normal(0, 1, 30), np.random.normal(0, 20, 30),
    #                              np.random.normal(0, 50, 30), np.random.normal(0, 120, 30)), axis=0)

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    R['model2NN'] = 'DNN_scale'
    # R['model2NN'] = 'DNN_adapt_scale'
    # R['model2NN'] = 'DNN_FourierBase'
    # R['model2NN'] = 'DNN_WaveletBase'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'DNN_FourierBase':
        if R['order2pLaplace_operator'] == 2:
            if R['epsilon'] == 0.1:
                R['hidden_layers'] = (125, 100, 80, 80, 60)  # 1*125+250*100+100*80+80*80+80*60+60*1= 44385 个参数
            else:
                R['hidden_layers'] = (125, 100, 80, 80, 60)  # 1*125+250*100+100*80+80*80+80*60+60*1= 44385 个参数
        elif R['order2pLaplace_operator'] == 5:
            if R['epsilon'] == 0.1:
                R['hidden_layers'] = (125, 100, 80, 80, 60)  # 1*125+250*120+120*80+80*80+80*80+80*1= 52605 个参数
            else:
                R['hidden_layers'] = (125, 100, 80, 80, 60)  # 1*125+250*120+120*80+80*80+80*80+80*1= 52605 个参数
        elif R['order2pLaplace_operator'] == 8:
            if R['epsilon'] == 0.1:
                R['hidden_layers'] = (125, 100, 80, 80, 60)  # 1*125+250*150+150*100+100*100+100*80+80*1= 70705 个参数
            else:
                R['hidden_layers'] = (125, 100, 80, 80, 60)  # 1*125+250*150+150*100+100*100+100*80+80*1= 70705 个参数
        else:
            R['hidden_layers'] = (125, 100, 80, 80, 60)

        if R['equa_name'] == '3scale2':
            R['hidden_layers'] = (175, 300, 200, 200, 100)  # 172775
    else:
        if R['order2pLaplace_operator'] == 2:
            if R['epsilon'] == 0.1:
                R['hidden_layers'] = (250, 100, 80, 80, 60)  # 1*250+250*100+100*80+80*80+80*60+60*1= 44510 个参数
            else:
                R['hidden_layers'] = (250, 100, 80, 80, 60)  # 1*250+250*100+100*80+80*80+80*60+60*1= 44510 个参数
        elif R['order2pLaplace_operator'] == 5:
            if R['epsilon'] == 0.1:
                R['hidden_layers'] = (250, 100, 80, 80, 60)  # 1*250+250*120+120*80+80*80+80*80+80*1= 52730 个参数
            else:
                R['hidden_layers'] = (250, 100, 80, 80, 60)  # 1*250+250*120+120*80+80*80+80*80+80*1= 52730 个参数
        elif R['order2pLaplace_operator'] == 8:
            if R['epsilon'] == 0.1:
                R['hidden_layers'] = (250, 100, 80, 80, 60)  # 1*250+250*150+150*100+100*100+100*80+80*1= 70830 个参数
            else:
                R['hidden_layers'] = (250, 100, 80, 80, 60)  # 1*250+250*150+150*100+100*100+100*80+80*1= 70830 个参数
        else:
            R['hidden_layers'] = (250, 100, 80, 80, 60)

        if R['equa_name'] == '3scale2':
            R['hidden_layers'] = (350, 300, 200, 200, 100)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 's2relu'
    # R['name2act_in'] = 'gelu'
    # R['name2act_in'] = 'sin'
    R['name2act_in'] = 'sinADDcos'

    # R['name2act_hidden'] = 'relu'
    # R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    # R['name2act_hidden'] = 'sin'
    R['name2act_hidden'] = 'sinADDcos'
    # R['name2act_hidden'] = 'elu'
    # R['name2act_hidden'] = 'gelu'
    # R['name2act_hidden'] = 'phi'

    R['name2act_out'] = 'linear'

    if R['model2NN'] == 'DNN_FourierBase' and R['name2act_hidden'] == 'tanh':
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'DNN_FourierBase' and R['name2act_hidden'] == 's2relu':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'DNN_FourierBase' and R['name2act_hidden'] == 'gelu':
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'DNN_FourierBase' and R['name2act_hidden'] == 'sinADDcos':
        # R['sfourier'] = 1.0
        R['sfourier'] = 0.5
    else:
        R['sfourier'] = 1.0

    solve_Multiscale_PDE(R)

#     对于FourierBased_DNN, [sin;cos] + s2relu选择要比其他激活函数的选择好很多, 且sFourier=0.5要比sFourier=1.0好些

