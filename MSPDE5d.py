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
import General_Laplace
import Load_data2Mat
import saveData
import plotData
import DNN_Log_Print


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s%s.txt' % ('log2', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary_out2file(R, log_fileout)

    # 一般 laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['init_boundary_penalty']                # Regularization parameter for boundary conditions
    penalty2WB = R['penalty2weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    hidden_layers = R['hidden_layers']

    input_dim = R['input_dim']
    out_dim = R['output_dim']
    act_func = R['name2act_hidden']

    # p laplace 问题需要的额外设置, 先预设一下
    p_index = R['order2pLaplace_operator']
    mesh_number = 2

    region_lb = 0.0
    region_rt = 1.0
    if R['PDE_type'] == 'general_Laplace':
        # -laplace u = f
        region_lb = 0.0
        region_rt = 1.0
        f, u_true, u00, u01, u10, u11, u20, u21, u30, u31, u40, u41 = General_Laplace.get_infos2Laplace_5D(
            input_dim=input_dim, out_dim=out_dim, intervalL=region_lb, intervalR=region_rt, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace':
        region_lb = 0.0
        region_rt = 1.0
        u_true, f, A_eps, u00, u01, u10, u11, u20, u21, u30, u31, u40, u41 = MS_LaplaceEqs.get_infos2pLaplace_5D(
            input_dim=input_dim, out_dim=out_dim, intervalL=0.0, intervalR=1.0, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'Possion_Boltzmann':
        region_lb = 0.0
        region_rt = 1.0
        u_true, f, A_eps, kappa, u00, u01, u10, u11, u20, u21, u30, u31, u40, u41 = MS_BoltzmannEqs.get_infos2Boltzmann_5D(
            input_dim=input_dim, out_dim=out_dim, intervalL=0.0, intervalR=1.0, equa_name=R['equa_name'])

    flag = 'WB2NN'
    if R['model2NN'] == 'DNN_FourierBase':
        W2NN, B2NN = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_layers, flag)
    else:
        W2NN, B2NN = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_layers, flag)

    global_steps = tf.compat.v1.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            XYZST_it = tf.compat.v1.placeholder(tf.float32, name='XYZST_it', shape=[None, input_dim])
            XYZST00 = tf.compat.v1.placeholder(tf.float32, name='XYZST00', shape=[None, input_dim])
            XYZST01 = tf.compat.v1.placeholder(tf.float32, name='XYZST01', shape=[None, input_dim])
            XYZST10 = tf.compat.v1.placeholder(tf.float32, name='XYZST10', shape=[None, input_dim])
            XYZST11 = tf.compat.v1.placeholder(tf.float32, name='XYZST11', shape=[None, input_dim])
            XYZST20 = tf.compat.v1.placeholder(tf.float32, name='XYZST20', shape=[None, input_dim])
            XYZST21 = tf.compat.v1.placeholder(tf.float32, name='XYZST21', shape=[None, input_dim])
            XYZST30 = tf.compat.v1.placeholder(tf.float32, name='XYZST30', shape=[None, input_dim])
            XYZST31 = tf.compat.v1.placeholder(tf.float32, name='XYZST31', shape=[None, input_dim])
            XYZST40 = tf.compat.v1.placeholder(tf.float32, name='XYZST40', shape=[None, input_dim])
            XYZST41 = tf.compat.v1.placeholder(tf.float32, name='XYZST41', shape=[None, input_dim])
            boundary_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            # 供选择的网络模式
            if R['model2NN'] == 'DNN':
                UNN = DNN_base.DNN(XYZST_it, W2NN, B2NN, hidden_layers, activateIn_name=R['name2act_in'],
                                   activate_name=R['name2act_hidden'])
                U00_NN = DNN_base.DNN(XYZST00, W2NN, B2NN, hidden_layers, activateIn_name=R['name2act_in'],
                                      activate_name=R['name2act_hidden'])
                U01_NN = DNN_base.DNN(XYZST01, W2NN, B2NN, hidden_layers, activateIn_name=R['name2act_in'],
                                      activate_name=R['name2act_hidden'])
                U10_NN = DNN_base.DNN(XYZST10, W2NN, B2NN, hidden_layers, activateIn_name=R['name2act_in'],
                                      activate_name=R['name2act_hidden'])
                U11_NN = DNN_base.DNN(XYZST11, W2NN, B2NN, hidden_layers, activateIn_name=R['name2act_in'],
                                      activate_name=R['name2act_hidden'])
                U20_NN = DNN_base.DNN(XYZST20, W2NN, B2NN, hidden_layers, activateIn_name=R['name2act_in'],
                                      activate_name=R['name2act_hidden'])
                U21_NN = DNN_base.DNN(XYZST21, W2NN, B2NN, hidden_layers, activateIn_name=R['name2act_in'],
                                      activate_name=R['name2act_hidden'])
                U30_NN = DNN_base.DNN(XYZST30, W2NN, B2NN, hidden_layers, activateIn_name=R['name2act_in'],
                                      activate_name=R['name2act_hidden'])
                U31_NN = DNN_base.DNN(XYZST31, W2NN, B2NN, hidden_layers, activateIn_name=R['name2act_in'],
                                      activate_name=R['name2act_hidden'])
                U40_NN = DNN_base.DNN(XYZST40, W2NN, B2NN, hidden_layers, activateIn_name=R['name2act_in'],
                                      activate_name=R['name2act_hidden'])
                U41_NN = DNN_base.DNN(XYZST41, W2NN, B2NN, hidden_layers, activateIn_name=R['name2act_in'],
                                      activate_name=R['name2act_hidden'])
            elif R['model2NN'] == 'DNN_scale':
                UNN = DNN_base.DNN_scale(XYZST_it, W2NN, B2NN, hidden_layers, R['freq'],
                                         activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U00_NN = DNN_base.DNN_scale(XYZST00, W2NN, B2NN, hidden_layers, R['freq'],
                                            activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U01_NN = DNN_base.DNN_scale(XYZST01, W2NN, B2NN, hidden_layers, R['freq'],
                                            activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U10_NN = DNN_base.DNN_scale(XYZST10, W2NN, B2NN, hidden_layers, R['freq'],
                                            activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U11_NN = DNN_base.DNN_scale(XYZST11, W2NN, B2NN, hidden_layers, R['freq'],
                                            activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U20_NN = DNN_base.DNN_scale(XYZST20, W2NN, B2NN, hidden_layers, R['freq'],
                                            activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U21_NN = DNN_base.DNN_scale(XYZST21, W2NN, B2NN, hidden_layers, R['freq'],
                                            activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U30_NN = DNN_base.DNN_scale(XYZST30, W2NN, B2NN, hidden_layers, R['freq'],
                                            activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U31_NN = DNN_base.DNN_scale(XYZST31, W2NN, B2NN, hidden_layers, R['freq'],
                                            activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U40_NN = DNN_base.DNN_scale(XYZST40, W2NN, B2NN, hidden_layers, R['freq'],
                                            activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U41_NN = DNN_base.DNN_scale(XYZST41, W2NN, B2NN, hidden_layers, R['freq'],
                                            activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
            elif R['model2NN'] == 'DNN_adapt_scale':
                UNN = DNN_base.DNN_adapt_scale(XYZST_it, W2NN, B2NN, hidden_layers, R['freq'],
                                               activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U00_NN = DNN_base.DNN_adapt_scale(XYZST00, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U01_NN = DNN_base.DNN_adapt_scale(XYZST01, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U10_NN = DNN_base.DNN_adapt_scale(XYZST10, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U11_NN = DNN_base.DNN_adapt_scale(XYZST11, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U20_NN = DNN_base.DNN_adapt_scale(XYZST20, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U21_NN = DNN_base.DNN_adapt_scale(XYZST21, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U30_NN = DNN_base.DNN_adapt_scale(XYZST30, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U31_NN = DNN_base.DNN_adapt_scale(XYZST31, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U40_NN = DNN_base.DNN_adapt_scale(XYZST40, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
                U41_NN = DNN_base.DNN_adapt_scale(XYZST41, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activateIn_name=R['name2act_in'], activate_name=R['name2act_hidden'])
            elif R['model2NN'] == 'DNN_FourierBase':
                UNN = DNN_base.DNN_FourierBase(XYZST_it, W2NN, B2NN, hidden_layers, R['freq'],
                                               activate_name=R['name2act_hidden'], sFourier=R['sfourier'])
                U00_NN = DNN_base.DNN_FourierBase(XYZST00, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activate_name=R['name2act_hidden'], sFourier=R['sfourier'])
                U01_NN = DNN_base.DNN_FourierBase(XYZST01, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activate_name=R['name2act_hidden'], sFourier=R['sfourier'])
                U10_NN = DNN_base.DNN_FourierBase(XYZST10, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activate_name=R['name2act_hidden'], sFourier=R['sfourier'])
                U11_NN = DNN_base.DNN_FourierBase(XYZST11, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activate_name=R['name2act_hidden'], sFourier=R['sfourier'])
                U20_NN = DNN_base.DNN_FourierBase(XYZST20, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activate_name=R['name2act_hidden'], sFourier=R['sfourier'])
                U21_NN = DNN_base.DNN_FourierBase(XYZST21, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activate_name=R['name2act_hidden'], sFourier=R['sfourier'])
                U30_NN = DNN_base.DNN_FourierBase(XYZST30, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activate_name=R['name2act_hidden'], sFourier=R['sfourier'])
                U31_NN = DNN_base.DNN_FourierBase(XYZST31, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activate_name=R['name2act_hidden'], sFourier=R['sfourier'])
                U40_NN = DNN_base.DNN_FourierBase(XYZST40, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activate_name=R['name2act_hidden'], sFourier=R['sfourier'])
                U41_NN = DNN_base.DNN_FourierBase(XYZST41, W2NN, B2NN, hidden_layers, R['freq'],
                                                  activate_name=R['name2act_hidden'], sFourier=R['sfourier'])

            X_it = tf.reshape(XYZST_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XYZST_it[:, 1], shape=[-1, 1])
            Z_it = tf.reshape(XYZST_it[:, 2], shape=[-1, 1])
            S_it = tf.reshape(XYZST_it[:, 3], shape=[-1, 1])
            T_it = tf.reshape(XYZST_it[:, 4], shape=[-1, 1])

            dUNN = tf.gradients(UNN, XYZST_it)[0]  # * 行 2 列

            # 变分形式的loss of interior，训练得到的 UNN 是 * 行 1 列, 因为 一个点对(x,y) 得到一个 u 值
            if R['loss_type'] == 'variational_loss':
                dUNN_Norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])  # 按行求和
                if R['PDE_type'] == 'general_Laplace':
                    dUNN_2Norm = tf.square(dUNN_Norm)
                    loss_it_variational = (1.0 / 2) *dUNN_2Norm - tf.multiply(f(X_it, Y_it, Z_it, S_it, T_it), UNN)
                elif R['PDE_type'] == 'pLaplace':
                    a_eps = A_eps(X_it, Y_it, Z_it, S_it, T_it)                          # * 行 1 列
                    AdUNN_pNorm = a_eps*tf.pow(dUNN_Norm, p_index)
                    if R['equa_name'] == 'multi_scale5D_4' or R['equa_name'] == 'multi_scale5D_5' or \
                            R['equa_name'] == 'multi_scale5D_6' or R['equa_name'] == 'multi_scale5D_7':
                        fxyzst = MS_LaplaceEqs.get_forceSide2pLaplace5D(x=X_it, y=Y_it, z=Z_it, s=S_it, t=T_it, equa_name=R['equa_name'])
                        loss_it_variational = (1.0 / p_index) * AdUNN_pNorm - tf.multiply(tf.reshape(fxyzst, shape=[-1, 1]), UNN)
                    else:
                        loss_it_variational = (1.0 / p_index) * AdUNN_pNorm - tf.multiply(f(X_it, Y_it, Z_it, S_it, T_it), UNN)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it, Y_it, Z_it, S_it, T_it)                          # * 行 1 列
                    AdUNN_pNorm = a_eps*tf.pow(dUNN_Norm, p_index)
                    Kappa = np.pi*np.pi
                    # Kappa = kappa(X_it, Y_it, Z_it, S_it, T_it)
                    if R['equa_name'] == 'multi_scale5D_4' or R['equa_name'] == 'multi_scale5D_5' or \
                            R['equa_name'] == 'multi_scale5D_6' or R['equa_name'] == 'multi_scale5D_7':
                        fxyzst = MS_BoltzmannEqs.get_forceSide2Boltzmann_5D(x=X_it, y=Y_it, z=Z_it, s=S_it, t=T_it,
                                                                            equa_name=R['equa_name'])
                        loss_it_variational = (1.0 / p_index)*(AdUNN_pNorm + Kappa*UNN*UNN) - tf.multiply(fxyzst, UNN)
                    else:
                        loss_it_variational = (1.0 / p_index) * (AdUNN_pNorm + Kappa*UNN*UNN) - \
                                              tf.multiply(f(X_it, Y_it, Z_it, S_it, T_it), UNN)
                loss_it = tf.reduce_mean(loss_it_variational)

            U_00 = tf.constant(0.0)
            U_01 = tf.constant(0.0)
            U_10 = tf.constant(0.0)
            U_11 = tf.constant(0.0)
            U_20 = tf.constant(0.0)
            U_21 = tf.constant(0.0)
            U_30 = tf.constant(0.0)
            U_31 = tf.constant(0.0)
            U_40 = tf.constant(0.0)
            U_41 = tf.constant(0.0)

            loss_bd_square2NN = tf.square(U00_NN - U_00) + tf.square(U01_NN - U_01) + tf.square(U10_NN - U_10) + \
                                tf.square(U11_NN - U_11) + tf.square(U20_NN - U_20) + tf.square(U21_NN - U_21) + \
                                tf.square(U30_NN - U_30) + tf.square(U31_NN - U_31) + tf.square(U40_NN - U_40) + \
                                tf.square(U41_NN - U_41)
            loss_bd = tf.reduce_mean(loss_bd_square2NN)

            if R['regular_wb_model'] == 'L1':
                regularSum2WB = DNN_base.regular_weights_biases_L1(W2NN, B2NN)    # 正则化权重和偏置 L1正则化
            elif R['regular_wb_model'] == 'L2':
                regularSum2WB = DNN_base.regular_weights_biases_L2(W2NN, B2NN)    # 正则化权重和偏置 L2正则化
            else:
                regularSum2WB = tf.constant(0.0)                                        # 无正则化权重参数

            PWB = penalty2WB * regularSum2WB
            loss = loss_it + boundary_penalty * loss_bd + PWB      # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            if R['train_model'] == 'group3_training':
                train_op1 = my_optimizer.minimize(loss_it, global_step=global_steps)
                train_op2 = my_optimizer.minimize(loss_bd, global_step=global_steps)
                train_op3 = my_optimizer.minimize(loss, global_step=global_steps)
                train_my_loss = tf.group(train_op1, train_op2, train_op3)
            elif R['train_model'] == 'group2_training':
                train_op2bd = my_optimizer.minimize(loss_bd, global_step=global_steps)
                train_op2union = my_optimizer.minimize(loss, global_step=global_steps)
                train_my_loss = tf.group(train_op2union, train_op2bd)
            elif R['train_model'] == 'union_training':
                train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

            if R['PDE_type'] == 'general_Laplace' or R['PDE_type'] == 'pLaplace' or R['PDE_type'] == 'Possion_Boltzmann':
                # 训练上的真解值和训练结果的误差
                U_true = u_true(X_it, Y_it, Z_it, S_it, T_it)
                train_mse = tf.reduce_mean(tf.square(U_true - UNN))
                train_rel = train_mse / tf.reduce_mean(tf.square(U_true))
            else:
                train_mse = tf.constant(0.0)
                train_rel = tf.constant(0.0)

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    # 画网格解图
    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        # test_bach_size = 400
        # size2test = 20
        # test_bach_size = 900
        # size2test = 30
        test_bach_size = 1600
        size2test = 40
        # test_bach_size = 4900
        # size2test = 70
        # test_bach_size = 10000
        # size2test = 100
        test_xyzst_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
        saveData.save_testData_or_solus2mat(test_xyzst_bach, dataName='testXYZST', outPath=R['FolderName'])
    elif R['testData_model'] == 'loadData':
        test_bach_size = 1600
        size2test = 40
        mat_data_path = 'dataMat_highDim'
        test_xyzst_bach = Load_data2Mat.get_randomData2mat(dim=input_dim, data_path=mat_data_path)
        saveData.save_testData_or_solus2mat(test_xyzst_bach, dataName='testXYZST', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            xyzst_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            xyzst00_batch, xyzst01_batch, xyzst10_batch, xyzst11_batch, xyzst20_batch, xyzst21_batch, xyzst30_batch, \
            xyzst31_batch, xyzst40_batch, xyzst41_batch = DNN_data.rand_bd_5D(batchsize_bd, input_dim,
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
                feed_dict={XYZST_it: xyzst_it_batch, XYZST00: xyzst00_batch, XYZST01: xyzst01_batch,
                           XYZST10: xyzst10_batch, XYZST11: xyzst11_batch, XYZST20: xyzst20_batch,
                           XYZST21: xyzst21_batch, XYZST30: xyzst30_batch, XYZST31: xyzst31_batch,
                           XYZST40: xyzst40_batch, XYZST41: xyzst41_batch, in_learning_rate: tmp_lr,
                           boundary_penalty: temp_penalty_bd})

            loss_it_all.append(loss_it_tmp)
            loss_bd_all.append(loss_bd_tmp)
            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_rel_tmp)

            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                DNN_Log_Print.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, pwb, loss_it_tmp, loss_bd_tmp, loss_tmp, train_mse_tmp,
                    train_rel_tmp, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                if R['PDE_type'] == 'general_laplace' or R['PDE_type'] == 'pLaplace' or R['PDE_type'] == 'Possion_Boltzmann':
                    u_true2test, u_nn2test = sess.run(
                        [U_true, UNN], feed_dict={XYZST_it: test_xyzst_bach})
                else:
                    u_true2test = u_true
                    u_nn2test = sess.run(UNN,  feed_dict={XYZST_it: test_xyzst_bach})

                point_square_error = np.square(u_true2test - u_nn2test)
                mse2test = np.mean(point_square_error)
                test_mse_all.append(mse2test)
                res2test = mse2test / np.mean(np.square(u_true2test))
                test_rel_all.append(res2test)

                DNN_Log_Print.print_and_log_test_one_epoch(mse2test, res2test, log_out=log_fileout)

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
        saveData.save_2testSolus2mat(u_true2test, u_nn2test, actName='utrue', actName1=act_func, outPath=R['FolderName'])

        # 绘制解的热力图(真解和DNN解)
        plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue', seedNo=R['seed'],
                                        outPath=R['FolderName'])
        plotData.plot_Hot_solution2test(u_nn2test, size_vec2mat=size2test, actName=act_func, seedNo=R['seed'],
                                        outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act_func,
                                  seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

        saveData.save_test_point_wise_err2mat(point_square_error, actName=act_func, outPath=R['FolderName'])

        plotData.plot_Hot_point_wise_err(point_square_error, size_vec2mat=size2test, actName=act_func,
                                         seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
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
    # store_file = 'Laplace5D'
    store_file = 'pLaplace5D'
    # store_file = 'Boltzmann5D'
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
    R['input_dim'] = 5  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    if store_file == 'Laplace5D':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace5D':
        R['PDE_type'] = 'pLaplace'
        # R['equa_name'] = 'multi_scale5D_1'          # general laplace
        # R['equa_name'] = 'multi_scale5D_2'            # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_3'  # multi-scale laplace
        R['equa_name'] = 'multi_scale5D_4'  # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_5'  # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_6'  # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_7'  # multi-scale laplace
    elif store_file == 'Boltzmann5D':
        R['PDE_type'] = 'Possion_Boltzmann'
        # R['equa_name'] = 'multi_scale5D_4'
        # R['equa_name'] = 'multi_scale5D_5'
        # R['equa_name'] = 'multi_scale5D_6'
        R['equa_name'] = 'multi_scale5D_7'

    if R['PDE_type'] == 'general_Laplace':
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
        R['batch_size2interior'] = 12500          # 内部训练数据的批大小
        # R['batch_size2interior'] = 10000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 2000
    elif R['PDE_type'] == 'pLaplace'or R['PDE_type'] == 'Possion_Boltzmann':
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
        R['batch_size2interior'] = 12500  # 内部训练数据的批大小
        # R['batch_size2interior'] = 10000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 2000

    # ---------------------------- Setup of DNN -------------------------------
    # 装载测试数据模式
    R['testData_model'] = 'loadData'
    # R['testData_model'] = 'random_generate'

    # R['loss_type'] = 'L2_loss'                             # loss类型:L2 loss
    R['loss_type'] = 'variational_loss'          # loss类型:PDE变分

    R['optimizer_name'] = 'Adam'                 # 优化器
    R['learning_rate'] = 2e-4                    # 学习率
    R['learning_rate_decay'] = 5e-5              # 学习率 decay
    R['train_model'] = 'union_training'
    # R['train_model'] = 'group2_training'
    # R['train_model'] = 'group3_training'

    # 正则化权重和偏置的模式
    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    R['penalty2weight_biases'] = 0.000                      # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001                    # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                   # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2bd_increase'] = 1
    # R['init_boundary_penalty'] = 1000  # Regularization parameter for boundary conditions
    R['init_boundary_penalty'] = 100  # Regularization parameter for boundary conditions

    # 网络的频率范围设置
    # R['freq'] = np.arange(1, 121)
    R['freq'] = np.concatenate((np.random.normal(0, 1, 30), np.random.normal(0, 20, 30),
                                 np.random.normal(0, 50, 30), np.random.normal(0, 100, 30)), axis=0)

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'DNN_scale'
    # R['model2NN'] = 'DNN_adapt_scale'
    R['model2NN'] = 'DNN_FourierBase'
    # R['model2NN'] = 'DNN_Sin+Cos_Base'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'DNN_FourierBase':
        R['hidden_layers'] = (250, 400, 400, 300, 300, 200)  # 250+500*400+400*400+400*300+300*300+300*200+200=630450
    else:
        # R['hidden_layers'] = (100, 10, 8, 6, 4)  # 测试
        # R['hidden_layers'] = (100, 80, 60, 60, 40, 40, 20)
        # R['hidden_layers'] = (200, 100, 80, 50, 30)
        # R['hidden_layers'] = (250, 400, 400, 300, 300, 200)  # 250+500*400+400*400+400*300+300*300+300*200+200=630450
        R['hidden_layers'] = (500, 400, 400, 300, 300, 200)  # 500+500*400+400*400+400*300+300*300+300*200+200=630700
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 200, 200, 100)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['name2act_in'] = 'relu'
    R['name2act_in'] = 'tanh'

    # R['name2act_hidden'] = 'relu'
    R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden']' = leaky_relu'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    # R['name2act_hidden'] = 'scsrelu'
    # R['name2act_hidden'] = 'sin'
    # R['name2act_hidden'] = 'sinAddcos'
    # R['name2act_hidden'] = 'elu'
    # R['name2act_hidden'] = 'phi'

    R['name2act_out'] = 'linear'

    if R['model2NN'] == 'DNN_FourierBase' and R['name2act_hidden'] == 'tanh':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'DNN_FourierBase' and R['name2act_hidden'] == 's2relu':
        R['sfourier'] = 0.5
    else:
        R['sfourier'] = 1.0

    solve_Multiscale_PDE(R)

