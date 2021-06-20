import DNN_tools


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)
    if R_dic['PDE_type'] == 'pLaplace' or R_dic['PDE_type'] == 'Possion_Boltzmann':
        DNN_tools.log_string('The order of pLaplace operator: %s\n' % (R_dic['order2pLaplace_operator']), log_fileout)
        DNN_tools.log_string('The epsilon to pLaplace operator: %f\n' % (R_dic['epsilon']), log_fileout)

    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('Activate function for network: %s\n' % str(R_dic['activate_func']), log_fileout)
    if R_dic['model'] == 'DNN_FourierBase':
        DNN_tools.log_string('The frequency to neural network: %s\n' % (R_dic['freqs']), log_fileout)

    if R_dic['model'] == 'DNN_FourierBase' and R_dic['activate_func'] == 'tanh':
        DNN_tools.log_string('The scale-factor to fourier basis: %s\n' % (R_dic['sfourier']), log_fileout)

    DNN_tools.log_string('hidden layer:%s\n' % str(R_dic['hidden_layers']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    if R_dic['loss_type'] == 'variational_loss':
        DNN_tools.log_string('Loss function: variational loss\n', log_fileout)
    else:
        DNN_tools.log_string('Loss function: L2 loss\n', log_fileout)

    if (R_dic['train_model']) == 'union_training':
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss', log_fileout)
    elif (R_dic['train_model']) == 'group3_training':
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss + loss_it + loss_bd', log_fileout)
    elif (R_dic['train_model']) == 'group2_training':
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss + loss_bd', log_fileout)

    DNN_tools.log_string('Batch-size 2 interior: %s\n' % str(R_dic['batch_size2interior']), log_fileout)
    DNN_tools.log_string('Batch-size 2 boundary: %s\n' % str(R_dic['batch_size2boundary']), log_fileout)

    DNN_tools.log_string('Initial boundary penalty: %s\n' % str(R_dic['init_boundary_penalty']), log_fileout)
    if R_dic['activate_penalty2bd_increase'] == 1:
        DNN_tools.log_string('The penalty of boundary will increase with training going on.\n', log_fileout)
    elif R_dic['activate_penalty2bd_increase'] == 2:
        DNN_tools.log_string('The penalty of boundary will decrease with training going on.\n', log_fileout)
    else:
        DNN_tools.log_string('The penalty of boundary will keep unchanged with training going on.\n', log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)
