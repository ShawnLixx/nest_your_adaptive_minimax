#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:00:00 2018

@author: fanxiao
"""
#%%
import os
import time
import argparse
import copy

import matplotlib.pyplot as plt
# from scipy.misc import imresize
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

USE_CUDA =  torch.cuda.is_available()

import tensorflow as tf


def get_gradient_norm(params, norm_type=2.0):
    with torch.no_grad():
        total_norm = torch.norm(torch.stack(
            [torch.norm(
                p.grad.detach(), norm_type) \
                        for p in params]), norm_type)
    return total_norm

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
#%%

class MLP(nn.Module) :
    def __init__(self, activation='relu'):
        super(MLP, self).__init__()
        
        self.linear1 = nn.Linear(2,4) #input dimension:2
        self.linear2 = nn.Linear(4,2)
        self.linear3 = nn.Linear(2,2)
        if activation == 'relu':
            self.active = nn.ReLU() 
        else :
            self.active = nn.ELU()
    
    def forward(self,input):
        x = self.active(self.linear1(input))
        x = self.active(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def init_weights_glorot(self):
        for m in self._modules :
            if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight)
                
def adjust_lr(optimizer, lr0, epoch, total_epochs):
    lr = lr0 * (0.1 ** (epoch / float(total_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#adjust learning rate when maximizing z_hat : alpha_t = 1/np.sqrt(t)
def adjust_lr_zt(optimizer, lr0, epoch):
    lr = lr0 * (1.0 / np.sqrt(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def evaluate (model, valid_data) :
    model.eval()
    COUNTER = 0
    ACCURACY = 0
    for x_, y_ in valid_data :
        if USE_CUDA:
            x_, y_ = x_.cuda(), y_.cuda()
        x_, y_ = Variable(x_), Variable(y_)
        out = model(x_)
        _, predicted = torch.max(out, 1)
        COUNTER += y_.size(0)
        ACCURACY += float(torch.eq(predicted,y_).sum().cpu().data.numpy())
    return ACCURACY / float(COUNTER) *100.0

# deprecated
# evaluate on adversarial examples
def evaluate_adversarial (model, loss_function, valid_data, epsilon=0.05) :
    model.eval()
    COUNTER = 0
    ACCURACY = 0
    for x_, y_ in valid_data :
        if USE_CUDA:
            x_, y_ = x_.cuda(), y_.cuda()
        x_, y_ = Variable(x_,requires_grad=True), Variable(y_)
        loss_true = loss_function(model(x_),y_)
        loss_true.backward()
        x_grad = x_.grad
        x_adversarial = x_.clone()
        if USE_CUDA:
            x_adversarial = x_adversarial.cuda()
        # x_adversarial.data = x_.data + epsilon * torch.sign(x_grad.data) * x_grad.data   
        x_adversarial.data = x_.data + epsilon * torch.sign(x_grad.data)      
        
        x_.grad.data.zero_()
        out = model(x_adversarial)
        _, predicted = torch.max(out, 1)
        COUNTER += y_.size(0)
        ACCURACY += float(torch.eq(predicted,y_).sum().cpu().data.numpy())
    return ACCURACY / float(COUNTER) *100.0


def train_WRM(model,optimizer,loss_function, train_loader,valid_loader, num_epoch,gamma=2,max_lr0=0.0001,min_lr0=0.001,min_lr_adjust=False,T_adv = 15, savepath=None, writer=None, args=None) :
    model.train()
    # T_adv = 15
    # train_hist={}
    # train_hist['loss']=[]
    # train_hist['loss_maxItr'] =[]
    # train_hist['acc_train']=[]
    # train_hist['acc_test']=[]
    # train_hist['ptime']=[]
    # train_hist['total_ptime']=[]

    steps = 0
    outer_steps = 0
    data_iter = iter(cycle(train_loader))
    # test after each epoch
    iter_per_epoch = len(train_loader)
    args.n_iter = iter_per_epoch * args.n_epoch
    num_record = args.n_epoch
    num_recorded = 0
    record_gap = args.n_iter // num_record
    while steps < args.n_iter:
        x_, y_ = next(data_iter)
        if USE_CUDA:
            x_, y_ = x_.cuda(), y_.cuda()
        # x_, y_ = Variable(x_), Variable(y_)
        
        #initialize z_hat with x_
        z_hat = x_.data.clone()
        if USE_CUDA:
            z_hat = z_hat.cuda()
        # z_hat = Variable(z_hat,requires_grad=True)
        z_hat.requires_grad_()
        
        #running the maximizer for z_hat
        optimizer_zt = torch.optim.Adam([z_hat], lr=max_lr0)
        loss_phi = 0 # phi(theta,z0)
        rho = 0 #E[c(Z,Z0)]
        inner_steps = 0
        # stopping criterion
        if args.our:
            required_err = args.stop_scale / (outer_steps + 1)
        while steps < args.n_iter:
            optimizer_zt.zero_grad()
            delta = z_hat - x_
            rho = torch.mean((torch.norm(delta.view(len(x_),-1),2,1)**2)) 
            # rho = torch.mean((torch.norm(z_hat-x_,2,1)**2))
            loss_zt=loss_function(model(z_hat),y_)
            #-phi_gamma(theta,z)
            loss_phi = - ( loss_zt - gamma * rho)
            loss_phi.backward()
            optimizer_zt.step()
            steps += 1
            # adjust_lr_zt(optimizer_zt,max_lr0, n+1)
            y_grad_norm = get_gradient_norm([z_hat]).item()
            if args.our:
                # both criterion
                if y_grad_norm ** 2 <= required_err:
                    break
                inner_steps += 1
                if inner_steps >= outer_steps:
                    break
            else:
                inner_steps += 1
                if inner_steps >= args.n_inner:
                    break
            
        # losses_maxItr.append(loss_phi.data[0]) #loss in max iteration phi(theta,z)
        # rhos.append(rho.data[0])

        # running the loss minimizer, using z_hat   
        optimizer.zero_grad()
        loss_adversarial = loss_function(model(z_hat),y_)
        loss_adversarial.backward()         
        optimizer.step()

        with torch.no_grad():
            delta = z_hat - x_
            rho = torch.mean((torch.norm(delta.view(len(x_),-1),2,1)**2)) 
            # rho = torch.mean((torch.norm(z_hat-x_,2,1)**2))
            loss_adversarial +=  -gamma * rho
            # losses_adv.append(loss_adversarial.data[0])

        outer_steps += 1
        steps += 1

        # record gradient norm
        x_grad_norm = get_gradient_norm(model.parameters()).item()
        with writer.as_default():
            tf.summary.scalar('x grad norm', x_grad_norm,
                    step=steps)
            tf.summary.scalar('y grad norm', y_grad_norm,
                    step=steps)
            tf.summary.scalar('x loss', loss_adversarial.item(),
                    step=steps)
            tf.summary.scalar('y loss', loss_phi.item(),
                    step=steps)

        if steps > num_recorded * record_gap:
            acc_train = evaluate(model,train_loader)

            adv_acc = evaluate_adversarial(model, loss_function, valid_loader, args.epsilon)

            # gradient norm of the primal function
            # primal_grad_norm = cal_worst_case(model, valid_loader, gamma, max_lr0)

            with writer.as_default():
                print(f"steps: {steps}")
                tf.summary.scalar('train acc', acc_train,
                        step=steps)
                tf.summary.scalar('test acc', adv_acc,
                        step=steps)
                # tf.summary.scalar('primal gradient norm', primal_grad_norm,
                #         step=steps)
            num_recorded += 1

            model.train()
            

        # if min_lr_adjust == True:
        #     adjust_lr(optimizer,min_lr0,ep,num_epoch) 

        # display and save
        # mean_loss_adv=torch.mean(torch.FloatTensor(losses_adv)) # E(l(theta,z))
        # mean_loss_maxItr=torch.mean(torch.FloatTensor(losses_maxItr)) #E[phi(theta,z)]
        # mean_loss_rho = torch.mean(torch.FloatTensor(rhos)) #E[c(Z,Z0)]
        # acc_train = evaluate(model,train_loader)
        # acc_test = evaluate(model,valid_loader)
        # epoch_end_time = time.time()
        # per_epoch_ptime = epoch_end_time - epoch_start_time
        # train_hist['loss'].append(mean_loss_adv)
        # train_hist['loss_maxItr'].append(-mean_loss_maxItr) #negative since minimize -loss
        # train_hist['acc_train'].append(acc_train)
        # train_hist['acc_test'].append(acc_test)
        # train_hist['ptime'].append(per_epoch_ptime)
        # print ('epoch %d , loss %.3f , acc train %.2f%% , acc test %.2f%%, phi %.3f,  rho %.3f, ptime %.2fs .' \
        #     %(ep, mean_loss_adv, acc_train, acc_test, -mean_loss_maxItr, mean_loss_rho, per_epoch_ptime))
        
        # save
        # if savepath is not None: #and (ep % 3==0):
        #     end_time = time.time()
        #     total_ptime = end_time - start_time
        #     train_hist['total_ptime'].append(total_ptime)
        #     saveCheckpoint(model,train_hist,savepath+'_ep'+str(ep))

    # if args.model_name == 'test':
    #     # draw a curve about epsilon
    #     for i, epsilon in enumerate(np.linspace(0, 3, 30)):
    #         adv_acc = evaluate_adversarial(model, loss_function, valid_loader, epsilon)
    #
    #         adv_acc = adv_acc / 100
    #         adv_acc = np.log(1 - adv_acc)
    #         with summary_writer.as_default():
    #             tf.summary.scalar('adv test acc vs. epsilon', adv_acc,
    #                     step=i)
        
         
def saveCheckpoint(model,train_hist, filename='Model') :
    # print('Saving..')
    state = {
        'model':  model.cpu().state_dict() if USE_CUDA else model.state_dict(),
        'train_hist' : train_hist
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+filename)
    if USE_CUDA:
        model.cuda()

def loadCheckpoint(model,filename='Model',path='./checkpoint/'):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(path+filename)
    model_params = checkpoint['model']
    model.load_state_dict(model_params)
    if USE_CUDA :
        model.cuda()
    train_hist = checkpoint['train_hist']

    return model,train_hist   

def synthetic_data(N_example) : 
    data_x = np.zeros((N_example,2))
    data_y = np.ones(N_example) 
    length = 0 
    while(length<N_example) :
        x = np.random.randn(100,2)
        l2 = np.linalg.norm(x, axis=1)
        x = x[np.any((l2>=1.3*np.sqrt(2),l2<=np.sqrt(2)/1.3), axis=0), :]
        y = [1 if (np.linalg.norm(i) - np.sqrt(2)) > 0 else 0 for i in x]  
        if length+len(x) <= N_example :
            data_x[length:length+len(x),:] = x
            data_y[length:length+len(x)] = y
        else :
            data_x[length:,:] = x[N_example-length,:]
            data_y[length:] = y[N_example-length]
        length += len(x)
    # print('num of class0:',len(data_y[data_y==0]))
    return data_x, data_y

#%%
def plotGraph(models,data_x, data_y, labels) :
    
    fig = plt.figure(figsize=(5,5))
    Colors = ['blue','orange','red','purple','grey','green']

    plt.scatter(data_x[data_y==0,0],data_x[data_y==0,1], c=Colors[0], marker='.')
    plt.scatter(data_x[data_y==1,0],data_x[data_y==1,1], facecolors='none', edgecolors=Colors[1])
    xmax = max(data_x[:,0])
    xmin = min(data_x[:,0])
    ymax = max(data_x[:,1])
    ymin = min(data_x[:,1])
     
    x1 = np.linspace(xmin, xmax)
    x2 = np.linspace(ymin,ymax)
    X1,X2 = np.meshgrid(x1,x2)
    features_X = np.vstack((X1.flatten(),X2.flatten())).T
    
    softmax = nn.Softmax()
    
    for i, model in enumerate(models):
        Z = softmax(model(Variable(torch.FloatTensor(features_X))))[:,0].data.numpy().reshape(len(x1),len(x2))
        CS = plt.contour(X1,X2,Z,[0.5],colors=Colors[i+2])
        CS.collections[0].set_label(labels[i])
        
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.legend()
    return fig

def plotGraph_one(model,data_x, data_y, labels,index) :
    
    fig = plt.figure(figsize=(2,2))
    Colors = ['blue','orange','red','purple','grey','green']

    theta = np.linspace(0, 2*np.pi,800)  
    x,y = np.cos(theta)*1.3*np.sqrt(2), np.sin(theta)*1.3*np.sqrt(2)  
    plt.plot(x, y, color='black', linewidth=1.0)   
    x,y = np.cos(theta)*np.sqrt(2)/1.3, np.sin(theta)*np.sqrt(2)/1.3
    plt.plot(x, y, color='black', linewidth=1.0)
    
    xmax = max(data_x[:,0])
    xmin = min(data_x[:,0])
    ymax = max(data_x[:,1])
    ymin = min(data_x[:,1])
     
    x1 = np.linspace(xmin, xmax)
    x2 = np.linspace(ymin,ymax)
    X1,X2 = np.meshgrid(x1,x2)
    features_X = np.vstack((X1.flatten(),X2.flatten())).T
    
    softmax = nn.Softmax()
    
    # for i, model in enumerate(models):
    Z = softmax(model(Variable(torch.FloatTensor(features_X))))[:,0].data.numpy().reshape(len(x1),len(x2))
    CS = plt.contour(X1,X2,Z,[0.5],colors=Colors[index+2])
    CS.collections[0].set_label(labels[index])
        
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.legend()
    return fig

def init_seed(seed=123):
    '''set seed of random number generators'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)
        
# plot figure 2 certificate vs. worst case
def plot_certificate(model,loss_train,gamma,valid_data_loader,max_lr0) :
    fig = plt.figure()
    certificate=[] #E_train[phi(theta,z)] + gamma*rho
    list_rho = []
    list_worst = []
    for rho in range(0,65,5):
        rho = rho/100.0
        certificate.append(loss_train+gamma*rho)
        
    #test worst case 
    list_rho = []
    list_worst = []
    for g in range(1,500,5) :
        g=g/100.0
        rho, e = cal_worst_case(model,valid_data_loader, g, max_lr0)
        list_rho.append(rho)
        list_worst.append(e + rho * g)
    
    plt.plot(list_rho,list_worst, c='red', label=r"Test worst-case: $\sup_{P:W_c(P,\hat{P}_{test}) \leq \rho } E_P [l(\theta_{WRM};Z)]$")
    plt.plot(np.array(range(0,65,5))/100.0,certificate,c='blue', label=r"Certificate: $E_{\hat{P}_n}[\phi_{\gamma}(\theta_{WRM};Z)]+\gamma \rho$")
    #0.068=last rho of train
    plt.plot([0.068,0.068],[0.0,1.1], linestyle="--", c="black") 
    plt.xlabel(r"$\rho$")
    plt.xlim([0,0.65])
    plt.ylim([0,1.3])
    plt.legend(loc="lower right")
    return fig

def cal_worst_case(model, valid_data_loader, gamma, max_lr0) :
    model.eval()
    loss_maxItr = []
    T_adv=15
    rhos = []
    loss_function = nn.CrossEntropyLoss()
    # copid model to accumulate the gradient of primal function
    temp_model = copy.deepcopy(model)
    temp_model.zero_grad()
    num_iters = len(valid_data_loader)
    for x_, y_ in valid_data_loader :
        if USE_CUDA:
            x_, y_ = x_.cuda(), y_.cuda()
        # x_, y_ = Variable(x_), Variable(y_)
        
        #initialize z_hat with x_
        z_hat = x_.data.clone()
        if USE_CUDA:
            z_hat = z_hat.cuda()
        # z_hat = Variable(z_hat,requires_grad=True)
        z_hat.requires_grad_()
        
        #running the maximizer for z_hat
        optimizer_zt = torch.optim.Adam([z_hat], lr=max_lr0)
        loss_zt = 0
        rho = 0
        for n in range(T_adv) :
            optimizer_zt.zero_grad()
            model.zero_grad()
            delta = z_hat - x_
            rho = torch.mean((torch.norm(delta.view(len(x_),-1),2,1)**2)) 
#            rho = torch.mean((torch.norm(z_hat-x_,2,1)**2))
            loss_zt = - ( loss_function(model(z_hat),y_)- gamma * rho)
            loss_zt.backward()
            optimizer_zt.step()
            adjust_lr_zt(optimizer_zt,max_lr0, n+1)
            
        # running the loss minimizer, using z_hat   
        # no zero grad to accumulate the gradient
        # optimizer.zero_grad()
        loss_adversarial = (1 / num_iters) * loss_function(temp_model(z_hat),y_)
        loss_adversarial.backward()         
        # loss_maxItr.append(-loss_zt.data[0])
        # rhos.append(rho.data[0]) 
    primal_grad_norm = get_gradient_norm(temp_model.parameters()).item()
    # phi_test = torch.mean(torch.FloatTensor(loss_maxItr)) #E_test[phi(theta,z)]
    # rho_test = torch.mean(torch.FloatTensor(rhos))
    # return rho_test, phi_test
    return primal_grad_norm

#%%
init_seed()
train_x, train_y = synthetic_data(10000)
valid_x, valid_y = synthetic_data(4000)

path="/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/projet/"
# path = "C:\\Users\\lingyu.yue\\Documents\\Xiao_Fan\\project"
if os.path.isdir(path):
    os.chdir(path)
else:
    os.chdir("./")
print(os.getcwd())

#%%
if __name__=='__main__':
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=8, help='random seed')
    parser.add_argument('--n_epoch', type=int, default=100, help='number of gradient calls')
    parser.add_argument('--n_inner', type=int, default=15, help='number of inner loop steps')
    parser.add_argument('--lr_y', type=float, default=0.08, help='learning rate of y')
    parser.add_argument('--lr_x', type=float, default=0.01, help='learning rate of x')
    parser.add_argument('--gamma', type=float, default=1.3, help='gamma in the loss')
    parser.add_argument('--epsilon', type=float, default=0.5, help='noise level')
    parser.add_argument('--stop_scale', type=float, default=None, help='the inner loop stopping criterion')
    parser.add_argument('--model_name', type=str, default=None, help='model name to save')
    parser.add_argument('--our', action='store_true', help='use our NeAda algorithm')
    parser.set_defaults(our=False)
    args = parser.parse_args()

    
    # LR0 = 0.01
    batch_size = 128
    loss_function = nn.CrossEntropyLoss()

    train_data = torch.utils.data.TensorDataset(
            torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long())
    valid_data = torch.utils.data.TensorDataset(
            torch.from_numpy(valid_x).float(), torch.from_numpy(valid_y).long())
    train_data_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_data_loader = torch.utils.data.DataLoader(
                    valid_data, batch_size=batch_size, shuffle=True, num_workers=2)


    # Tensorboard
    if args.model_name is None:
        model_name = f'lry_{args.lr_y}_lrx_{args.lr_x}_gamma_{args.gamma}_eps_{args.epsilon}'
        if not args.our:
            model_name += f'_instep_{args.n_inner}'
        if args.our:
            model_name += '_our'
            if args.stop_scale is not None:
                model_name += f'_stop_{args.stop_scale}'
            else:
                args.stop_scale = 1
    else:
        model_name = args.model_name
    log_dir = os.path.join('logs', model_name)
    summary_writer = tf.summary.create_file_writer(log_dir)


    
    LR0 = args.lr_x # learning rate for classify
    MAX_LR0= args.lr_y # learning rate for maximize the perturbation, and for IFGM's step size also.
    GAMMA=args.gamma
    


    net_WRM = MLP(activation='elu')
    if USE_CUDA :
        net_WRM.cuda()
    net_WRM.init_weights_glorot()

    optimizer = torch.optim.Adam(net_WRM.parameters(), lr=LR0)
    train_WRM(net_WRM,optimizer,loss_function,
            train_data_loader,valid_data_loader, 30 ,GAMMA,  max_lr0=MAX_LR0,
            min_lr0=LR0, savepath='syn_WRM',
            writer=summary_writer, args=args)

    # rho = 0.072, epsilon = sqrt(rho) = 0.2683281572999748 (RELU)
    # rho = 0.070, epsilon = 0.265 (ELU)
    
#%%
# if __name__=='__main__':
#     labels = ['ERM','FGM','IFGM','WRM']
#     fig = plotGraph([net_ERM,net_FGM,net_IFGM,net_WRM],train_x, train_y, labels)
#     fig.savefig('fig1-elu.pdf')
#
# #%%
# if __name__=='__main__':
#     LR0 = 0.01
#     MAX_LR0=0.08
#     GAMMA=2
#     batch_size = 128
#     loss_function = nn.CrossEntropyLoss()
#
#     model = MLP('elu')
#     net_WRM, train_hist = loadCheckpoint(model,'syn_WRM_ep30')
#     fig = plot_certificate(net_WRM,train_hist['loss_maxItr'][-1],GAMMA,valid_data_loader,MAX_LR0)
#     fig.legend()
#     fig.savefig('fig2-syn.pdf')
#%%
#certificate=[] #E_train[phi(theta,z)] + gamma*rho
#list_rho = []
#list_worst = []
#for rho in range(0,65,5):
#    rho = rho/100.0
#    certificate.append(train_hist['loss_maxItr'][-1]+2.0*rho)
#
#for g in range(120,300,5) :
#    g=g/100.0
#    rho, e = cal_worst_case(net_WRM,valid_data_loader, g, 0.05)
#    print (rho, e+rho*g)
#    list_rho.append(rho)
#    list_worst.append(e + rho * g)
#
#plt.plot(list_rho,list_worst, c='red', label=r"Test worst-case: $\sup_{P:W_c(P,\hat{P}_{test}) \leq \rho } E_P [l(\theta_{WRM};Z)]$")
#plt.plot(np.array(range(0,65,5))/100.0,certificate,c='blue', label=r"Certificate: $E_{\hat{P}_n}[\phi_{\gamma}(\theta_{WRM};Z)]+\gamma \rho$")
#    
