
import os
import os.path as osp
import time
import random

import numpy as np
import torch

from parse_base import parse_arguments
import sklearn.model_selection


from data_factory.data_loader import get_loader_segment_base
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")
RUNNINGAVE_PARAM = 0.7
torch.backends.cudnn.benchmark = True


def save_model(args, aug_model, optimizer, epoch, itr, save_path):
    """
    save CTFP model's checkpoint during training

    Parameters:
        args: the arguments from parse_arguments in ctfp_tools
        aug_model: the CTFP Model
        optimizer: optimizer of CTFP model
        epoch: training epoch
        itr: training iteration
        save_path: path to save the model
    """
    torch.save(
        {
            "args": args,
            "state_dict": aug_model.module.state_dict()
            if torch.cuda.is_available() and not args.use_cpu
            else aug_model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "last_epoch": epoch,
            "iter": itr,
        },
        save_path,
    )


if __name__ == "__main__":
    args = parse_arguments()
    # logger
    manual_seed = args.seed
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    # if you are using GPU
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get deivce
    if args.use_cpu:
        device = torch.device("cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    data_path = args.data_path
    dataset = args.dataset
    train_loader = get_loader_segment(data_path, batch_size=args.batch_size, win_size=args.win_size, step = args.step_size,
                                               mode='train',
                                               dataset=dataset)
    
    test_loader = get_loader_segment(data_path, batch_size=args.batch_size, win_size=args.win_size,step = args.step_size,
                                              mode='test',
                                              dataset=dataset)
    
    # model build part 
    
    # optimizer part 
    
    # print("Num of Parameters: %d" % num_params)

    # GPU
    # if torch.cuda.is_available() and not args.use_cpu:
    #     aug_model = aug_model.cuda()
    #     atten_model = atten_model.cuda()
        

    
    # loss 
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    best_loss = float("inf")
    
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        # Train model
        # aug_model.train()
        # atten_model.train()
        accuracy_ = []
        total_loss = [] 
        nll_=[]
        pos_loss_ = []
        neg_loss_=[]
        recall_=[]
        f_score_=[]
        num_observes = []
        
        
        for temp_idx, x in enumerate(train_loader):
            ## x is a tuple of (values, times, stdv, masks)
            start = time.time()
            optimizer.zero_grad()
            
            x = map(cvt, x)
            values, times, vars, masks = x 
            #import pdb; pdb.set_trace()
    
            labels = torch.zeros([values.shape[0],values.shape[1]]).to(values.device)
            # run model

            loss_,labels= run_model(args, aug_model,atten_model, values,labels, times, vars, masks,mode='train')
            
            
            neg_label = labels # 1 0 1 0 
            pos_label =(labels<=0).to(masks.dtype) # 0 1 0 1
            
            total_time = count_total_time(aug_model)
            
            if regularization_coeffs:
                reg_states = get_regularization(aug_model, regularization_coeffs)
                reg_loss = sum(
                    reg_state * coeff
                    for reg_state, coeff in zip(reg_states, regularization_coeffs)
                    if coeff != 0
                )
                loss = loss_ + reg_loss
            # import pdb ;pdb.set_trace()
            pos_loss = (pos_label.to(loss_.device) *loss_).sum()
            neg_loss = (neg_label.to(loss_.device) *loss_).sum()
            loss = pos_loss-(neg_loss*1.5) 
            # loss = args.alpha *per_loss -( args.beta * neg_loss)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                aug_model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            nll_.append(loss.sum().data.cpu().numpy())
            pos_loss_.append(pos_loss.data.cpu().numpy())
            neg_loss_.append(neg_loss.data.cpu().numpy())
            
            time_meter.update(time.time() - start)
            loss_meter.update(loss.item())
            steps_meter.update(count_nfe(aug_model))
            grad_meter.update(grad_norm)
            tt_meter.update(total_time)
            num_observes.append(torch.sum(masks).data.cpu().numpy())

            if not args.no_tb_log:
                writer.add_scalar("train/NLL", loss.cpu().data.item(), itr)
            itr += 1
        
        negloss =np.sum(np.array(neg_loss_))/np.sum(num_observes)
        posloss = np.sum(np.array(pos_loss_)) /np.sum(num_observes)
        
        nll = np.sum(np.array(nll_))/np.sum(num_observes)

        
        
        
        print("Train Epoch {:04d} | NLL {:.4f} | Neg_Loss : {:.4f} | Pos_Loss :{:.4f} ".format(epoch,nll,negloss,posloss))
        
        
        evaluate(epoch,aug_model,atten_model,test_loader,loss_fn,'test',args)