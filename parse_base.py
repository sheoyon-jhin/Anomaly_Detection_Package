
import argparse
import os.path as osp

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams"]
def parse_arguments():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser("Continuous Time Flow Process")
    parser.add_argument("--data_path", type=str, default="dataset/PSM")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dataset",type=str,default='PSM')
   
    parser.add_argument("--detection_window",type=int,default = 10)
    parser.add_argument("--win_size", type=int, default=50)
   
    
    parser.add_argument("--input_size", type=int, default=25)
    
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    
    parser.add_argument("--feature_dim", type=int, default=55)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam")
    
    parser.add_argument("--weight_decay", type=float, default=0.0)
    
    # Regularizations
    parser.add_argument("--seed",type=int,default=2022,help="SEED")
    
    parser.add_argument("--save", type=str, default="ctfp")
    parser.add_argument("--effective_shape",type=int)
    
    args = parser.parse_args()
    
    args.save = osp.join("experiments", args.save)

    # args.effective_shape = args.input_size
    return args

