import argparse 
import os

def getcfg()-> any:
    parser = argparse.ArgumentParser('Training of the Architectures', add_help=True)
    parser.add_argument('-c','--config_file', type=str, 
                                help='Path to the configuration file',
                                default='/home/ubuntu/myproject/wcr_neural_kan/config/config_1D.yaml')
    args=parser.parse_args()
    return args.config_file