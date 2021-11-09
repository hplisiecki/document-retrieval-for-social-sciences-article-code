import pickle
import scipy.io
import os
from main_etm import etm_training
import argparse

# computes the results for the given model

save = 'core/'

# Maximum / minimum document frequency
max_df = 0.7
min_df = 10 # choose desired value for min_df

def parser(path_load, saturation, save):
    parser = argparse.ArgumentParser(description='The Embedded Topic Model')

    ### data and file related arguments
    parser.add_argument('--dataset', type=str, default=saturation, help='name of corpus')
    parser.add_argument('--data_path', type=str, default= path_load, help='directory containing data')
    parser.add_argument('--emb_path', type=str, default= '', help='directory containing word embeddings')
    parser.add_argument('--save_path', type=str, default= save, help='path to save results')
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training')

    ### model-related arguments
    parser.add_argument('--num_topics', type=int, default=20, help='number of topics')
    parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
    parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
    parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
    parser.add_argument('--theta_act', type=str, default='relu',
                        help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
    parser.add_argument('--train_embeddings', type=int, default=0, help='whether to fix rho or train it')

    ### optimization-related arguments
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train...150 for 20ng 100 for others')
    parser.add_argument('--mode', type=str, default='eval', help='train or eval model')
    parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
    parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
    parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
    parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
    parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
    parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

    ### evaluation, visualization, and logging-related arguments
    parser.add_argument('--num_words', type=int, default=10, help='number of words for topic viz')
    parser.add_argument('--log_interval', type=int, default=2, help='when to log training')
    parser.add_argument('--visualize_every', type=int, default=10, help='when to visualize results')
    parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
    parser.add_argument('--load_from', type=str,default=f'{save}//etm_{saturation}_K_20_Htheta_800_Optim_adam_Clip_1.0_ThetaAct_relu_Lr_0.0005_Bsz_1000_RhoSize_300_trainEmbeddings_0', help='the name of the ckpt to eval from')
    parser.add_argument('--tc', type=int, default=1, help='whether to compute topic coherence or not')
    parser.add_argument('--td', type=int, default=1, help='whether to compute topic diversity or not')

    args = parser.parse_known_args()
    return args

# saturation = 'demok', 'norm_mean', 'tfidf_mean', 'all', 'ten'

saturation = 'norm_mean'
path_load = save + './min_df_' + str(min_df) + saturation

args = parser(path_load, saturation ,save)
args[0].mode = 'eval'
etm_training(args[0], f'{saturation}.kv')
