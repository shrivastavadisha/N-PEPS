from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import json
import os
import random
import torch
import time
import multiprocessing

import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch import nn
from tqdm import tqdm
from shutil import copyfile
from data_utils import AggDataset, collate_fn

import params
from model.model import PCCoder
from model.att_model import AttModel
from cuda import use_cuda, LongTensor, FloatTensor
from env.env import ProgramEnv
from env.operator import Operator, operator_to_index
from env.statement import Statement, statement_to_index
from dsl.program import Program
from dsl.example import Example

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save(model, optimizer, epoch, save_dir):
    to_save = model.module if hasattr(model, "module") else model
    torch.save(to_save.state_dict(), os.path.join(save_dir, "best_model.th"))
    torch.save({"optimizer": optimizer.state_dict(), "last_epoch": epoch}, os.path.join(save_dir, "optim.th"))


def get_accuracy(pred, gold, target_pad_idx):
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(target_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


def get_predictions(att_model, global_state, PE_states, PE_statements, PE_operators, PE_solution_scores, att_type,
                    return_att_weights=False):
    statement_pred, operator_pred, att_weights = att_model(PE_states, global_state, PE_statements,
                                    PE_operators, PE_solution_scores, att_type)

    statement_pred = statement_pred.view(-1, params.num_statements)
    operator_pred = operator_pred.view(-1, params.num_operators)

    if return_att_weights:
        return statement_pred, operator_pred, att_weights
    else:
        return statement_pred, operator_pred, None

def calculate_loss(att_model, global_state, PE_states, PE_statements, PE_operators, PE_solution_scores,
                    att_type, global_statement, global_operator, statement_criterion, operator_criterion,
                    epoch, return_att_weights=False, tb_writer=None, mode='train'):

    statement_pred, operator_pred, att_weights = get_predictions(att_model, global_state, PE_states, PE_statements,
                                PE_operators, PE_solution_scores, att_type, return_att_weights)

    statement_loss = statement_criterion(statement_pred, global_statement)
    operator_loss = operator_criterion(operator_pred, global_operator)

    num_correct_statements, num_statements = get_accuracy(statement_pred, global_statement, params.num_statements)
    num_correct_operators, num_operators = get_accuracy(operator_pred, global_operator, params.num_operators)#num_operators will be same as num_statements
    assert num_statements == num_operators

    if return_att_weights:
        statement_pred_tag = mode+'_statement_pred'
        operator_pred_tag = mode+'_operator_pred'
        statement_gt_tag = mode+'_statement_gt'
        operator_gt_tag = mode+'_operator_gt'
        PE_statements_tag = mode+'_PE_statements'
        PE_solution_scores_tag = mode+'_PE_solution_scores'
        att_weights_tag = mode+'_att_weights'

        bs, max_len_g, num_examples = att_weights.size(0), att_weights.size(2), PE_statements.size(1)
        att_weights = torch.mean(att_weights, dim=1).view(bs, max_len_g, num_examples, -1)

        tb_writer.add_text(statement_pred_tag, str(statement_pred.max(1)[1]), epoch)
        tb_writer.add_text(operator_pred_tag, str(operator_pred.max(1)[1]), epoch)
        tb_writer.add_text(statement_gt_tag, str(global_statement), global_step=None)
        tb_writer.add_text(operator_gt_tag, str(global_operator), global_step=None)
        tb_writer.add_text(PE_statements_tag, str(PE_statements), global_step=None)
        tb_writer.add_text(PE_solution_scores_tag, str(PE_solution_scores), global_step=None)
        tb_writer.add_text(att_weights_tag, str(att_weights.detach()), epoch)


    return statement_loss, operator_loss, num_correct_statements, num_statements, num_correct_operators


def main():
    parser = argparse.ArgumentParser()
    #ind or tot only affects the number of PE solutions discovered
    parser.add_argument('--agg_train_path', type=str, default='data/E1/agg_train_dataset_rand_exclude_one_zero_tot')
    parser.add_argument('--agg_val_path', type=str, default='data/E1/agg_val_dataset_rand_exclude_one_zero_tot')
    parser.add_argument('--att_type', type=str, default='ca_sc', help='ca=N-PEPS, ca_sc=N-PEPS+U')
    parser.add_argument('--att_lr_optimizer', type=str, default='adam')
    parser.add_argument('--att_lr_scheduler', type=str, default='cosinewarm')
    parser.add_argument('--att_learn_rate', type=float, default=3e-4)
    parser.add_argument('--key_type', type=str, default='sii', help='sig=N-PEPS-PG, sii=N-PEPS-PP, sij=N-PEPS')
    parser.add_argument('--example_type', type=str, default='all', help='set, all')
    parser.add_argument('--return_att_weights', default=False, action='store_true')
    parser.add_argument('--num_instances', type=int, default=None)
    #model params
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--d_k', type=float, default=64)
    parser.add_argument('--initialize_head', default=True, action='store_false')
    parser.add_argument('--include_pos_emb', default=True, action='store_false')
    parser.add_argument('--include_self_attention', default=False, action='store_true')
    parser.add_argument('--self_attention_type', type=str, default='key', help='key, val, both')
    parser.add_argument('--include_ff', default=True, action='store_false')
    parser.add_argument('--include_res_ln', default=True, action='store_false')
    args = parser.parse_args()

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    # Define paths for storing tensorboard logs (was used for hyperparameter selection)
    # dir_name = 'data#' + '_'.join([str(x) for x in args.agg_train_path.split("/")[-1].split('_')[3:]]) +'#att_type#' + \
    #             args.att_type + '#att_lr_optimizer#' + args.att_lr_optimizer + '#att_lr_scheduler#' + args.att_lr_scheduler\
    #             + '#att_learn_rate#' + str(args.att_learn_rate) + '#key_type#' + args.key_type \
    #             + '#example_type#' + args.example_type + '#include_sa#' + str(args.include_self_attention) \
    #             + '#sa_type#' + args.self_attention_type + '#num_instances#' + str(args.num_instances)
    dir_name =''

    save_dir = os.path.join(params.model_output_path, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    tb_writer = SummaryWriter(os.path.join(save_dir, "logs"))

    # Load models
    global_model = PCCoder()
    global_model.load(params.global_model_path)
    global_model.eval()

    PE_model = PCCoder()
    PE_model.load(params.PE_model_path)
    PE_model.eval()

    if args.initialize_head:
        head_params = []
        for name, weights in global_model.named_parameters():
            if name == 'statement_head.weight' or name=='operator_head.weight':
                head_params.append(weights.data)
    else:
        head_params = None

    #Define Attention Model
    att_model = AttModel(include_self_attention=args.include_self_attention, include_pos_emb=args.include_pos_emb,
                include_ff=args.include_ff, include_res_ln=args.include_res_ln, dropout=args.dropout,
                return_att_weights=args.return_att_weights, n_head=args.n_head, d_k=args.d_k, head_params=head_params,
                self_attention_type=args.self_attention_type)

    att_model.to(device)

    #Define optimizer and loss
    if args.att_lr_scheduler == 'cosine':
        if args.att_lr_optimizer == 'adam':
            optimizer = torch.optim.Adam(att_model.parameters(), lr=args.att_learn_rate)
        if args.att_lr_optimizer == 'sgd':
            optimizer = torch.optim.SGD(att_model.parameters(), lr=args.att_learn_rate)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    if args.att_lr_scheduler == 'cosinewarm':
        if args.att_lr_optimizer == 'adam':
            optimizer = torch.optim.Adam(att_model.parameters(), lr=args.att_learn_rate)
        if args.att_lr_optimizer == 'sgd':
            optimizer = torch.optim.SGD(att_model.parameters(), lr=args.att_learn_rate)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    if args.att_lr_scheduler == 'tr2':
        if args.att_lr_optimizer == 'adam':
            optimizer = torch.optim.Adam(att_model.parameters(), lr=args.att_learn_rate)
        if args.att_lr_optimizer == 'sgd':
            optimizer = torch.optim.SGD(att_model.parameters(), lr=args.att_learn_rate)
        lr_sched = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.att_learn_rate/2.0, max_lr=0.1,
                    step_size_up=5, mode="triangular2", cycle_momentum=False)

    if args.att_lr_scheduler == 'reduceonplateau':
        if args.att_lr_optimizer == 'adam':
            optimizer = torch.optim.Adam(att_model.parameters(), lr=args.att_learn_rate)
        if args.att_lr_optimizer == 'sgd':
            optimizer = torch.optim.SGD(att_model.parameters(), lr=args.att_learn_rate)
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    if params.load_from_checkpoint:
        print("=> loading checkpoint '{}'".format(params.checkpoint_dir))
        model_path = os.path.join(params.checkpoint_dir, 'best_model.th')
        opt_path = os.path.join(params.checkpoint_dir, 'optim.th')
        status_dict = torch.load(opt_path, map_location=torch.device('cpu'))
        att_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        optimizer.load_state_dict(status_dict['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(params.checkpoint_dir, status_dict['last_epoch']))

    statement_criterion = nn.CrossEntropyLoss(ignore_index=params.num_statements, reduction='sum')
    operator_criterion = nn.CrossEntropyLoss(ignore_index=params.num_operators, reduction='sum')

    # Train and Val DataLoaders
    train_dataset = AggDataset(args.agg_train_path, args.num_instances, global_model, PE_model, args.key_type,
                                args.example_type) #, args.include_state_emb)
    train_data_loader = DataLoader(train_dataset, batch_size=params.att_batch_size, shuffle=True, collate_fn=collate_fn)

    if args.num_instances!=None:
        val_instances = int(args.num_instances/4.0)
    else:
        val_instances = None

    val_dataset = AggDataset(args.agg_val_path, val_instances, global_model, PE_model, args.key_type, args.example_type)
                            #args.include_state_emb)
    val_data_loader = DataLoader(val_dataset, batch_size=params.att_batch_size, shuffle=True, collate_fn=collate_fn)

    best_val_loss = np.inf

    for epoch in range(params.num_epochs):
        print("Epoch %d" % epoch)

        ########################Training Loop#############################################
        att_model.train()
        total_statements, total_correct_statements, total_correct_operators, total_statement_loss, \
                                            total_operator_loss, total_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for batch in tqdm(train_data_loader):

            global_state = Variable(batch[0]).to(device)
            global_statement = Variable(batch[1]).view(-1).to(device)
            global_operator = Variable(batch[2]).view(-1).to(device)
            PE_states = Variable(batch[3]).to(device)
            PE_statements = Variable(batch[4]).to(device)
            PE_operators = Variable(batch[5]).to(device)
            PE_solution_scores = Variable(batch[6]).to(device)


            optimizer.zero_grad()

            statement_loss, operator_loss, num_correct_statements, num_statements, num_correct_operators = \
                                                                calculate_loss(att_model, global_state, PE_states,
                                                                PE_statements, PE_operators, PE_solution_scores,
                                                                args.att_type, global_statement, global_operator,
                                                                statement_criterion, operator_criterion,
                                                                epoch, args.return_att_weights, tb_writer, mode='train')

            loss = statement_loss + operator_loss
            loss.backward()
            optimizer.step()


            total_statements += num_statements
            total_correct_statements += num_correct_statements
            total_correct_operators +=num_correct_operators
            total_statement_loss +=statement_loss.item()
            total_operator_loss +=operator_loss.item()
            total_loss += loss.item()

        avg_train_loss = total_loss/total_statements
        avg_statement_accuracy = total_correct_statements/ total_statements
        avg_operator_accuracy = total_correct_operators/ total_statements
        avg_statement_loss = total_statement_loss/ total_statements
        avg_operator_loss = total_operator_loss/ total_statements

        tb_writer.add_scalar("metrics/train_loss", avg_train_loss, epoch)
        tb_writer.add_scalar("metrics/train_statement_loss", avg_statement_loss, epoch)
        tb_writer.add_scalar("metrics/train_operator_loss", avg_operator_loss, epoch)
        tb_writer.add_scalar("metrics/train_statement_accuracy", avg_statement_accuracy, epoch)
        tb_writer.add_scalar("metrics/train_operator_accuracy", avg_operator_accuracy, epoch)

        print("Train loss: Total %f" % avg_train_loss, "Statement %f" % avg_statement_loss,
                                                             "Operator %f" % avg_operator_loss)
        print("Train accuracy: Statement %f" % avg_statement_accuracy,
                                                             "Operator %f" % avg_operator_accuracy)

        ######################################Evaluation Loop############################################
        att_model.eval()

        with torch.no_grad():

            total_statements, total_correct_statements, total_correct_operators, total_statement_loss, \
                                    total_operator_loss, total_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            for batch in tqdm(val_data_loader):
                global_state = Variable(batch[0]).to(device)
                global_statement = Variable(batch[1]).view(-1).to(device)
                global_operator = Variable(batch[2]).view(-1).to(device)
                PE_states = Variable(batch[3]).to(device)
                PE_statements = Variable(batch[4]).to(device)
                PE_operators = Variable(batch[5]).to(device)
                PE_solution_scores = Variable(batch[6]).to(device)


                statement_loss, operator_loss, num_correct_statements, num_statements, num_correct_operators = \
                                                                calculate_loss(att_model, global_state, PE_states,
                                                                PE_statements, PE_operators, PE_solution_scores,
                                                                args.att_type, global_statement, global_operator,
                                                                statement_criterion, operator_criterion,
                                                                epoch, args.return_att_weights, tb_writer, mode='val')

                loss = statement_loss + operator_loss

                total_statements += num_statements
                total_correct_statements += num_correct_statements
                total_correct_operators +=num_correct_operators
                total_statement_loss +=statement_loss.item()
                total_operator_loss +=operator_loss.item()
                total_loss += loss.item()

        avg_val_loss = total_loss/total_statements
        avg_statement_accuracy = total_correct_statements/ total_statements
        avg_operator_accuracy = total_correct_operators/ total_statements
        avg_statement_loss = total_statement_loss/ total_statements
        avg_operator_loss = total_operator_loss/ total_statements

        tb_writer.add_scalar("metrics/val_loss", avg_val_loss, epoch)
        tb_writer.add_scalar("metrics/val_statement_loss", avg_statement_loss, epoch)
        tb_writer.add_scalar("metrics/val_operator_loss", avg_operator_loss, epoch)
        tb_writer.add_scalar("metrics/val_statement_accuracy", avg_statement_accuracy, epoch)
        tb_writer.add_scalar("metrics/val_operator_accuracy", avg_operator_accuracy, epoch)

        print("Val loss: Total %f" % avg_val_loss, "Statement %f" % avg_statement_loss,
                                                             "Operator %f" % avg_operator_loss)
        print("Val accuracy: Statement %f" % avg_statement_accuracy,
                                                             "Operator %f" % avg_operator_accuracy)

        if args.att_lr_scheduler =='reduceonplateau':
            lr_sched.step(avg_val_loss)
        else:
            lr_sched.step()

        if avg_val_loss < best_val_loss:
            print("Found new best att_model")
            best_val_loss = avg_val_loss
            save(att_model, optimizer, epoch, save_dir)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr == params.patience:
                print("Ran out of patience. Stopping training early...")
                break

if __name__ == '__main__':
    main()
