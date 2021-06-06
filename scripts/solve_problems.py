import argparse
import os
import json
import multiprocessing
import torch
import numpy as np
import time
import params
import random
import itertools
from model.model import PCCoder
from env.env import ProgramEnv
from env.statement import Statement, statement_to_index
from env.search import cab, dfs, agg_and_cab
from dsl.example import Example
from dsl.program import Program
from dsl.value import Value
from model.att_model import AttModel
from utils import generate_attributes_from_programs

# for reproducibility
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

device = torch.device('cpu')

def load_model(model_load_dir, model, include_self_attention, key_type):
  model_path = os.path.join(model_load_dir, 'best_model.th')
  loaded_model_dict = torch.load(model_path, map_location=torch.device('cpu'))
  model.load_state_dict(loaded_model_dict)
  return model

def load_problems(path):
    problems = []
    with open(path) as fh:
        for line in fh:
            problem = json.loads(line.rstrip())
            problems.append(problem)
    return problems

def aggregate_and_search(problem, global_model, PE_model=None, att_model=None, PE_solutions=None, PE_solution_scores=None,
                        method='peps', agg_type='mean_sc', agg_mode=None, alpha=0.0, beta=1.0,
                        gps_timeout=5, max_program_len=4, key_type='sig', PE_true_indices=None, example_type='all'):

  examples = Example.from_line(problem)
  global_env = ProgramEnv(examples)

  if params.search_method == 'beam':
    if agg_mode == 'program' or agg_mode == 'both':
      if agg_type =='ca' or agg_type=='ca_sc':

        if key_type =='sig':
          model = global_model
        if key_type =='sii':
          if example_type == 'set':
            model = global_model
          if example_type == 'all':
            model = PE_model
        if key_type =='sij':
          model = PE_model

        PE_preds = generate_attributes_from_programs(PE_solutions, problem, model, PE_solution_scores, agg_type)

      else:
        PE_preds, _, _ = generate_attributes_from_programs(PE_solutions, problem, agg_type=agg_type)

    if agg_mode=='none':
      PE_preds = None

    global_solution = agg_and_cab(global_env, global_model, att_model, PE_preds, PE_solution_scores, agg_type, \
                    agg_mode, alpha, beta, gps_timeout, max_program_len)

  return global_solution


def solve_problem(PE_data, PE_model, global_data, index, peps_timeout, max_program_len, solution_index_check):
  #PE search env and example
  PE = [Example.from_dict(PE_data)]
  PE_env = ProgramEnv(PE)

  test_examples = global_data['examples'].copy()

  #PE search
  if params.search_method == 'beam':
    solution = cab(PE_env, PE_model, peps_timeout, max_program_len)
  elif params.search_method == 'dfs':
    solution = dfs(PE_env, PE_model, peps_timeout, max_program_len)

  solution_score = 0.0
  mod_solution = solution.copy()
  mod_solution, fail_counter = store_stats(mod_solution, [PE_data])
  if fail_counter == 1:
    return mod_solution, solution_score, solution_index_check, []

  #Testing the solution of PE search with the set of 5 examples.
  true_indices = []
  if len(solution['result'])>=1:
    # solution_score+=1.0 # it already satisfies one example
    for i in range(len(test_examples)):
      if i == index:
        solution_index_check[i+1] = True
        true_indices.append(i+1)
        solution_score+=1.0
        continue
      test_example = [Example.from_dict(test_examples[i])]
      test_env = ProgramEnv(test_example)

      for s, statement in enumerate(solution['result']):
        used_args = []
        for next_statement in solution['result'][s:]:
            used_args += [x for x in next_statement.args if isinstance(x, int)]

        to_drop = []
        for j in range(params.max_program_vars):
            if j >= test_env.num_vars or test_env.real_var_idxs[j] not in used_args:
                to_drop.append(1)
            else:
                to_drop.append(0)
        drop_idx = random.choice([j for j in range(len(to_drop)) if to_drop[j] > 0])

        if test_env.num_vars < params.max_program_vars:
          new_env = test_env.step_safe(statement)
        else:
          new_env = test_env.step_safe(statement, drop_idx)
        if new_env is None:
            break
        else:
            test_env = new_env

      if new_env is not None and new_env.is_solution():
        solution_score+=1.0
        solution_index_check[i+1] = True
        true_indices.append(i+1)


  return mod_solution, solution_score/len(global_data['examples']), solution_index_check, true_indices

def get_total_peps_time(PE_solutions, peps_timeout, gps_timeout):
  total_time = 0.0
  for solution in PE_solutions:
    total_time+=solution['time']
  remaining_peps_time = 5*peps_timeout-total_time
  gps_timeout+=remaining_peps_time
  return gps_timeout, total_time

def find_PE_solutions(problem, PE_model, peps_timeout, max_program_len, gps_timeout, agg_inp, add_residual_time_peps, seq):
  '''
  Find PE solutions
  '''
  # get PE data as a list
  original_timeout = gps_timeout
  PE = problem['examples']
  PE_solutions = []
  PE_solution_scores = []
  PE_true_indices = []
  done_index = -1
  #print("Doing PE Searches...")
  solution_index_check = {1: False, 2:False, 3:False, 4: False, 5:False}

  for j in range(len(PE)):
    if agg_inp =='ind':
      solution_index_check = {1: False, 2:False, 3:False, 4: False, 5:False}

    PE_solution, PE_solution_score, solution_index_check, true_indices = solve_problem(PE[j], PE_model, problem, j,
                                    peps_timeout, max_program_len, solution_index_check)
    PE_solutions.append(PE_solution)
    PE_solution_scores.append(PE_solution_score)
    PE_true_indices.append(true_indices)

    # if we find a single PE solution which satisfies all examples, return the corresponding index
    if PE_solution_score == 1.0:
      done_index = j
      break

    num_solved = sum(value == True for value in solution_index_check.values())
    if num_solved == 5:
      break

  gps_timeout, peps_time = get_total_peps_time(PE_solutions, peps_timeout, gps_timeout)
  if not add_residual_time_peps:
    gps_timeout = original_timeout

  return PE_solutions, PE_solution_scores, done_index, gps_timeout, peps_time, PE_true_indices


def store_stats(solution, problem, fail_counter=0):

  if solution['result'] is False:
    solution['result'] = "Failed"
    fail_counter+= 1
  else:
    values = [Value.construct(x) for x in problem[0]['inputs']]
    value_types = [x.type for x in values]
    solution['result'] = Program(value_types, solution['result']).encode()
  return solution, fail_counter

def solve_problems(test_problems, global_model, PE_model, method, agg_inp, agg_mode, agg_type, alpha, beta,
                  gps_timeout, peps_timeout, max_program_len, att_model, add_residual_time_peps, key_type,
                  example_type, seq):
  """
  Attempts to predict programs for the given I/O sample sets.
  """
  counter = 0
  fail_counter = 0
  global_solutions = []
  PE_solution_scores = None
  PE_solutions = None
  PE_true_indices = None
  global_timeouts = 0.0
  agg_count=0
  peps_time = 0.0
  for i in range(len(test_problems)): #iterate over the test data
      global_timeout = gps_timeout
      gt = []
      problem = test_problems[i]
      examples = Example.from_line(problem)
      env = ProgramEnv(examples)

      counter += 1
      print("\rSolving problems... %d (failed: %d)" % (counter, fail_counter), end="")
      if method == 'peps':
        # find PE solutions
        PE_solutions, PE_solution_scores, done_index, global_timeout, peps_time, PE_true_indices = find_PE_solutions(problem,
                                  PE_model, peps_timeout, max_program_len, global_timeout, agg_inp, add_residual_time_peps, seq)

        if done_index >= 0:
          global_solution = PE_solutions[done_index]
          global_solution['PE_solution_scores'] = PE_solution_scores
          global_solution['time'] = peps_time
          global_solutions.append(global_solution)
          continue

      # aggregate PE solutions to find a global solution
      global_timeouts+=global_timeout
      agg_count+=1


      global_solution = aggregate_and_search(problem, global_model, PE_model, att_model, PE_solutions, PE_solution_scores,
                        method, agg_type, agg_mode, alpha, beta, global_timeout, max_program_len, key_type,
                        PE_true_indices, example_type)

      global_solution, fail_counter = store_stats(global_solution, problem['examples'], fail_counter)
      global_solution['PE_results'] = PE_solutions
      global_solution['PE_solution_scores'] = PE_solution_scores
      global_solution['time']+= peps_time
      global_solutions.append(global_solution)
  return global_solutions

def main():
  #Get command-line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--machine_name', type=str, default='ins-7')
  #inference-related
  parser.add_argument('--method', type=str, help='gps, peps', default='peps')
  parser.add_argument('--agg_inp', type=str, help='ind, tot', default='tot')
  parser.add_argument('--agg_type', type=str, default='ca',
    help='mean=Mean-N-PEPS, sum=Sum-N-PEPS, mean_sc=Mean-N-PEPS+U, ca=N-PEPS, ca_sc=N-PEPS+U')
  parser.add_argument('--key_type', type=str, default='sij', help='sig=N-PEPS-PG, sii=N-PEPS-PP, sij=N-PEPS')
  parser.add_argument('--example_type', type=str, default='all', help='set, all')
  parser.add_argument('--agg_mode', type=str, help='program, state, all, none', default='program')
  parser.add_argument('--alpha', type=float, default=0.8)
  parser.add_argument('--gps_timeout', type=float, default=1.0)
  parser.add_argument('--peps_timeout', type=float, default=0.8)
  parser.add_argument('--add_residual_time_peps', default=True, action='store_false')
  #att model params
  parser.add_argument('--dropout', type=float, default=0.0)
  parser.add_argument('--include_pos_emb', default=True, action='store_false')
  parser.add_argument('--include_self_attention', default=False, action='store_true')
  parser.add_argument('--self_attention_type', type=str, default='key', help='key, val, both')
  parser.add_argument('--include_ff', default=True, action='store_false')
  parser.add_argument('--include_res_ln', default=True, action='store_false')
  parser.add_argument('--return_att_weights', default=False, action='store_true')
  parser.add_argument('--seq', default=True, action='store_false')
  #dataset-related
  parser.add_argument('--test_path', type=str, default='data/E1/test_splits/len_4/split_5')
  parser.add_argument('--result_path', type=str, default='results/E1/test/')
  parser.add_argument('--att_model_path', type=str, default='trained_models/E1/N-PEPS')
  parser.add_argument('--num_of_problems', type=int, default=-1)
  parser.add_argument('--max_program_len', type=int, default=4)
  args = parser.parse_args()


  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)

  beta = 1.0 - args.alpha
  # Load test data
  test_problems = load_problems(args.test_path)
  if args.num_of_problems != -1:
    test_problems = test_problems[:args.num_of_problems]

  # Load models
  global_model = PCCoder()
  global_model.load(params.global_model_path)
  global_model.eval()

  PE_model = PCCoder()
  PE_model.load(params.PE_model_path)
  PE_model.eval()

  if args.agg_type == 'ca' or args.agg_type =='ca_sc':
    att_model = AttModel(include_self_attention=args.include_self_attention, include_pos_emb=args.include_pos_emb,
                include_ff=args.include_ff, include_res_ln=args.include_res_ln, dropout=args.dropout,
                return_att_weights=args.return_att_weights, self_attention_type=args.self_attention_type)
    att_model= load_model(args.att_model_path, att_model, args.include_self_attention, args.key_type)
    att_model.eval()
  else:
    att_model = None

  # Carry out inference
  results = solve_problems(test_problems, global_model, PE_model, args.method,
                           args.agg_inp, args.agg_mode, args.agg_type, args.alpha, beta, args.gps_timeout,
                           args.peps_timeout, args.max_program_len, att_model, args.add_residual_time_peps,
                           args.key_type, args.example_type, args.seq)

  # Calculate percentage of problems solved
  solved = len([x for x in results if x['result'] != 'Failed'])
  print("Solved: %d\\%d:" % (solved, len(results)), str(100.0 * solved / len(results)) + '%')

  # Store the results
  out_file_name = args.machine_name + '#' +  args.test_path.split("/")[3].split("_")[0]+'#' + args.test_path.split("/")[-1].split("_")[-1]\
                  + '#' + args.method+ '#' + str(args.gps_timeout) + '#' + str(args.seed)

  if args.method == 'peps':
    out_file_name += '#' + str(args.peps_timeout) + '#' + str(args.agg_inp)\
              + '#'+ args.agg_mode + '#' + args.agg_type + '#' + str(args.alpha)+ '#' + args.att_model_path.split("/")[-1]\
              + '#' + str(args.seed)


  out_file = os.path.join(args.result_path, out_file_name)
  os.makedirs(args.result_path, exist_ok=True)
  open(out_file, 'w').write('\n'.join([json.dumps(x) for x in results]))


if __name__ == '__main__':
  main()
