import argparse
import os
import json
import torch
import numpy as np

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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def load_problems(path):
    problems = []
    with open(path) as fh:
        for line in fh:
            problem = json.loads(line.rstrip())
            problems.append(problem)
    return problems

def solve_problem(PE_data, PE_model, global_data, index, peps_timeout, max_program_len):
  #PE search env and example
  PE = [Example.from_dict(PE_data)]
  PE_env = ProgramEnv(PE)

  test_examples = global_data['examples'].copy()
  test_examples.pop(index)

  #PE search
  if params.search_method == 'beam':
    solution = cab(PE_env, PE_model, peps_timeout, max_program_len)
  elif params.search_method == 'dfs':
    solution = dfs(PE_env, PE_model, peps_timeout, max_program_len)

  solution_score = 0.0
  mod_solution = solution.copy()
  mod_solution, fail_counter = store_stats(mod_solution, [PE_data])
  if fail_counter == 1:
    return mod_solution, solution_score

  #Testing the solution of PE search with the set of 5 examples.
  if len(solution['result'])>=1:
    solution_score+=1.0 # it already satisfies one example
    for i in range(len(test_examples)):
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

  return mod_solution, solution_score/len(global_data['examples'])

def find_PE_solutions(problem, PE_model, peps_timeout, max_program_len):
  '''
  Find PE solutions
  '''
  # get PE data as a list
  PE = problem['examples']
  PE_solutions = []
  PE_solution_scores = []

  #print("Doing PE Searches...")
  for j in range(len(PE)):
    #print("PE index:", j)
    PE_solution, PE_solution_score = solve_problem(PE[j], PE_model, problem, j, peps_timeout, max_program_len)
    PE_solutions.append(PE_solution)
    PE_solution_scores.append(PE_solution_score)

  return PE_solutions, PE_solution_scores


def store_stats(solution, problem, fail_counter=0):

  if solution['result'] is False:
    solution['result'] = "Failed"
    fail_counter+= 1
  else:
    values = [Value.construct(x) for x in problem[0]['inputs']]
    value_types = [x.type for x in values]
    solution['result'] = Program(value_types, solution['result']).encode()
  return solution, fail_counter

def generate_data(problems, PE_model, max_program_len, peps_timeout_type, output_data_path):
  """
  Attempts to predict programs for the given I/O sample sets.
  """
  PE_solution_scores = None
  PE_solutions = None
  if peps_timeout_type == 'rand':
    peps_timeouts = np.linspace(0.1, 1.0, 10)
    num_of_timeouts_per_problem = 2
  if peps_timeout_type =='0.5':
    peps_timeouts = [0.5]
  if peps_timeout_type == '0.5+-0.1':
    peps_timeouts = [0.4, 0.5, 0.6]
  f = open(output_data_path, 'w')

  for i in range(len(problems)): #iterate over the data
      problem = problems[i]
      #print("Problem: ", i+1)
      global_solution = problem['program']
      examples = problem['examples']
      if peps_timeout_type == 'rand':
        chosen_timeouts = np.random.choice(peps_timeouts, num_of_timeouts_per_problem)
      else:
        chosen_timeouts = peps_timeouts
      for peps_timeout in chosen_timeouts:
        #find PE solutions
        PE_solutions, PE_solution_scores = find_PE_solutions(problem, PE_model, peps_timeout,
                                                        max_program_len)
        data = dict(global_program=global_solution, examples=examples, peps_timeout=peps_timeout, \
              PE_solutions=PE_solutions, PE_solution_scores=PE_solution_scores)
        f.write(json.dumps(data) + '\n')


def main():
  #Get command-line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=42)
  #dataset-related
  parser.add_argument('--input_data_path', type=str, default='../data/E2/train_dataset')
  parser.add_argument('--output_data_path', type=str, default='../data/E2/agg_train_dataset_rand')
  parser.add_argument('--peps_timeout_type', type=str, default='rand')
  parser.add_argument('--num_of_problems', type=int, default=-1)
  parser.add_argument('--max_program_len', type=int, default=12)
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)

  # Load data
  problems = load_problems(args.input_data_path)
  if args.num_of_problems != -1:
    problems = problems[:args.num_of_problems]

  # Load PEPS model
  PE_model = PCCoder()
  PE_model.load(params.PE_model_path)
  PE_model.eval()

  # Generate data
  generate_data(problems, PE_model, args.max_program_len, args.peps_timeout_type, args.output_data_path)


if __name__ == '__main__':
  main()
