import torch
import argparse
import json
import random
import params
from env.env import ProgramEnv
from dsl.program import Program
from dsl.example import Example


def test_with_other_PE(examples, program, index, solution_index_check):
    solution = Program.parse(program['result'])
    test_examples = examples.copy()
    true_indices = []

    if solution:
        for i in range(len(test_examples)):
            if i == index:
                solution_index_check[i+1] = True
                true_indices.append(i+1)
                continue
            test_example = [Example.from_dict(test_examples[i])]
            test_env = ProgramEnv(test_example)

            for s, statement in enumerate(solution.statements):
                used_args = []
                for next_statement in solution.statements[s:]:
                    used_args += [x for x in next_statement.args if isinstance(x, int)]
                #print("Used Args:", used_args)

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
                solution_index_check[i+1] = True
                true_indices.append(i+1)

    return solution_index_check, true_indices

def check_criteria(include_perfect_PE_programs, PE_solution_scores):
    if include_perfect_PE_programs:
        if sum(PE_solution_scores) != 0.0:
            return True
    else:
        if (sum(PE_solution_scores) != 0.0) and (1.0 not in PE_solution_scores):
            return True

def generate_preprocessed_data(input_path, include_perfect_PE_programs, agg_inp='tot', max_len=None):

    def check_prog_data(line):
        data = json.loads(line.rstrip())
        PE_solution_scores = data['PE_solution_scores']

        if check_criteria(include_perfect_PE_programs, PE_solution_scores):

            PE_programs = data['PE_solutions']
            test_examples = data['examples']

            solution_index_check = {1: False, 2:False, 3:False, 4: False, 5:False}
            mod_PE_solutions = []
            mod_PE_solution_scores = []
            PE_true_indices = []

            for j in range(len(PE_programs)):
                PE_program = PE_programs[j]
                #print(PE_program)
                if PE_program['result']!='Failed' and len(PE_program['result'])>=1:
                    solution_index_check, true_indices = test_with_other_PE(test_examples, PE_program, j, solution_index_check)
                    #print(solution_index_check)
                    mod_PE_solutions.append(PE_program)
                    mod_PE_solution_scores.append(PE_solution_scores[j])
                    PE_true_indices.append(true_indices)

                if agg_inp == 'tot':
                    num_solved = sum(value == True for value in solution_index_check.values())
                    if num_solved == 5:
                        break

            mod_data = dict(global_program=data['global_program'], examples=test_examples, \
                        peps_timeout=data['peps_timeout'], PE_solutions=mod_PE_solutions, \
                        PE_solution_scores=mod_PE_solution_scores, PE_true_indices=PE_true_indices)
            line=json.dumps(mod_data)
            return line


    lines = open(input_path, 'r').read().splitlines()
    if max_len is not None:
        selected_lines = random.sample(lines, max_len)
        lines = selected_lines

    filtered_lines = [x for x in map(check_prog_data, lines) if x is not None]

    if include_perfect_PE_programs:
        output_path = input_path + '_exclude_zero'
    else:
        output_path = input_path + '_exclude_one_zero'

    if agg_inp=='tot':
        output_path+='_' + agg_inp

    with open(output_path, 'w') as f:
        f.write('\n'.join(filtered_lines))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='../data/E2/agg_val_dataset_0.5')
    parser.add_argument('--agg_inp', type=str, default='tot')
    parser.add_argument('--include_perfect_PE_programs', default=True, action='store_false')
    args = parser.parse_args()

    generate_preprocessed_data(args.input_path, args.include_perfect_PE_programs, agg_inp=args.agg_inp)


