import torch
import params
import random
from env.env import ProgramEnv
from env.statement import Statement, statement_to_index
from dsl.example import Example
from dsl.function import Function, OutputOutOfRangeError, NullInputError
from dsl.program import Program
import numpy as np


def generate_preds(program, env, model, PE_solution_scores, i):
    statements = []
    drop = []
    states = []
    PE_sc = []
    failed_env = False
    #Obtain statement and drop vectors for all steps in the PE Program
    for step, statement in enumerate(program.statements):

        if model is not None:
            states.append(model.encoder(torch.LongTensor([env.get_encoding()])).view(-1))
        # Translate absolute indices to post-drop indices
        f, args = statement.function, list(statement.args)

        for j, arg in enumerate(args):
            if isinstance(arg, int):
                args[j] = env.real_var_idxs.index(arg)

        statement = Statement(f, args)
        statement_index = statement_to_index[statement]
        statements.append(statement_index)

        used_args = []
        for next_statement in program.statements[step:]:
            used_args += [x for x in next_statement.args if isinstance(x, int)]

        to_drop = []
        for j in range(params.max_program_vars):
            if j >= env.num_vars or env.real_var_idxs[j] not in used_args:
                to_drop.append(1)
            else:
                to_drop.append(0)

        to_drop = random.choice([j for j in range(len(to_drop)) if to_drop[j] > 0])
        if PE_solution_scores:
            PE_sc.append(PE_solution_scores[i])

        if env.num_vars < params.max_program_vars:
            try:
                env.step(statement)
            except(NullInputError, OutputOutOfRangeError):
                failed_env = True
                break
        else:
            try:
                # Choose a random var (that is not used anymore) to drop.
                env.step(statement, to_drop)
            except(NullInputError, OutputOutOfRangeError):
                failed_env = True
                break

    return statements, states, PE_sc, failed_env


def generate_attributes_from_programs(programs, problem, model=None, PE_solution_scores=None, agg_type='ca',
                                    key_type='sig', PE_true_indices=None, example_type='all'):
    '''
        Generate step-wise one-hot vectors for statements
    '''

    PE_data = problem['examples']

    if key_type =='sig':
        examples = Example.from_line(problem)
        global_env = ProgramEnv(examples)

    statement_preds = []
    drop_indices = []
    PE_states = []
    PE_soln_scores = []

    for i in range(len(programs)): #num of solutions discovered (includes Failed solutions)
        #Get the PE Program and corresponding environment
        if programs[i]['result']!= 'Failed':
            program = Program.parse(programs[i]['result'])

            if program:
                if key_type =='sig': #all examples and global model
                    reqd_envs = [global_env.copy()]

                if key_type =='sii':

                    if example_type == 'set': #take examples in a set
                        PE = [Example.from_dict(data['examples'][k-1]) for k in PE_true_indices[i]]
                        PE_env = ProgramEnv(PE)
                        reqd_envs = [PE_env]

                    if example_type == 'all': #take examples individually
                        reqd_envs = []
                        for k in PE_true_indices[i]:
                            PE = [Example.from_dict(data['examples'][k-1])]
                            PE_env = ProgramEnv(PE)
                            reqd_envs.append(PE_env)

                if key_type =='sij': #the set version is equal to sig. So, there is only the all version here
                    reqd_envs = []
                    for k in [1, 2, 3, 4, 5]:
                        PE = [Example.from_dict(data['examples'][k-1])]
                        PE_env = ProgramEnv(PE)
                        reqd_envs.append(PE_env)

                for reqd_env in reqd_envs:

                    statements, states, PE_sc, failed_env = generate_preds(program, reqd_env, model, PE_solution_scores, i)
                    if not failed_env:
                        statement_preds.append(statements)
                        PE_states.append(states)
                        PE_soln_scores.append(PE_sc)
                    else:
                        statement_preds.append([params.num_statements])
                        PE_states.append([torch.zeros(params.state_dim)])
                        PE_soln_scores.append([0.0])

            elif agg_type=='ca' or agg_type=='ca_sc':
                statement_preds.append([params.num_statements])
                PE_states.append([torch.zeros(params.state_dim)])
                PE_soln_scores.append([0.0])

        elif agg_type=='ca' or agg_type=='ca_sc':
            statement_preds.append([params.num_statements])
            PE_states.append([torch.zeros(params.state_dim)])
            PE_soln_scores.append([0.0])


    return statement_preds, PE_states, PE_soln_scores

def convert_to_probs(list_preds, step_index, size):
    step_entry = []
    step_indices = []
    for i in range(len(list_preds)):
        entry = list_preds[i]
        if len(entry)> step_index:
            if entry[step_index] != params.num_statements:
                step_entry.append(entry[step_index])
                step_indices.append(i)
    if step_indices:
        step_entry = torch.LongTensor(step_entry)
        mod_entry = torch.nn.functional.one_hot(step_entry, size)
        mod_entry = mod_entry.type(torch.FloatTensor)
        return mod_entry, step_indices
    else:
        return [], []


def weight_attributes(weights, PE_step_indices, agg_type, num_beams):

    if agg_type =='sum':
        agg_weights = torch.sum(weights[0], dim=0)
        agg_weights = agg_weights.unsqueeze(0).repeat(num_beams, 1)

    if agg_type =='mean':
        agg_weights = torch.mean(weights[0], dim=0)
        agg_weights = agg_weights.unsqueeze(0).repeat(num_beams, 1)

    if agg_type =='mean_sc':
        sc = weights[1]
        sc = [x for x in sc if x != 0.0]
        rel_sc = [sc[i] for i in PE_step_indices]
        rel_sc = torch.FloatTensor(rel_sc).view(-1, 1)
        new_weights = torch.mul(rel_sc, weights[0])
        agg_weights = torch.mean(new_weights, dim=0)
        agg_weights = agg_weights.unsqueeze(0).repeat(num_beams, 1)

    return agg_weights