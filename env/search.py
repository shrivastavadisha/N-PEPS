import time
import numpy as np
import params
from env.statement import index_to_statement
from utils import weight_attributes, convert_to_probs
from data_utils import ints_to_tensor

import torch

FAILED_SOLUTION = [None]

def dfs(env, model, timeout, max_program_len):
    """
    Perform a DFS tree-search where the nodes are program environments and the edges are statements.
    A limited width is explored (number of statements) to aid with larger lengths.
    """
    max_depth = max_program_len
    width = params.dfs_max_width
    timeout = params.timeout
    start_time = time.time()
    state = {'num_steps': 0, 'num_invalid': 0}
    state['end_time'] = start_time + timeout

    def helper(env, statements, state):
        state['num_steps'] += 1
        if 'end_time' in state and time.time() >= state['end_time']:
            return FAILED_SOLUTION

        if env.is_solution():
            return statements

        depth = len(statements)
        if depth >= max_depth:
            return False

        statement_pred, statement_probs, drop_indx = model.predict(torch.LongTensor(env.get_encoding()).unsqueeze(0))

        statement_pred = statement_pred[0]
        drop_indx = drop_indx[0]

        if env.num_vars == params.max_program_vars:
            to_drop = drop_indx
        else:
            to_drop = None

        num_tries = 0
        for statement_index in reversed(statement_pred[-width:]):
            statement = index_to_statement[statement_index]

            new_env = env.step_safe(statement, to_drop)
            if new_env is None:
                state['num_invalid'] += 1
                continue

            res = helper(new_env, statements + [statement], state)
            if res:
                if res != FAILED_SOLUTION:
                    res[depth] = env.statement_to_real_idxs(res[depth])
                return res
            num_tries += 1

        return False

    res = helper(env, [], state)
    if res == FAILED_SOLUTION:
        res = False
    return {'result': res, 'num_steps': state['num_steps'], 'time': time.time() - start_time,
            'num_invalid': state['num_invalid']}


def cab(env, model, timeout, max_program_len, max_beam_size=6553600):
    """
    Performs a CAB search. Each iteration, beam_search is called with an increased beam size and
    width. We increase the beam size exponentially each iteration, to ensure that the majority
    of paths explored are new. This prevents the need of caching which slows things down.

    max_beam_size is provided as a safety precaution
    """
    start_time = time.time()
    state = {'num_invalid': 0, 'num_steps': 0, 'end_time': start_time + timeout}

    res = False
    beam_size = params.cab_beam_size
    width = params.cab_width
    width_growth = params.cab_width_growth
    max_depth = max_program_len

    while time.time() < state['end_time']:
        res = beam_search(env, max_depth, model, beam_size, width, state)
        if res is not False or beam_size >= max_beam_size:
            break
        beam_size *= 2
        width += width_growth
    ret = {'result': res, 'num_steps': state['num_steps'], 'time': time.time() - start_time,
           'beam_size': beam_size, 'num_invalid': state['num_invalid'], 'width': width}
    return ret

def agg_and_cab(global_env, global_model, att_model, PE_preds, PE_solution_scores, agg_type,
                agg_mode, alpha, beta, timeout, max_program_len, max_beam_size=6553600):

    start_time = time.time()
    state = {'num_invalid': 0, 'num_steps': 0, 'end_time': start_time + timeout}
    res = False
    beam_size = params.cab_beam_size
    width = params.cab_width
    width_growth = params.cab_width_growth
    max_depth = max_program_len

    while time.time() < state['end_time']:
        res = agg_and_beam_search(global_env, global_model, att_model, state, PE_preds,
                                        PE_solution_scores, beam_size, width, max_depth,
                                        alpha, beta, agg_mode, agg_type)
        if res is not False or beam_size >= max_beam_size:
            break
        beam_size *= 2
        width += width_growth
    ret = {'result': res, 'num_steps': state['num_steps'], 'time': time.time() - start_time,
           'beam_size': beam_size, 'num_invalid': state['num_invalid'], 'width': width}
    return ret

def agg_and_beam_search(global_env, global_model, att_model, state, PE_preds,
                        PE_solution_scores, beam_size, expansion_size, max_depth,
                        alpha, beta, agg_mode, agg_type):
    """
    Performs a beam search where the nodes are program environments and the edges are possible statements.
    """

    def helper(beams, state):

        if time.time() >= state['end_time']:
            return FAILED_SOLUTION

        for i, (env, statements, prob) in enumerate(beams):
            if env.is_solution():
                return statements

        assert len(beams) > 0, "Empty beam list received!"
        depth = len(beams[0][1])
        if depth >= max_depth:
            return FAILED_SOLUTION

        new_beams = []
        global_env_encodings = [beam[0].get_encoding() for beam in beams]
        global_env_encodings = torch.LongTensor(global_env_encodings)
        global_statement_probs, global_drop_indx = global_model.predict(global_env_encodings)


        if (agg_mode =='program' or agg_mode == 'all') and agg_mode!='none':
            statement_vector_size = global_statement_probs.shape[1]
            num_beams = global_statement_probs.shape[0]

            if agg_type=='ca' or agg_type=='ca_sc':
                PE_statement_preds = PE_preds[0]
                PE_states = PE_preds[1]
                PE_sc = PE_preds[2]
                global_state = global_model.encoder(global_env_encodings)
                global_state = global_state.unsqueeze(1)
                PE_states = ints_to_tensor(PE_states).repeat(num_beams, 1, 1, 1)
                PE_statement_preds = ints_to_tensor(PE_statement_preds).repeat(num_beams, 1, 1)
                PE_sc = ints_to_tensor(PE_sc).repeat(num_beams, 1, 1)
                PE_statement_probs, att_weights = att_model.predict(PE_states, global_state, PE_statement_preds,
                                        PE_sc, agg_type)
                #Uncomment this line, we you need attention weights to be stored
                #np.save("att_weights_" + str(state['num_steps']), att_weights.detach().numpy())

            else:
                PE_statement_preds = PE_preds
                PE_statement_probs, step_PE_indices = convert_to_probs(PE_statement_preds,
                                                        state['num_steps'], statement_vector_size)
                if step_PE_indices:
                    PE_statement_probs = weight_attributes((PE_statement_probs, PE_solution_scores), step_PE_indices,
                                                    agg_type, num_beams)
                else:
                    PE_statement_probs = torch.zeros_like(global_statement_probs)


            statement_probs = alpha * PE_statement_probs + beta * global_statement_probs

        else:
            statement_probs = global_statement_probs


        statement_pred = np.argsort(statement_probs.cpu().numpy())
        for beam_num, (env, statements, prob) in enumerate(beams):
            if time.time() >= state['end_time']:
                return FAILED_SOLUTION

            if env.num_vars == params.max_program_vars:
                to_drop = global_drop_indx[beam_num]
            else:
                to_drop = None

            for statement_index in reversed(statement_pred[beam_num, -expansion_size:]):
                statement = index_to_statement[statement_index]
                new_env = env.step_safe(statement, to_drop)
                if new_env is None:
                    state['num_invalid'] += 1
                    continue
                new_beams.append((new_env, statements + [env.statement_to_real_idxs(statement)],
                                  prob * statement_probs[beam_num, statement_index]))

        state['num_steps'] += 1
        if len(new_beams) == 0:
            return FAILED_SOLUTION
        new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]
        return helper(new_beams, state)


    res = helper([(global_env, [], 1)], state)
    if res == FAILED_SOLUTION:
        res = False

    return res

def beam_search(env, max_depth, model, beam_size, expansion_size, state):

    def helper(beams, state):
        if time.time() >= state['end_time']:
            return FAILED_SOLUTION

        for env, statements, prob in beams:
            if env.is_solution():
                return statements

        assert len(beams) > 0, "Empty beam list received!"
        depth = len(beams[0][1])
        if depth >= max_depth:
            return FAILED_SOLUTION

        new_beams = []

        env_encodings = [beam[0].get_encoding() for beam in beams]
        env_encodings = torch.LongTensor(env_encodings)
        statement_probs, drop_indx = model.predict(env_encodings)
        statement_pred = np.argsort(statement_probs.cpu().numpy())

        for beam_num, (env, statements, prob) in enumerate(beams):
            if time.time() >= state['end_time']:
                return FAILED_SOLUTION

            if env.num_vars == params.max_program_vars:
                to_drop = drop_indx[beam_num]
            else:
                to_drop = None

            for statement_index in reversed(statement_pred[beam_num, -expansion_size:]):
                statement = index_to_statement[statement_index]
                new_env = env.step_safe(statement, to_drop)
                if new_env is None:
                    state['num_invalid'] += 1
                    continue


                new_beams.append((new_env, statements + [env.statement_to_real_idxs(statement)],
                                  prob * statement_probs[beam_num, statement_index]))

        state['num_steps'] += 1
        if len(new_beams) == 0:
            return FAILED_SOLUTION

        new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]
        return helper(new_beams, state)

    res = helper([(env, [], 1)], state)
    if res == FAILED_SOLUTION:
        res = False

    return res