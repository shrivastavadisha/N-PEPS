import torch
from torch.utils.data import Dataset
import tqdm
import numpy as np
import json
import params
import random

from cuda import use_cuda, LongTensor, FloatTensor
from env.env import ProgramEnv
from env.operator import Operator, operator_to_index
from env.statement import Statement, statement_to_index
from dsl.program import Program
from dsl.example import Example
from dsl.function import Function, OutputOutOfRangeError, NullInputError


def ints_to_tensor(ints, pad_index=0.0):
	"""
	Converts a nested list of integers to a padded tensor.
	"""
	if isinstance(ints, torch.Tensor):
		return ints
	if isinstance(ints, list):
		if isinstance(ints[0], int):
			return torch.LongTensor(ints)
		if isinstance(ints[0], float):
			return torch.FloatTensor(ints)
		if isinstance(ints[0], torch.Tensor):
			return pad_tensors(ints, pad_index)
		if isinstance(ints[0], list):
			return ints_to_tensor([ints_to_tensor(inti, pad_index) for inti in ints], pad_index)

def pad_tensors(tensors, pad_index):
	"""
	Takes a list of `N` M-dimensional tensors (M<4) and returns a padded tensor.

	The padded tensor is `M+1` dimensional with size `N, S1, S2, ..., SM`
	where `Si` is the maximum value of dimension `i` amongst all tensors.
	"""
	rep = tensors[0]
	padded_dim = []
	for dim in range(rep.dim()):
		max_dim = max([tensor.size(dim) for tensor in tensors])
		padded_dim.append(max_dim)
	padded_dim = [len(tensors)] + padded_dim
	padded_tensor = pad_index * torch.ones(padded_dim)
	padded_tensor = padded_tensor.type_as(rep)
	for i, tensor in enumerate(tensors):
		size = list(tensor.size())
		if len(size) == 1:
			padded_tensor[i, :size[0]] = tensor
		elif len(size) == 2:
			padded_tensor[i, :size[0], :size[1]] = tensor
		elif len(size) == 3:
			padded_tensor[i, :size[0], :size[1], :size[2]] = tensor
		else:
			raise ValueError('Padding is supported for upto 3D tensors at max.')
	return padded_tensor


class AggDataset(Dataset):

	def __init__(self, filename, max_len, global_model, PE_model, key_type, example_type):

		f = open(filename, 'r')
		lines = f.read().splitlines()
		if max_len is not None:
			selected_lines = random.sample(lines, max_len)
			lines = selected_lines
		self.global_model = global_model
		self.PE_model = PE_model
		self.lines = lines
		self.key_type = key_type
		self.example_type = example_type

	def __len__(self):
		return len(self.lines)

	def __getitem__(self, idx):
		return self.generate_prog_data(self.lines[idx])

	def generate_prog_data(self, line):
		data = json.loads(line.rstrip())
		examples = Example.from_line(data)
		env = ProgramEnv(examples) # all examples
		global_env = env.copy()
		program = Program.parse(data['global_program'])
		PE_solution_scores = data['PE_solution_scores']
		PE_satisfiability_set = data['PE_true_indices']


		PE_states = []
		PE_statements = []
		PE_operators = []
		PE_sc = []

		global_state, global_statement, global_operator, failed_env = self.generate_ground_truth_data(program,
														global_env, self.global_model)

		PE_solutions = data['PE_solutions']

		for j in range(len(PE_solutions)):
			PE_program = PE_solutions[j]['result']
			#print(PE_program)
			if PE_program != 'Failed':
				#print(PE_program)
				PE_program = Program.parse(PE_program)
				#print(PE_program)
				if PE_program:
					if self.key_type =='sig': #all examples and global model
						reqd_envs = [env.copy()]
						reqd_model = self.global_model

					if self.key_type =='sii':

						if self.example_type == 'set': #take examples in a set
							PE = [Example.from_dict(data['examples'][k-1]) for k in PE_satisfiability_set[j]]
							PE_env = ProgramEnv(PE)
							reqd_envs = [PE_env]
							reqd_model = self.global_model

						if self.example_type == 'all': #take examples individually
							reqd_envs = []
							#print(PE_satisfiability_set[j])
							for k in PE_satisfiability_set[j]:
								PE = [Example.from_dict(data['examples'][k-1])]
								PE_env = ProgramEnv(PE)
								reqd_envs.append(PE_env)
							reqd_model = self.PE_model

					if self.key_type =='sij': #the set version is equal to sig. So, there is only the all version here
						reqd_envs = []
						for k in [1, 2, 3, 4, 5]:
							PE = [Example.from_dict(data['examples'][k-1])]
							PE_env = ProgramEnv(PE)
							reqd_envs.append(PE_env)
						reqd_model = self.PE_model

					for reqd_env in reqd_envs:

						PE_state, PE_statement, PE_operator, failed_env = self.generate_ground_truth_data(PE_program, reqd_env, reqd_model)
						if not failed_env:
							PE_states.append(PE_state)
							PE_statements.append(PE_statement)
							PE_operators.append(PE_operator)
							step_wise_scores = []
							for l in range(len(PE_state)):
								step_wise_scores.append(PE_solution_scores[j])
							PE_sc.append(step_wise_scores)

		if not PE_statements: #this is required else ints_to_tensor throws an error of empty list
			PE_states.append([torch.zeros(params.state_dim)])
			PE_statements.append([params.num_statements])
			PE_operators.append([params.num_operators])
			PE_sc.append([0.0])

		return global_state, global_statement, global_operator, PE_states, PE_statements, PE_operators, PE_sc

	def generate_ground_truth_data(self, program, env, model):
		inputs = []
		statements = []
		operators = []
		failed_env=False

		for i, statement in enumerate(program.statements):
			inputs.append(model.encoder(torch.LongTensor([env.get_encoding()])).view(-1))

			# Translate absolute indices to post-drop indices
			f, args = statement.function, list(statement.args)
			for j, arg in enumerate(args):
				if isinstance(arg, int):
					args[j] = env.real_var_idxs.index(arg)

			statement = Statement(f, args)
			statements.append(statement_to_index[statement])

			used_args = []
			for next_statement in program.statements[i:]:
				used_args += [x for x in next_statement.args if isinstance(x, int)]

			to_drop = []
			for j in range(params.max_program_vars):
				if j >= env.num_vars or env.real_var_idxs[j] not in used_args:
					to_drop.append(1)
				else:
					to_drop.append(0)

			operator = Operator.from_statement(statement)
			operators.append(operator_to_index[operator])

			if env.num_vars < params.max_program_vars:
				try:
					env.step(statement)
				except(NullInputError, OutputOutOfRangeError):
					failed_env = True
					break
			else:
				try:
					# Choose a random var (that is not used anymore) to drop.
					env.step(statement, random.choice([j for j in range(len(to_drop)) if to_drop[j] > 0]))
				except(NullInputError, OutputOutOfRangeError):
					failed_env = True
					break

		return inputs, statements, operators, failed_env

def collate_fn(data):

	global_states, global_statements, global_operators, PE_states, PE_statements,\
			PE_operators, PE_solution_scores = zip(*data)


	global_states = ints_to_tensor(list(global_states)) #view(-1, 256) before detach to flatten max_len dim
	global_statements = ints_to_tensor(list(global_statements), pad_index=params.num_statements)#.view(-1)
	global_operators = ints_to_tensor(list(global_operators), pad_index =params.num_operators) #.view(-1)
	PE_states = ints_to_tensor(list(PE_states)) #.view(-1, 5, 256)
	PE_statements =  ints_to_tensor(list(PE_statements), pad_index=params.num_statements)#.view(-1, 5)
	PE_operators = ints_to_tensor(list(PE_operators), pad_index=params.num_operators) #.view(-1, 5)
	PE_solution_scores = ints_to_tensor(list(PE_solution_scores)) #.view(-1, 5)

	return global_states, global_statements, global_operators, PE_states, PE_statements,\
		PE_operators, PE_solution_scores