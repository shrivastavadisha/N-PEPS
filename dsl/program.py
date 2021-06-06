from dsl.types import INT, LIST
from dsl.impl import NAME2FUNC
from dsl.value import NULLVALUE
from env.statement import Statement
###########################Added by me starts##############################
import json
import params
from env.env import ProgramEnv
from env.operator import Operator, operator_to_index
from env.statement import Statement, statement_to_index
from dsl.example import Example
###########################Added by me ends ##############################

def get_used_indices(program):
    used = set()
    for statement in program.statements:
        used |= set(statement.args)
    return used


def get_unused_indices(program):
    """Returns unused indices of variables/statements in program."""
    used = get_used_indices(program)
    all_indices = set(range(len(program.var_types) - 1))
    return all_indices - used


class Program(object):
    """
    Attributes:
        input_types: List of Type (INT,LIST) representing the inputs
        statements: List of statements that were done so far.
    """
    def __init__(self, input_types, statements):
        self.input_types = input_types
        self.statements = statements
        self.var_types = self.input_types + [statement.output_type for statement in self.statements]
        self._encoded = None

    def encode(self):
        toks = [x.name for x in self.input_types]
        for statement in self.statements:
            parts = [x for x in [statement.function] + list(statement.args) if x is not None]
            tok = ','.join(map(str, parts))
            toks.append(tok)

        return '|'.join(toks)

    @property
    def encoded(self):
        if self._encoded is None:
            self._encoded = self.encode()
        return self._encoded

    def __str__(self):
        return self.encoded

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return self.encoded < other.encoded

    def __len__(self):
        return len(self.statements)

    def __hash__(self):
        return hash(self.encoded)

    @classmethod
    def parse(cls, encoding):
        input_types = []
        statements = []

        def get_statement(term):
            args = []
            parts = term.split(',')

            lambd = None
            func = NAME2FUNC[parts[0]]

            for inner in parts[1:]:
                if inner.isdigit():
                    args.append(int(inner))
                else:
                    args.append(NAME2FUNC[inner])

            return Statement(func, args)

        for tok in encoding.split('|'):
            if ',' in tok:
                statements.append(get_statement(tok))
            else:
                if tok == INT.name:
                    typ = INT
                elif tok == LIST.name:
                    typ = LIST
                else:
                    raise ValueError('invalid input type {}'.format(tok))
                input_types.append(typ)

        return Program(input_types, statements)

    def __call__(self, *inputs):
        if not self.statements:
            return NULLVALUE
        vals = list(inputs)
        for statement in self.statements:
            args = []
            for arg in statement.args:
                if isinstance(arg, int):
                    args.append(vals[arg])
                else:
                    args.append(arg)
            val = statement.function(*args)
            vals.append(val)
        return vals[-1]


##################################### Added by me starts ##############################################################
def generate_prog_data(line):
    data = json.loads(line.rstrip())
    examples = Example.from_line(data)
    env = ProgramEnv(examples)
    program = Program.parse(data['program'])

    inputs = []
    statements = []
    drop = []
    operators = []
    for i, statement in enumerate(program.statements):
        print(statement)
        inputs.append(env.get_encoding())

        print(inputs, inputs[0].shape)
        # Translate absolute indices to post-drop indices
        f, args = statement.function, list(statement.args)
        for j, arg in enumerate(args):
            if isinstance(arg, int):
                args[j] = env.real_var_idxs.index(arg)

        statement = Statement(f, args)
        print(statement)
        statements.append(statement_to_index[statement])

        print(statements)
        used_args = []
        for next_statement in program.statements[i:]:
            used_args += [x for x in next_statement.args if isinstance(x, int)]

        to_drop = []
        for j in range(params.max_program_vars):
            if j >= env.num_vars or env.real_var_idxs[j] not in used_args:
                to_drop.append(1)
            else:
                to_drop.append(0)

        drop.append(to_drop)
        print(drop)

        operator = Operator.from_statement(statement)
        print(operator)
        operators.append(operator_to_index[operator])

        if env.num_vars < params.max_program_vars:
            env.step(statement)
        else:
            # Choose a random var (that is not used anymore) to drop.
            env.step(statement, random.choice([j for j in range(len(to_drop)) if to_drop[j] > 0]))
        print(operators)
    return inputs, statements, drop, operators


if __name__ == '__main__':

    f = open('train_dataset', 'r')
    max_len = 1
    lines = f.read().splitlines()
    lines = lines[:max_len]

    for line in lines:
        res = generate_prog_data(line)

##################################### Added by me starts ##############################################################