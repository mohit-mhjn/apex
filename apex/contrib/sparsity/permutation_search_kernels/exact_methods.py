import logging
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from .permutation_utilities import sum_after_2_to_4

try:
    import gurobipy as grb
except ModuleNotFoundError:
    print("Gurobi Installation Required!")
except:
    print("Unknown Error with Gurobi! Execution Stalled.")

# ==================== BASE ===================
logging.basicConfig()
logger = logging.getLogger(__name__ + ': ')
logger.setLevel(logging.ERROR)


# ============== Traceback Mechanics =====================
class ModelNotSolvedException(Exception):

    def __init__(self):
        pass


class NotInitializedException(Exception):

    def __init__(self):
        pass


# ==============  CORE  =====================


class OptimizationModel(object):
    """
    Base Class for Mathematical Programming Type Formulations for 2-4 Sparsity.
    You can access generic objects like input_matrix, pre-processing routines and then
    post processing stuff like sparsity mask construction and building output from partitions,
     pretty print, output validation etc.
    For the inherited classes define the following things:
        method: create_decision_variables
        method: create_constraints
        method: set_objective_function
        method: solve >> You got to do this | If you're using other ways - make sure this method is your main()
    or whatever way you'd want to make those partitions.
    Just register your outputs back to your inherited class into self.solutions as list of list of column indexes
    with each internal list corresponding to a partition of size = 4, and that's it!
    For facility use the method - self.add_solution()
    """

    # ================== Facility Methods ==================
    @staticmethod
    def create_input_matrix(number_of_rows, number_of_columns, seed=None, p=0.3):
        """
        Generic method to generate a problem instance
        :return: None, assigns self.input_matrix
        """
        logger.info("!! GENERATED A INPUT MATRIX !!")

        if seed:
            np.random.seed(seed)
        _matrix = np.zeros([number_of_rows, number_of_columns])
        for r_indx in range(number_of_rows):
            for c_indx in range(number_of_columns):
                domain_selector = np.random.uniform(0, 1)
                if domain_selector > p:
                    _matrix[r_indx][c_indx] = round(np.random.uniform(0.001, 0.01), 2)
                else:
                    _matrix[r_indx][c_indx] = round(np.random.uniform(0.1, 1), 2)
        return _matrix

    @staticmethod
    def find_min_two(array):
        """
        Returns indexes of minimum two entries in the array
        :param array:
        :return:
        """
        least = 99999
        least_index = 99999  # infinity wrt scope
        second_least = 99999
        second_least_index = 99999  # infinity wrt scope

        for idx, entry in enumerate(array):
            if entry <= least:
                # swap >>
                second_least = least
                second_least_index = least_index
                # Assign >>
                least = entry
                least_index = idx

            elif entry <= second_least:
                second_least = entry
                second_least_index = idx

        return least_index, second_least_index

    def __init__(self, input_matrix, matrix_layout="row_major"):

        assert matrix_layout in ["row_major", "column_major"], "matrix_layout is row_major or column_major"

        if matrix_layout == "row_major":
            logger.info(""" The input matrix is converted to COLUMN-MAJOR
                    Note: A 2D column major input matrix differs from a usual matrix in terms of indexing.
                    An access for index i,j using M[i][j] returns the value from column - i and row - j """)
            self.input_matrix = np.transpose(input_matrix)
        else:
            self.input_matrix = input_matrix

        self.number_of_columns = self.input_matrix.shape[0]
        self.number_of_rows = self.input_matrix.shape[1]
        self.number_of_partitions = int(self.number_of_columns / 4)
        logger.info(f"Problem Dimensions: Columns = {self.number_of_columns}, Rows = {self.number_of_rows}")

        # Parameters
        self.config = {
            "write_lp": False,
            "write_log": False,
            "display_log": False,
            "mip_gap": 0.001,
            "time_limit": 3600,
        }

        # This has to be defined for each inherited formulation >>
        self.model_solved = False
        self.model_timeout = False
        self.display_log = False
        self.mip_gap = self.config["mip_gap"]  # percent
        self.time_limit = self.config["time_limit"]

        # Times >>
        self.optimization_time = 0
        self.preprocessing_time = 0
        self.postprocessing_time = 0
        # If required to track other times, do so in the inherited classes

        # Solution Containers >>
        self.solutions = []  # Retrieved from the inherited model, container of partition definitions
        self.optimal_partitions = []  # Container of partition objects >> Auto_construct method fills this in

        # Internals >>
        self._is_subset = False
        self._column_mapper = {}  # to map the subset solutions back to the original form

    def get_partition_cost(self, i, j, k, l):
        """
        Utility function to return the cost of a partition with columns i,j,k,l
        """
        cols = self.number_of_columns
        rows = self.number_of_rows
        s = self.input_matrix

        cost_of_delete = 0
        _key = cols * cols * cols * i + cols * cols * j + cols * k + l
        for r_indx in range(rows):
            _array = [s[i][r_indx], s[j][r_indx],
                      s[k][r_indx], s[l][r_indx]]

            temp1, temp2 = self.find_min_two(_array)
            cost_of_delete += _array[temp2] + _array[temp1]

        return _key, cost_of_delete

    def get_partition_cost_cuda(self, i, j, k, l):
        """
        Utility function to return the cost of a partition with columns i,j,k,l computed in parallel
        """
        cols = self.number_of_columns
        rows = self.number_of_rows
        s = self.input_matrix[[i, j, k, l], :]
        _key = cols * cols * cols * i + cols * cols * j + cols * k + l
        mag = sum_after_2_to_4(np.transpose(s))
        return _key, np.sum(s) - mag

    def subset_columns(self, columns):
        """
        Sometimes one would like to execute the model on a subset of columns rather than all
        This method would do the necessary adjustments, just call this before solve
        :param columns:
        :return:
        """
        logger.warning("Input Matrix will be subset based on these columns in this object, "
                       "generate costs methods should be utilized with care!")
        assert len(columns) % 4 == 0, "Columns must be in the multiple of 4!"
        self.number_of_columns = len(columns)
        self.number_of_partitions = int(self.number_of_columns / 4)
        columns = sorted(columns)

        # Subset Input matrix
        logger.debug("Subsetting input matrix to the necessary columns")
        self.input_matrix = self.input_matrix[columns, :]
        self._column_mapper = {i: columns[i] for i in range(self.number_of_columns)}
        self._is_subset = True

    def add_solution(self, solution):
        assert isinstance(solution, list), "Solution must be list of int's"
        assert len(solution) == 4, "Size of a partition is 4"
        if not self.solutions:
            self.solutions = []
        if self._is_subset:
            solution = [self._column_mapper[c] for c in solution]
        self.solutions.append(solution)
        return

    def show_input_matrix(self):
        a_df = pd.DataFrame(self.input_matrix)
        a_df.columns = list(range(a_df.shape[1]))
        print(a_df)

    # # ================== Optimization ==================
    # Template for inheritors

    def _create_decision_variables(self):
        # Define decision variables for the model
        pass

    def _create_main_constraints(self):
        # Preprocess parameters and define constraints for the model
        pass

    def _set_objective_function(self):
        # Define objective function
        pass

    def solve(self):
        """
        A generic solution method for mathematical programming subclasses
        :return:
        """

    # # ================== Output ==================
    def check_solve_status(self):

        if self.model_timeout:
            logger.error("Model was timed out :(")
            raise ModelNotSolvedException

        if not self.model_solved:
            logger.error("Model hasn't been solved yet!")
            raise ModelNotSolvedException

        if not self.solutions:
            logger.error("Solution Not found!\n\nIdentify and load the solutions with add_solutions\n")
            raise ModelNotSolvedException

        return True

    def get_apex_solution(self):

        assert self.check_solve_status(), "Unable to verify model solve!"

        if self._is_subset:
            logger.error("NVIDIA Apex integration doesn't permit solving on subset "
                         "of columns, pls execute solve for the full matrix")
            raise ModelNotSolvedException
        else:
            permutation_sequence = [column_index for partition in self.solutions for column_index in partition]
            duration = self.preprocessing_time + self.optimization_time + self.postprocessing_time
            result = np.transpose(self.input_matrix)[:, permutation_sequence]
        return result, duration, permutation_sequence

    def construct_solution(self):
        assert self.check_solve_status(), "Unable to verify model solve!"

        partitions = []
        for s in self.solutions:
            p = Partitions(*s)
            if self._is_subset:
                p.auto_construct(self.input_matrix, self.number_of_rows,
                                 column_index_mapper={v: k for k, v in self._column_mapper.items()})
            else:
                p.auto_construct(self.input_matrix, self.number_of_rows)
            partitions.append(p)
        self.optimal_partitions = partitions
        return

    def pprint_output(self):
        """
        Pretty Print the output partitioning
        :return: None
        """
        if not self.optimal_partitions:
            logger.error("Solution needs post processing, Run self.construct_solution() method first")

        for partition_number, deletion in enumerate(self.optimal_partitions):

            logger.info(f"Partition : {partition_number + 1}")
            _print_rows = [[0] * self.number_of_rows for _ in range(4)]  # default - no deletions
            for r_indx in range(self.number_of_rows):
                _loc = deletion.indexes_to_delete[r_indx]
                _print_rows[_loc[0]][r_indx] = 1  # Perform deletions in this r_indx
                _print_rows[_loc[1]][r_indx] = 1  # Perform deletions in this r_indx

            _temp_arr = deletion.cols
            for idx, _row in enumerate(_print_rows):
                _piece1 = _row[:16]
                _piece2 = _row[max(len(_row) - 4, 16):]
                if _piece2:
                    str2 = " . . . " + f"{_piece2}"[1:]
                else:
                    str2 = "]"
                logger.info(f"\tColumn: {_temp_arr[idx]} => {_piece1}"[:-1] + str2)
        return

    def validate_output(self):
        """
        Does a validation check on the solved model
        Depends on the definitions of decision variables >> Need to be defined for each formulation
        :return:
        """
        logger.warning(f"This is a base class, No definition for validate_output()")
        pass


class Partitions(object):
    # __slots__ = ("cols","indexes_to_delete", "add_deletion", "cost_del_in_row", "C")

    def __init__(self, i, j, k, l):
        self.cols = [i, j, k, l]
        self.indexes_to_delete = {}
        self.C = 0
        # container for deletions in rows of columns i,j,k,l
        # Keyed by row index >>
        # In a Deletion object for combination i,j,k,l >>
        # In row m two of the cols will be deleted with some deletion cost
        # Sum of deletion cost for all rows is the total deletion cost cijkl
        # self.del_in_row[m] = (two minimum col in i,j,k,l and row m)

    def cols_to_delete(self, r_index):
        """
        For post processing purposes
        :param r_index:
        :return:
        """
        [a, b] = self.indexes_to_delete[r_index]
        return [self.cols[a], self.cols[b]]

    def auto_construct(self, input_matrix, number_of_rows, column_index_mapper=None):
        """
        Provided an input matrix this method automatically reconstructs the deletion configuration for
        post-optimization retrieval and further processing
        :param input_matrix:
        :param number_of_rows:
        :param column_index_mapper: Map columns i,j,k,l to cols p,q,r,s in input matrix, provide a dict of {i:p,j:q .. }
        :return:
        """
        [i, j, k, l] = self.cols
        if column_index_mapper:
            [i, j, k, l] = [column_index_mapper[c] for c in [i, j, k, l]]
        for r_indx in range(number_of_rows):
            cost_of_delete = 0
            _array = [input_matrix[i][r_indx], input_matrix[j][r_indx],
                      input_matrix[k][r_indx], input_matrix[l][r_indx]]

            temp1, temp2 = OptimizationModel.find_min_two(_array)
            cost_of_delete += _array[temp2] + _array[temp1]
            self.add_deletion(r_indx, [temp1, temp2], cost_of_delete)
        return

    def add_deletion(self, row_index, indexes_to_delete, cost_of_delete):
        """
        : arg:
        Add deletion per row for the set of these 4 columns >> Most of the business is done to efficiently
        retrieve solution later
        :param row_index:
        :param indexes_to_delete:
        :param cols_to_delete:
        :param cost_of_delete:
        :return:
        """
        self.indexes_to_delete[row_index] = indexes_to_delete
        self.C += cost_of_delete
        return

    def __repr__(self):
        return "[{},{},{},{}]".format(*self.cols)


# ==============  Quadratic Model  =====================
class BqpModel(OptimizationModel):
    """
    Naive Quadratic version of the naive 2:4 sparsity pruning problem
    """

    def __init__(self, input_matrix):
        logger.info("This is a BQP-formulation")
        super().__init__(input_matrix)
        self.config["BQP"] = {"relax_y_vars": True,
                              "relax_x_vars": False}

        # Model Specific Parameters
        self.relax_y_vars = self.config["BQP"]["relax_y_vars"]
        self.relax_x_vars = self.config["BQP"]["relax_x_vars"]
        # Cannot do this - turns the problem non-convex

        # Model Specific Things
        self.model = grb.Model('24sparsity-BQP')

        # Variables
        self.x_ik = None
        self.y_ij = None

        # Constraints
        self.column_assignment = None
        self.partition_capacity = None
        self.sparse_selector = None

    # ================== Decision variables ==================
    def create_decision_variables(self):
        """
        Creates decision variables of the model based on defined parameters
        called as a part of the solve method
        :return: None, creates model attributes
        """
        logger.info("Creating Decision Vars")
        self.x_ik = defaultdict(dict)
        x_var_type = grb.GRB.BINARY if not self.relax_x_vars else grb.GRB.CONTINUOUS
        y_var_type = grb.GRB.BINARY if not self.relax_y_vars else grb.GRB.CONTINUOUS
        for i in range(self.number_of_columns):
            for k in range(self.number_of_partitions):
                self.x_ik[i][k] = self.model.addVar(name=f'X_{i}_{k}', vtype=x_var_type, ub=1, lb=0)
        # NOTE : Columns are indexed i and rows are indexed j >> cost matrix is a column major

        self.y_ij = defaultdict(dict)
        for i in range(self.number_of_columns):
            for j in range(self.number_of_rows):
                self.y_ij[i][j] = self.model.addVar(name=f'Y_{i}_{j}', vtype=y_var_type, ub=1, lb=0)

    # # ================== Constraints ==================
    def create_constraints(self):
        """
        Add Constraints to the Model
        called as a part of the solve method
        :return: None, generates model attributes
        """
        logger.info("Adding Constraints")
        logger.info("\tLoad : Every Column is assigned to 1 partition")
        self.column_assignment = {i: self.model.addLConstr(
            lhs=grb.quicksum([self.x_ik[i][k] for k in range(self.number_of_partitions)]),
            sense=grb.GRB.EQUAL,
            name=f'col_assign_{i}',
            rhs=1)
            for i in range(self.number_of_columns)}

        logger.info("\tLoad : Every Partition has exactly 4 columns")
        self.partition_capacity = {k: self.model.addLConstr(
            lhs=grb.quicksum([self.x_ik[i][k] for i in range(self.number_of_columns)]),
            sense=grb.GRB.EQUAL,
            name=f'cap_partition_{k}',
            rhs=4)
            for k in range(self.number_of_partitions)}

        logger.info("\tLoad : Delete 2 entries from every row in each partition")
        self.sparse_selector = {(j, k): self.model.addConstr(
            sum(self.x_ik[i][k] * self.y_ij[i][j] for i in range(self.number_of_columns)) == 2,
            f"sparse_selector_{j}_{k}")
            for j in range(self.number_of_rows)
            for k in range(self.number_of_partitions)}

        return

    # # ================== Costs and objective function ==================
    def set_objective_function(self):
        logger.info("Setting Objective Function")
        objective = grb.quicksum(self.input_matrix[i][j] * self.y_ij[i][j]
                                 for i in range(self.number_of_columns)
                                 for j in range(self.number_of_rows))
        self.model.setObjective(objective, grb.GRB.MINIMIZE)
        return

    def solve(self):
        """
        Main method of this formulation
        :return:
        """
        pre_start_time = time.time()
        self.create_decision_variables()
        self.create_constraints()
        self.set_objective_function()
        pre_end_time = time.time()
        self.preprocessing_time += round(pre_end_time - pre_start_time, 3)

        logger.info(f"Solving Model ...")

        if not self.display_log:
            self.model.setParam('LogToConsole', 0)

        self.model.setParam(grb.GRB.Param.MIPGap, self.mip_gap)
        self.model.setParam(grb.GRB.Param.TimeLimit, self.time_limit)

        opt_start_time = time.time()
        self.model.optimize()
        opt_end_time = time.time()
        self.optimization_time += round(opt_end_time - opt_start_time, 3)

        post_start_time = time.time()
        if self.model.getAttr(grb.AttrConstClass.Status) == 2:
            logger.info('The solution is optimal and the objective value '
                        'is {:,.2f}!'.format(self.model.objVal))

            if not self.relax_x_vars:
                for k in range(self.number_of_partitions):
                    _partition = []
                    for i in range(self.number_of_columns):
                        if self.x_ik[i][k].x > 0.5:
                            _partition.append(i)
                    self.add_solution(_partition)
            else:
                logger.warning("CRITICAL: This is a relaxed model - Solutions wouldn't be registered automatically!")
            self.model_solved = True

        elif self.model.getAttr(grb.AttrConstClass.Status) == 9:
            logger.error("The model failed to solve the problem within time limit!")
            self.model_timeout = True

        else:
            logger.error("Unknown exception with the model")
        post_end_time = time.time()
        self.postprocessing_time += round(post_end_time - post_start_time, 3)


# ==============  Linear Model  =====================
class BlpModel(OptimizationModel):
    """
    Linearized version of the naive 2:4 sparsity pruning problem
    """

    # todo: solve LP relaxation of this model

    def __init__(self, input_matrix):
        logger.info("This is a BLP-formulation")
        super().__init__(input_matrix)
        self.config["BLP"] = {"relax_z_vars": True,
                              "relax_x_vars": False}

        # Model Specific Things
        self.model = grb.Model('24sparsity-Linearized')

        # Variables
        self.x_ik = None
        self.z_ijk = None

        # Constraints
        self.column_assignment = None
        self.partition_capacity = None
        self.deletion_flag = None
        self.sparse_selector = None

        # Model Specific Parameters
        self.relax_z_vars = self.config["BLP"]["relax_z_vars"]
        self.relax_x_vars = self.config["BLP"]["relax_x_vars"]

    # ================== Decision variables ==================
    def create_decision_variables(self):
        """
        Creates decision variables of the model based on defined parameters
        called as a part of the solve method
        :return: None, creates model attributes
        """
        logger.info("Creating Decision Vars")
        self.x_ik = defaultdict(dict)
        x_var_type = grb.GRB.BINARY if not self.relax_x_vars else grb.GRB.CONTINUOUS
        z_var_type = grb.GRB.BINARY if not self.relax_z_vars else grb.GRB.CONTINUOUS

        for i in range(self.number_of_columns):
            for k in range(self.number_of_partitions):
                self.x_ik[i][k] = self.model.addVar(name=f'X_{i}_{k}', vtype=x_var_type, ub=1, lb=0)
        # NOTE : Columns are indexed i and rows are indexed j >> cost matrix is a column major

        self.z_ijk = defaultdict(lambda: defaultdict(dict))
        for i in range(self.number_of_columns):
            for j in range(self.number_of_rows):
                for k in range(self.number_of_partitions):
                    self.z_ijk[i][j][k] = self.model.addVar(name=f'Z_{i}_{j}_{k}', vtype=z_var_type, ub=1, lb=0)

    # # ================== Constraints ==================
    def create_constraints(self):
        """
        Add Constraints to the Model
        called as a part of the solve method
        :return: None, generates model attributes
        """
        logger.info("Adding Constraints")

        logger.info("\tLoad : Every Column is assigned to 1 partition")
        self.column_assignment = {i: self.model.addLConstr(
            lhs=grb.quicksum([self.x_ik[i][k] for k in range(self.number_of_partitions)]),
            sense=grb.GRB.EQUAL,
            name=f'col_assign_{i}',
            rhs=1)
            for i in range(self.number_of_columns)}

        logger.info("\tLoad : Every Partition has exactly 4 columns")
        self.partition_capacity = {k: self.model.addLConstr(
            lhs=grb.quicksum([self.x_ik[i][k] for i in range(self.number_of_columns)]),
            sense=grb.GRB.EQUAL,
            name=f'cap_partition_{k}',
            rhs=4)
            for k in range(self.number_of_partitions)}

        logger.info("\tLoad : Delete entry if column i in partition k")
        self.deletion_flag = {(i, j, k): self.model.addLConstr(
            lhs=self.z_ijk[i][j][k],
            sense=grb.GRB.LESS_EQUAL,
            name=f'delete_index_from_partition_{i}_{j}_{k}',
            rhs=self.x_ik[i][k])
            for i in range(self.number_of_columns)
            for j in range(self.number_of_rows)
            for k in range(self.number_of_partitions)}

        logger.info("\tLoad : Delete 2 entries from every row in each partition")
        self.sparse_selector = {(j, k): self.model.addLConstr(
            lhs=grb.quicksum([self.z_ijk[i][j][k] for i in range(self.number_of_columns)]),
            sense=grb.GRB.EQUAL,
            name=f"sparse_selector_{j}_{k}",
            rhs=2)
            for j in range(self.number_of_rows)
            for k in range(self.number_of_partitions)}

        return

    # # ================== Costs and objective function ==================
    def set_objective_function(self):
        logger.info("Setting Objective Function")
        objective = grb.quicksum(self.input_matrix[i][j] * self.z_ijk[i][j][k]
                                 for i in range(self.number_of_columns)
                                 for j in range(self.number_of_rows)
                                 for k in range(self.number_of_partitions))
        self.model.setObjective(objective, grb.GRB.MINIMIZE)
        return

    def solve(self):
        """
        Main method of this formulation
        :return:
        """
        pre_start_time = time.time()
        self.create_decision_variables()
        self.create_constraints()
        self.set_objective_function()
        pre_end_time = time.time()
        self.preprocessing_time += round(pre_end_time - pre_start_time, 3)

        logger.info(f"Solving Model ...")

        if not self.display_log:
            self.model.setParam('LogToConsole', 0)

        self.model.setParam(grb.GRB.Param.MIPGap, self.mip_gap)
        self.model.setParam(grb.GRB.Param.TimeLimit, self.time_limit)

        opt_start_time = time.time()
        self.model.optimize()
        opt_end_time = time.time()
        self.optimization_time += round(opt_end_time - opt_start_time, 3)

        post_start_time = time.time()
        if self.model.getAttr(grb.AttrConstClass.Status) == 2:
            logger.info('The solution is optimal and the objective value '
                        'is {:,.2f}!'.format(self.model.objVal))

            if not self.relax_x_vars:
                for k in range(self.number_of_partitions):
                    _partition = []
                    for i in range(self.number_of_columns):
                        if self.x_ik[i][k].x > 0.5:
                            _partition.append(i)
                    self.add_solution(_partition)
            else:
                logger.warning("CRITICAL: This is a relaxed model - Solutions wouldn't be registered automatically!")
            self.model_solved = True

        elif self.model.getAttr(grb.AttrConstClass.Status) == 9:
            logger.error("The model failed to solve the problem within time limit!")
            self.model_timeout = True
        else:
            logger.error("Unknown exception with the model")
        post_end_time = time.time()
        self.postprocessing_time += round(post_end_time - post_start_time, 3)
        return


# ==============  Assignment MODEL  =====================
class MdaModel(OptimizationModel):
    """
    4-dimensional assignment formulation for 2:4 sparsity pruning
    """

    def __init__(self, input_matrix):
        logger.info("This is a 4-Dimensional Assignment Formulation")
        super().__init__(input_matrix)
        self.config["MDA"] = {"relax_z_vars": True,
                              "relax_x_vars": False}

        # Model Specific Things
        self.model = grb.Model('24sparsity_4DA')

        # Variables >>
        self.x1_ij = defaultdict(dict)
        self.x2_ij = defaultdict(dict)
        self.x3_ij = defaultdict(dict)
        self.z_ijkl = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        # Constraints >>
        self.stage_1_column_assignment = None
        self.stage_2_column_assignment = None
        self.stage_3_column_assignment = None
        self.pairing_constraint_11 = None
        self.pairing_constraint_12 = None
        self.pairing_constraint_21 = None
        self.pairing_constraint_22 = None
        self.pairing_constraint_31 = None
        self.pairing_constraint_32 = None
        self.deletion_flag_stage_1 = None
        self.deletion_flag_stage_2 = None
        self.deletion_flag_stage_3 = None
        self.column_inclusion_all_stages = None

        # Model Specific Parameters
        self.relax_z_vars = self.config["MDA"]["relax_z_vars"]
        self.relax_x_vars = self.config["MDA"]["relax_x_vars"]

    # ================== Decision variables ==================
    def create_decision_variables(self):
        """
        Creates decision variables of the model based on defined parameters
        called as a part of the solve method
        :return: None, creates model attributes
        """
        logger.info("Creating Decision Vars")

        x_var_type = grb.GRB.BINARY if not self.relax_x_vars else grb.GRB.CONTINUOUS
        z_var_type = grb.GRB.BINARY if not self.relax_z_vars else grb.GRB.CONTINUOUS

        for i in range(self.number_of_columns - 3):
            for j in range(i + 1, self.number_of_columns - 2):
                self.x1_ij[i][j] = self.model.addVar(name=f'X1_{i}_{j}', ub=1, lb=0, vtype=x_var_type)
        # NOTE : number_of_columns are indexed i and rows are indexed j >> cost matrix is a column major

        for i in range(1, self.number_of_columns - 2):
            for j in range(i + 1, self.number_of_columns - 1):
                self.x2_ij[i][j] = self.model.addVar(name=f'X2_{i}_{j}', ub=1, lb=0, vtype=x_var_type)

        for i in range(2, self.number_of_columns - 1):
            for j in range(i + 1, self.number_of_columns):
                self.x3_ij[i][j] = self.model.addVar(name=f'X3_{i}_{j}', ub=1, lb=0, vtype=x_var_type)

        # Initialize z
        for i in range(self.number_of_columns - 3):
            for j in range(i + 1, self.number_of_columns - 2):
                for k in range(j + 1, self.number_of_columns - 1):
                    for l in range(k + 1, self.number_of_columns):
                        key, del_cost = self.get_partition_cost(i, j, k, l)
                        self.z_ijkl[i][j][k][l] = self.model.addVar(
                            name=f'z_{i}_{j}_{k}_{l}', vtype=z_var_type, ub=1, lb=0, obj=del_cost)

    # # ================== Constraints ==================
    def create_constraints(self):
        """
        Add Constraints to the Model
        called as a part of the solve method
        :return: None, generates model attributes
        """

        logger.info("Adding Constraints")
        logger.info("\tLoad : There are exactly k (= number_of_partitions) pairings in each stage")
        self.stage_1_column_assignment = self.model.addLConstr(
            lhs=grb.quicksum(
                [self.x1_ij[i][j] for i in range(self.number_of_columns - 3) for j in
                 range(i + 1, self.number_of_columns - 2)]),
            sense=grb.GRB.EQUAL,
            name=f'stage_1_col_assign',
            rhs=self.number_of_partitions)

        self.stage_2_column_assignment = self.model.addLConstr(
            lhs=grb.quicksum(
                [self.x2_ij[j][k] for j in range(1, self.number_of_columns - 2) for k in
                 range(j + 1, self.number_of_columns - 1)]),
            sense=grb.GRB.EQUAL,
            name=f'stage_2_col_assign',
            rhs=self.number_of_partitions)

        self.stage_3_column_assignment = self.model.addLConstr(
            lhs=grb.quicksum(
                [self.x3_ij[k][l] for k in range(2, self.number_of_columns - 1) for l in
                 range(k + 1, self.number_of_columns)]),
            sense=grb.GRB.EQUAL,
            name=f'stage_3_col_assign',
            rhs=self.number_of_partitions)

        logger.info("\tLoad : A column is paired with at most one other column in each stage")
        self.pairing_constraint_11 = {i: self.model.addLConstr(
            lhs=grb.quicksum([self.x1_ij[i][j] for j in range(i + 1, self.number_of_columns - 2)]),
            sense=grb.GRB.LESS_EQUAL,
            name=f'pairing_constraint_11_{i}',
            rhs=1)
            for i in range(self.number_of_columns - 3)}

        self.pairing_constraint_12 = {j: self.model.addLConstr(
            lhs=grb.quicksum([self.x1_ij[i][j] for i in range(j)]),
            sense=grb.GRB.LESS_EQUAL,
            name=f'pairing_constraint_12_{j}',
            rhs=1)
            for j in range(1, self.number_of_columns - 2)}

        self.pairing_constraint_21 = {j: self.model.addLConstr(
            lhs=grb.quicksum([self.x2_ij[j][k] for k in range(j + 1, self.number_of_columns - 1)]),
            sense=grb.GRB.LESS_EQUAL,
            name=f'pairing_constraint_21_{j}',
            rhs=1)
            for j in range(1, self.number_of_columns - 2)}

        self.pairing_constraint_22 = {k: self.model.addLConstr(
            lhs=grb.quicksum([self.x2_ij[j][k] for j in range(1, k)]),
            sense=grb.GRB.LESS_EQUAL,
            name=f'pairing_constraint_22_{k}',
            rhs=1)
            for k in range(2, self.number_of_columns - 1)}

        self.pairing_constraint_31 = {k: self.model.addLConstr(
            lhs=grb.quicksum([self.x3_ij[k][l] for l in range(k + 1, self.number_of_columns)]),
            sense=grb.GRB.LESS_EQUAL,
            name=f'pairing_constraint_31_{k}',
            rhs=1)
            for k in range(2, self.number_of_columns - 1)}

        self.pairing_constraint_32 = {l: self.model.addLConstr(
            lhs=grb.quicksum([self.x3_ij[k][l] for k in range(2, l)]),
            sense=grb.GRB.LESS_EQUAL,
            name=f'pairing_constraint_32_{l}',
            rhs=1)
            for l in range(3, self.number_of_columns)}

        logger.info(
            "\tLoad : Delete entry from combination i,j,k,l if column i matched "
            "with j matched with k matched with l")
        self.deletion_flag_stage_1 = {(i, j): self.model.addLConstr(
            lhs=grb.quicksum(
                self.z_ijkl[i][j][k][l] for k in range(j + 1, self.number_of_columns - 1) for l in
                range(k + 1, self.number_of_columns)),
            sense=grb.GRB.EQUAL,
            name=f'delete_index_from_partition_stage_1_{i}_{j}',
            rhs=self.x1_ij[i][j])
            for i in range(self.number_of_columns - 3)
            for j in range(i + 1, self.number_of_columns - 2)}

        self.deletion_flag_stage_2 = {(j, k): self.model.addLConstr(
            lhs=grb.quicksum(self.z_ijkl[i][j][k][l] for i in range(j) for l in range(k + 1, self.number_of_columns)),
            sense=grb.GRB.EQUAL,
            name=f'delete_index_from_partition_stage_2_{j}_{k}',
            rhs=self.x2_ij[j][k])
            for j in range(1, self.number_of_columns - 2)
            for k in range(j + 1, self.number_of_columns - 1)}

        self.deletion_flag_stage_3 = {(k, l): self.model.addLConstr(
            lhs=grb.quicksum(self.z_ijkl[i][j][k][l] for j in range(1, k) for i in range(j)),
            sense=grb.GRB.EQUAL,
            name=f'delete_index_from_partition_stage_3_{k}_{l}',
            rhs=self.x3_ij[k][l])
            for k in range(2, self.number_of_columns - 1)
            for l in range(k + 1, self.number_of_columns)}

        logger.info(
            "\tLoad : Every column must be included in one and exactly one ijkl pairing through all stages")
        self.column_inclusion_all_stages = {_n: self.model.addLConstr(
            lhs=grb.quicksum([
                sum([self.z_ijkl[_n][j][k][l]
                     for j in range(_n + 1, self.number_of_columns - 2)
                     for k in range(j + 1, self.number_of_columns - 1)
                     for l in range(k + 1, self.number_of_columns)
                     ]) +
                sum([self.z_ijkl[i][_n][k][l]
                     for i in range(0, _n)
                     for k in range(_n + 1, self.number_of_columns - 1)
                     for l in range(k + 1, self.number_of_columns)
                     ]) +
                sum([self.z_ijkl[i][j][_n][l]
                     for i in range(0, _n - 1)
                     for j in range(i + 1, _n)
                     for l in range(_n + 1, self.number_of_columns)]) +
                sum([self.z_ijkl[i][j][k][_n]
                     for i in range(0, _n - 2)
                     for j in range(i + 1, _n - 1)
                     for k in range(j + 1, _n)
                     ])
            ]),
            sense=grb.GRB.EQUAL,
            name=f"inclusion_for_column_{_n}",
            rhs=1)
            for _n in range(self.number_of_columns)}  # need to be adjusted for boundaries >>
        return

    # # ================== Costs and objective function ==================

    def set_objective_function(self):
        logger.info("Setting Objective Function")
        # Not required >> objective function is set simultaneously with variables
        return

    # # ================== Solve ==================

    def solve(self):
        """
        Reconstruct solution objects with necessary details
        :return:
        """
        pre_start_time = time.time()
        self.create_decision_variables()
        self.create_constraints()
        self.set_objective_function()
        pre_end_time = time.time()
        self.preprocessing_time += round(pre_end_time - pre_start_time, 3)

        logger.info(f"Solving Model ...")

        if not self.display_log:
            self.model.setParam('LogToConsole', 0)

        self.model.setParam(grb.GRB.Param.MIPGap, self.mip_gap)
        self.model.setParam(grb.GRB.Param.TimeLimit, self.time_limit)

        opt_start_time = time.time()
        self.model.optimize()
        opt_end_time = time.time()
        self.optimization_time += round(opt_end_time - opt_start_time, 3)

        post_start_time = time.time()
        if self.model.getAttr(grb.AttrConstClass.Status) == 2:
            logger.info('The solution is optimal and the objective value '
                        'is {:,.2f}!'.format(self.model.objVal))

            if not self.relax_x_vars:
                for i in range(self.number_of_columns - 3):
                    for j in range(i + 1, self.number_of_columns - 2):
                        for k in range(j + 1, self.number_of_columns - 1):
                            for l in range(k + 1, self.number_of_columns):
                                if self.z_ijkl[i][j][k][l].x > 0.5:
                                    self.add_solution([i, j, k, l])
            else:
                logger.warning("CRITICAL: This is a relaxed model - Solutions wouldn't be registered automatically!")
            self.model_solved = True

        elif self.model.getAttr(grb.AttrConstClass.Status) == 9:
            logger.error("The model failed to solve the problem within time limit!")
            self.model_timeout = True

        else:
            logger.error("Unknown exception with the model")
        post_end_time = time.time()
        self.postprocessing_time += round(post_end_time - post_start_time, 3)
        return


# ==============  SET PACKING MODEL  =====================
class SetPartitionModel(OptimizationModel):
    """
    set-packing version of the 2:4 sparsity pruning problem
    """

    def __init__(self, input_matrix, **kwargs):
        logger.info("This is a set-packing Formulation")
        super().__init__(input_matrix, **kwargs)
        self.config["SETPART"] = {"relax_x_vars": False}

        self.model = grb.Model('24sparsity_setPack')

        # Variables >>
        self.xi = defaultdict()

        # Constraints >>
        self.membership_in_unique_selection = None

        # Model Specific Parameters >>
        self.relax_x_vars = self.config["SETPART"]["relax_x_vars"]

        # Internals >>
        self._memberships = defaultdict(set)
        self._solution_sample = []
        self._all_costs = []

    # # ================== Decision variables ==================
    def create_decision_variables(self, restricted_column_set):
        """
        Creates decision variables of the model based on defined parameters
        called as a part of the solve method
        :return: None, creates model attributes
        """
        logger.info("Creating Decision Vars")
        # If not provided use all combinations else subset solution space
        if restricted_column_set:
            assert isinstance(restricted_column_set, list), "Restricted column set must be a list of partitions"
            self._solution_sample = restricted_column_set
        else:
            self._solution_sample = combinations(range(self.number_of_columns), 4)

        x_var_type = grb.GRB.BINARY if not self.relax_x_vars else grb.GRB.CONTINUOUS
        for q in self._solution_sample:
            # Generate costs and insert variables on the fly
            key, cost = self.get_partition_cost(*q)
            self._all_costs.append(cost)
            self.xi[key] = self.model.addVar(name=f"X_{key}", vtype=x_var_type, obj=cost, lb=0, ub=1)
            for c in q:
                self._memberships[c].add(key)

    # # # ================== Constraints ==================
    def create_constraints(self):
        """
        Add Constraints to the Model
        called as a part of the solve method
        :return: None, generates model attributes
        """
        logger.info("Adding Constraints")
        logger.info("\tLoad : A column must belong to exactly one partition")

        self.membership_in_unique_selection = {i: self.model.addLConstr(
            lhs=grb.quicksum(
                [self.xi[m] for m in self._memberships[i]]),
            sense=grb.GRB.EQUAL,
            name=f'uniqueness_for_column_{i}',
            rhs=1)
            for i in range(self.number_of_columns)}

        return

    # # # ================== Costs and objective function ==================
    # Objective Function set through obj argument during variable creation >> Not Required!

    def set_objective_function(self):
        logger.info("Setting Objective Function")
        # Already done while creating variables
        return

    def solve(self, restricted_column_set=None):

        if restricted_column_set is None:
            restricted_column_set = []

        pre_start_time = time.time()
        self.create_decision_variables(restricted_column_set)
        self.create_constraints()
        self.set_objective_function()
        pre_end_time = time.time()
        self.preprocessing_time += round(pre_end_time - pre_start_time, 3)

        logger.info(f"Solving Model ...")

        if not self.display_log:
            self.model.setParam('LogToConsole', 0)

        self.model.setParam(grb.GRB.Param.MIPGap, self.mip_gap)
        self.model.setParam(grb.GRB.Param.TimeLimit, self.time_limit)

        opt_start_time = time.time()
        self.model.optimize()
        opt_end_time = time.time()
        self.optimization_time += round(opt_end_time - opt_start_time, 3)

        post_start_time = time.time()
        if self.model.getAttr(grb.AttrConstClass.Status) == 2:
            logger.info('The solution is optimal and the objective value '
                        'is {:,.2f}!'.format(self.model.objVal))

            if not self.relax_x_vars:
                cols = self.number_of_columns
                # Iterator must be empty by now >>
                if not restricted_column_set:
                    self._solution_sample = combinations(range(cols), 4)
                for i, j, k, l in self._solution_sample:
                    indx = cols * cols * cols * i + cols * cols * j + cols * k + l
                    if self.xi[indx].x > 0.5:
                        self.add_solution([i, j, k, l])
            else:
                logger.warning("CRITICAL: This is a relaxed model - Solutions wouldn't be registered automatically!")
            self.model_solved = True

        elif self.model.getAttr(grb.AttrConstClass.Status) == 9:
            logger.error("The model failed to solve the problem within time limit!")
            self.model_timeout = True

        else:
            logger.error("Unknown exception with the model")
        post_end_time = time.time()
        self.postprocessing_time += round(post_end_time - post_start_time, 3)
        return

def call_SetPartition(input_matrix):
    model = SetPartitionModel(input_matrix)
    model.solve()
    return model.get_apex_solution()


if __name__ == "__main__":

    logger.info(" ================== TEST 1 => ")
    matrix = OptimizationModel.create_input_matrix(24, 24)
    opt = SetPartitionModel(matrix)
    opt.solve()
    opt.construct_solution()

    logger.info(" *********************************** TESTING baseClass\n\n")

    matrix = OptimizationModel.create_input_matrix(64, 32, None)
    opt = OptimizationModel(matrix)
    opt.subset_columns([4, 5, 6, 7, 8, 9, 10, 11])
    # opt.show_input_matrix()
    opt.solve()
    opt.add_solution([1, 2, 3, 7])
    opt.add_solution([5, 6, 4, 0])

    opt.solutions.append([1, 2, 3, 4])
    try:
        opt.construct_solution()
    except ModelNotSolvedException:
        pass
    opt.pprint_output()

    logger.info(" *********************************** Testing BQP Model Formulation")
    logger.info(" ================== TEST 1 => ")
    matrix = OptimizationModel.create_input_matrix(12, 12)
    opt = BqpModel(matrix)
    opt.solve()
    opt.construct_solution()
    opt.pprint_output()

    logger.info(" ================== TEST 2 => ")
    matrix = OptimizationModel.create_input_matrix(24, 24)
    opt = BqpModel(matrix)
    opt.subset_columns([8, 9, 10, 11, 12, 13, 14, 15])
    opt.solve()
    opt.construct_solution()
    opt.pprint_output()

    logger.info(" *********************************** Testing BLP Model Formulation")
    logger.info(" ================== TEST 1 => ")
    matrix = OptimizationModel.create_input_matrix(16, 16)
    opt = BlpModel(matrix)
    opt.subset_columns([0, 1, 2, 6, 7, 8, 11, 12])
    opt.solve()
    opt.construct_solution()
    opt.pprint_output()

    logger.info(" ================== TEST 2 => ")
    matrix = OptimizationModel.create_input_matrix(20, 20)
    opt = BlpModel(matrix)
    opt.solve()
    opt.construct_solution()
    opt.pprint_output()

    logger.info(" *********************************** Testing Set Packing Model Formulation")
    logger.info(" ================== TEST 1 => ")
    matrix = OptimizationModel.create_input_matrix(64, 32)
    opt = SetPartitionModel(matrix)
    opt.solve()
    # opt.solve(restricted_column_set=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]])
    opt.construct_solution()
    opt.pprint_output()

# Std dev of partition costs
# weakness of lower bound
#