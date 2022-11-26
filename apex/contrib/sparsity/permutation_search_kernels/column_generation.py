"""
Column Generation Method applied to 2:4 sparsity permutation problem
"""
import random
from .permutation_utilities import *
from .exact_methods import *

try:
    import gurobipy as grb
except ModuleNotFoundError:
    print("Gurobi Installation Required!")
except:
    print("Unknown Error with Gurobi! Execution Stalled.")


# ==============  CORE  =====================
class MasterProblem(SetPartitionModel):

    def __init__(self, input_matrix):
        logger.info("This is a CG Master Problem")
        super().__init__(input_matrix, matrix_layout="column_major")
        self.relax_x_vars = True

    def extend_model_variable(self, partition):
        """
        Plugin for column generation method
        :param partition:
        :return:
        """
        assert isinstance(partition, list), "Partition must be list of column indices"
        assert len(partition) == 4, "A partition has exactly 4 columns"
        assert all([c < self.number_of_columns for c in partition]), "Column indices out of scope"
        assert self.model_solved, "This method is only applicable on a solved model"
        
        _key, _cost = self.get_partition_cost_cuda(*partition)
        
        # Do not add duplicate solutions --> 
        if not _key in self._key_partition_dict:
            
            new_column = grb.Column()
            
            # Equivalent to adding in memberships >>
            new_column.addTerms([1, 1, 1, 1], [self.membership_in_unique_selection[c] for c in partition])

            self.xi[_key] = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS,
                                            obj=_cost,
                                            column=new_column, name=f"new_var_subproblem_{_key}")
            # update to keys dict -
            self._key_partition_dict[_key] = partition
            for c in partition:
                self._memberships[c].add(_key)
            return True
        else:
            return False

    def re_solve(self):
        assert self.model_solved, "Only solved model can re_solve"
        logger.info(f"Solving Model ...")

        if not self.display_log:
            self.model.setParam('LogToConsole', 0)

        self.model.setParam(grb.GRB.Param.MIPGap, self.mip_gap)
        self.model.setParam(grb.GRB.Param.TimeLimit, self.time_limit)

        opt_start_time = time.time()
        self.model.optimize()
        opt_end_time = time.time()
        self.optimization_time += round(opt_end_time - opt_start_time, 3)

        if self.model.getAttr(grb.AttrConstClass.Status) == 2:
            logger.info('The solution is optimal and the objective value '
                        'is {:,.2f}!'.format(self.model.objVal))

    def get_dual_costs(self):
        _dual_costs = defaultdict()
        for c in range(self.number_of_columns):
            _dual_costs[c] = self.membership_in_unique_selection[c].getAttr("Pi")
        return _dual_costs


class CG_Model(OptimizationModel):
    def __init__(self, input_matrix):
        logger.info("This is a CG-formulation")
        super().__init__(input_matrix)
        self.config["CG"] = {"seed": 1,
                             "maxIterations": 1000,
                             "max_tail_length": 100, #  < 1% improvement in last this many iterations to terminate
                              
                             "starting_solution_method": ["shuffle_and_chunk"],
                             # {shuffle_and_chunk, local_optimization}
                             "shuffle_and_chunk_attempts" : 3, 
                             "local_optimization_chunk_size": 16,
                             "local_optimization_number_of_attempts": 3,
                             
                             "sub_problem_type": "blp",
                             # {blp - binary linear problem, greedy, }
                             "greedy_sample_size": 4,
                            
                             }

        self.number_of_iterations = 0
        self.master_time = 0
        self.sub_problem_time = 0
        self.gap_closure_time = 0
        self.objective_value = 0
        self.starting_solution = 0
        self.lower_bound = 0
        self.number_of_solutions = 0

        # Model Specific Things
        self.maxIterations = self.config["CG"]["maxIterations"]
        self.starting_solution_method = self.config["CG"]["starting_solution_method"]
        self.sub_problem_type = self.config["CG"]["sub_problem_type"]
        self.max_tail_length = self.config["CG"]["max_tail_length"]
        
        # Heuristic Parameters >>
        self.shuffle_and_chunk_attempts = self.config["CG"]["shuffle_and_chunk_attempts"]
        self.local_optimization_chunk_size = self.config["CG"]["local_optimization_chunk_size"]
        self.local_optimization_number_of_attempts = self.config["CG"]["local_optimization_number_of_attempts"]
        self.greedy_sample_size = self.config["CG"]["greedy_sample_size"]

        # Internals >>
        self._iteration_counter = 0
        self._solution_counter = 0
        self._memberships = defaultdict(set)
        self._solution_pool = []
        self._terminate_flag = False
        self._master_model = None

    # ********** INITIAL SOLUTION GENERATION *******************
    def shuffle_and_chunk(self, lst, n):
        """Yield successive n-sized chunks from shuffled lst"""
        assert isinstance(lst, list), "arg: lst must be list type"
        assert isinstance(n, int), "arg: n must be a int"
        if self.config["CG"]["seed"]:
            random.seed(self.config["CG"]["seed"])
        for _ in range(self.shuffle_and_chunk_attempts):
            random.shuffle(lst)
            for i in range(0, len(lst), n):
                self._solution_pool.append(sorted(lst[i:i + n])) # in lex order
                self.number_of_solutions += 1

    def local_optimization(self, lst):
        """
        Divide columns into N-chunks
        solve optimal for each chunk
        Add to the solution (This will yield a local minima)
        """
        assert isinstance(lst, list), "arg: lst must be list type"
        # Idea - solve set partitioning within these n-indices
        local_problem_size = self.local_optimization_chunk_size
        number_of_pass = self.local_optimization_number_of_attempts
        if self.config["CG"]["seed"]:
            random.seed(self.config["CG"]["seed"])
        for _ in range(number_of_pass):
            random.shuffle(lst)
            local_problem_instances = []
            for i in range(0, len(lst), local_problem_size):
                local_problem_instances.append(lst[i:i + local_problem_size])
            logger.debug(f"Local Optimization - Created {len(local_problem_instances)} local parts")
            for i, local_problem in enumerate(local_problem_instances):
                logger.debug(f"\t Solving local problem instance {i}")
                # Solve set-packing for each local instance
                # Components: combinations, costs and memberships
                local_model = SetPartitionModel(self.input_matrix, matrix_layout="column_major")
                local_model.subset_columns(local_problem)
                local_model.solve()
                self._solution_pool += local_model.solutions
                self.number_of_solutions += len(local_model.solutions)

    def generate_initial_solution_set_and_costs(self):  # Initial phase of Column Generation
        logger.info("Generating initial solution set")

        if "shuffle_and_chunk" in self.starting_solution_method:
            self.shuffle_and_chunk(list(range(self.number_of_columns)), 4)  # [[3, 4, 5, 7], [0, 1, 2, 6]]

        if "local_optimization" in self.starting_solution_method:
            self.local_optimization(list(range(self.number_of_columns)))

        return
    
    # *******************************************************************
    # *******************************************************************

    def solve_subproblem_blp_model(self, dual_costs):
        """
        It's a BIP of order n-sq (row times columns)
        :param dual_costs:
        :return:
        """
        sub_model = grb.Model("CG_subproblem_BLP")

        # Variables >>
        _sub_var_xi = [sub_model.addVar(vtype=grb.GRB.BINARY, name=f"x_{i}", obj=-1*dual_costs[i])
                       for i in range(self.number_of_columns)]
        _sub_var_zij = [[sub_model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=1, name=f"z_{i}_{j}",
                                          obj=self.input_matrix[i][j])
                         for j in range(self.number_of_rows)]
                        for i in range(self.number_of_columns)]

        # Constraints >>
        # Select exactly 4 columns in a partition >>
        sub_model.addLConstr(
            lhs=grb.quicksum(_sub_var_xi[i] for i in range(self.number_of_columns)),
            sense=grb.GRB.EQUAL,
            rhs=4,
            name="sum_xi"
        )

        # Only select the row deletions from the selected columns in the partitions
        for j in range(self.number_of_rows):
            for i in range(self.number_of_columns):
                sub_model.addLConstr(
                    lhs=_sub_var_zij[i][j],
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=_sub_var_xi[i],
                    name=f"row_deletion_{i}_{j}")

        # Delete exactly two entries from each row
        for j in range(self.number_of_rows):
            sub_model.addLConstr(
                lhs=grb.quicksum([_sub_var_zij[i][j] for i in range(self.number_of_columns)]),
                sense=grb.GRB.EQUAL,
                rhs=2,
                name=f"sum_zi{j}")

        # Solve sub-problem
        # Note: This is already a NP Hard problem, but much smaller in size than the Lin Formulation
        # Disable console logs for sub-problem
        sub_model.setParam('OutputFlag', 0)
        sub_model.setParam('LogToConsole', 0)

        # Might set some parameter (tmlim or MIPgap) for large scale >>
        sub_model.optimize()
        # print(sub_model.objVal)

        # Retrieve the solution
        # The following has to pass because exactly 4 variables would take value = 1
        # So, this is also an implicit solution validation
        [i, j, k, l] = [_i for _i in range(self.number_of_columns) if _sub_var_xi[_i].x >= 0.5]
        return sub_model.objVal, [i, j, k, l]

    def solve_subproblem_greedy_heuristic(self, dual_costs):
        """
        It's a heuristic to solve BIP of order n-sq (row times columns)
        :param dual_costs:
        :return:
        """
        logger.debug("Subproblem greedy heuristic")
        sample_size = self.greedy_sample_size
        # Find columns with most expensive shadow price
        # Make their combinations and find the reduced costs
        # the least reduced costs >>
        column_sample = sorted(list(range(self.number_of_columns)),
                               key=lambda x: dual_costs[x], reverse=True)[:sample_size]
        min_cost_combination = []
        min_cost = grb.GRB.INFINITY
        column_sample = sorted(column_sample)
        for comb in combinations(column_sample, 4):
            key, cost = self.get_partition_cost_cuda(*comb)
            reduced_cost = cost - sum([dual_costs[c] for c in comb])
            if reduced_cost < min_cost:
                min_cost_combination = comb
                min_cost = reduced_cost
        return min_cost, list(min_cost_combination)
    
    def solve_subproblem_simulated_annealing(self, dual_costs):
        """
        Again a subproblem heuristic, solving BIP using simulated annealing
        """
        logger.info("Subproblem simulated annealing")
        # Starting solution 
        column_sample = sorted(list(range(self.number_of_columns)),
                               key=lambda x: dual_costs[x], reverse=True)
        pass
        

    def solve_subproblem(self, dual_costs):
        number_of_solves = int(self.number_of_columns/4) # ccg or just one solution
        all_solutions = []  # Subproblem can generate either 1 or multiple partitions

        def update_duals(s):
            for c in s:
                dual_costs[c] = -1 * grb.GRB.INFINITY
        
        method = {
            "blp": self.solve_subproblem_blp_model,
            "greedy": self.solve_subproblem_greedy_heuristic
        }[self.sub_problem_type]
        
        for i in range(number_of_solves):
            r, solution = method(dual_costs)
            if i == 0 and r > 0:
                self._terminate_flag = True
                logger.info(f"subproblem terminated! with r[0] = {r}\n No new column can enter the basis!")
                # returns empty list of all_solutions
                break
            else:
                update_duals(solution)
                all_solutions.append(solution)

        return all_solutions

    # # # ================== Costs and objective function ==================
    def solve(self):
        """
        A generic solution method for mathematical programming subclasses
        :return:
        """

        if self.input_matrix is None:
            logger.error("Generate a problem instance before solving model!")
            logger.error("Use the available methods : self.create_input_matrix(), self.show_input_matrix()")
            raise NotInitializedException

        progress_tracker = []
        logger.info("Starting Column Generation")
        global_start_time = time.time()
        self.generate_initial_solution_set_and_costs()
        # self._solution_pool_is_populated now
        # the ignition for the master engine
        self._master_model = MasterProblem(self.input_matrix)
        self._master_model.solve(restricted_column_set=self._solution_pool)
        self.starting_solution = self._master_model.model.objVal
        _ = 0
        
        for _ in range(self.maxIterations):
            logger.debug(f"SOLVING MASTER-PROBLEM IN ITERATION {_ + 1}")

            if time.time() - global_start_time > self.config["time_limit"]:
                self.model_timeout = True
                logger.error("Column Generation - TIME_OUT_ERROR")
                break

            dual_costs = self._master_model.get_dual_costs()

            # Get's a warm start after first iteration
            logger.debug(f"\t\t SOLVING SUB-PROBLEM IN ITERATION {_ + 1}")

            subproblem_start = time.time()
            # Subproblem may themselves update the terminate flag if stopping criteria achieved
            new_solutions = self.solve_subproblem(dual_costs)
            subproblem_end = time.time()
            self.sub_problem_time += subproblem_end - subproblem_start

            n = 0
            for s in new_solutions:
                if self._master_model.extend_model_variable(s):
                    self._solution_pool.append(s)
                    self.number_of_solutions += 1
                    n += 1
                    
            if n == 0: 
                self._terminate_flag = True
                logger.info("No new solutions were discovered in this iteration!  - !! Terminating CG !!")
            
            if self._terminate_flag:
                # this could happen due to other reasons also (like some heuristic set this as true)
                break

            self._master_model.re_solve()

            if len(progress_tracker) < self.max_tail_length:
                progress_tracker.append(self._master_model.model.objVal)
            else:
                progress_tracker.pop(0)
                progress_tracker.append(self._master_model.model.objVal)
                if (progress_tracker[0] - progress_tracker[-1]) / progress_tracker[-1] < 0.0001:
                    # Improvement in last few iterations < 1 %
                    logger.info(f"No or (< 1%) improvement in last {self.max_tail_length} iterations")
                    break

        # Solve set-pack to determine integer output >>
        logger.info(f"CG Completed in {_ + 1} iterations")
        self.number_of_iterations = _ + 1
        self.master_time = self._master_model.optimization_time
        logger.info("Solve set-packing with current solution sample >>")
        
        logger.debug("Updating Variable Type")

        self.lower_bound = self._master_model.model.objVal

        gap_start_time = time.time()
        final = SetPartitionModel(self.input_matrix, matrix_layout="column_major")
        final.solve(restricted_column_set=self._solution_pool)
        self.objective_value = final.model.objVal
        self.solutions = final.solutions
        gap_end_time = time.time()
        self.gap_closure_time += gap_end_time - gap_start_time
        self.optimization_time = self.master_time + self.sub_problem_time + self.gap_closure_time
        self.model_solved = True
        return

def call_ColumnGeneration(input_matrix):
    model = CG_Model(input_matrix)
    model.solve()
    return model.get_apex_solution()

if __name__ == "__main__":
    logger.info("Testing Column Generation Model Formulation")
    matrix = CG_Model.create_input_matrix(32, 32)
    opt = CG_Model(matrix)
    opt.show_input_matrix()
    opt.solve()
    # opt.pprint_output()

