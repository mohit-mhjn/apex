import numpy as np
from .permutation_utilities import *
from .exhaustive_search import Exhaustive_Search
from .exact_methods import *
from .column_generation import *
from .simulated_annealing import *


def accelerated_search_for_good_permutation(matrix_group, options=None):
    """This function is used to call the permutation search CUDA kernels.
    users can provide prefer search strategy by providing a valid 'options' as a dictionary,
    or users can implement their customized 'accelerated_search_for_good_permutation' function.
    """
    input_matrix = matrix_group.cpu().detach().numpy()

    # mohit: Temporary patch - need deep knowledge of matrix group here
    # If input matrix is 2D - following operation wouldn't change anything
    input_matrix = input_matrix.reshape(input_matrix.shape[0], input_matrix.shape[1])
    input_matrix = np.abs(input_matrix)

    print("\n[accelerated_search_for_good_permutation] input matrix shape: \'{:}\'.".format(input_matrix.shape))

    result = np.copy(input_matrix)
    # init a sequential permutation search sequence
    input_channel_num = matrix_group.size()[1]
    permutation_sequence = [n for n in range(input_channel_num)]
    duration = 0.0

    if options == None:
        options = {}
    if 'strategy' not in options:  # right now, the default permutation search strategy is: 'exhaustive' search
        options['strategy'] = 'exhaustive'
    print("[accelerated_search_for_good_permutation] the permutation strategy is: \'{:} search\'.".format(
        options['strategy']))

    # define sub options for each search strategy
    if options['strategy'] == 'exhaustive':
        # right now, the default options for 'exhaustive' search is: 'exhaustive,8,100'
        if 'stripe_group_size' not in options:
            options['stripe_group_size'] = 8
        if 'escape_attempts' not in options:
            options['escape_attempts'] = 100
    elif options['strategy'] in ['progressive channel swap',
                                 'progressive channel swap - SA']:  # copy param defaults of progressive search to progressive search - SA
        # just swaps meaningful channels, keeping the good swaps, until the search time limit expires.
        if 'progressive_search_time_limit' not in options:
            options['progressive_search_time_limit'] = 60
        if 'improvement_threshold' not in options:
            options['improvement_threshold'] = 1e-9

    # execute the requested strategy
    if options['strategy'] == 'exhaustive':
        result, duration, permutation_sequence = Exhaustive_Search(result,
                                                                   stripe_group_size=options['stripe_group_size'],
                                                                   escape_attempts=options['escape_attempts'])
    elif options['strategy'] == 'progressive channel swap':
        real_swap_num = 0
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < options['progressive_search_time_limit']:
            src = np.random.randint(result.shape[1])
            dst = np.random.randint(result.shape[1])
            src_group = int(src / 4)
            dst_group = int(dst / 4)
            if src_group == dst_group:  # channel swapping within a stripe does nothing
                continue
            new_sum, improvement = try_swap(result, dst, src)
            if improvement > options['improvement_threshold']:
                result[..., [src, dst]] = result[..., [dst, src]]
                permutation_sequence[src], permutation_sequence[dst] = permutation_sequence[dst], permutation_sequence[
                    src]
                real_swap_num += 1
        duration = time.perf_counter() - start_time
        print("\tFinally swap {} channel pairs until the search time limit expires.".format(real_swap_num))

    elif options['strategy'] == 'SIMULATED_ANNEALING':

        # Get Essential Parameters for simulated annealing, defaults returned
        SA_initial_t = options.get("SA_initial_t", 1000)  # Starting temperature (boiling point)
        SA_room_t = options.get("SA_room_t", 10e-3)  # Steady state temperature
        SA_tfactor = options.get("SA_tfactor", 0.95)  # Temperature falls by this factor
        SA_epochs = options.get("SA_epochs", 100)  # Temperature steps

        result, duration, permutation_sequence = Simulated_Annealing(result, initial_temp=SA_initial_t,
                                                                     room_temp=SA_room_t,
                                                                     tfactor=SA_tfactor,
                                                                     epochs=SA_epochs,
                                                                     threshold=options['improvement_threshold'],
                                                                     timelimit=options['progressive_search_time_limit'])

    elif options["strategy"] in ["BQP", "BLP", "MDA", "SETPART"]:
        model_collection = {"BQP": BqpModel,
                            "BLP": BlpModel,
                            "MDA": MdaModel,
                            "SETPART": SetPartitionModel
                            }
        model = model_collection[options["strategy"]](input_matrix)
        model.solve()
        result, duration, permutation_sequence = model.get_apex_solution()

    elif options["strategy"] in ["CG"]:
        model = CG_Model(input_matrix)
        model.solve()
        result, duration, permutation_sequence = model.get_apex_solution()

    elif options['strategy'] == 'user defined':
        # need to get the permutated matrix (result) by applying
        # customized permutation search function
        print("[accelerated_search_for_good_permutation] Use the user customized permutation search function!")
    else:
        print("[accelerated_search_for_good_permutation] Cannot find the implementation of the required strategy!")
    print("[accelerated_search_for_good_permutation] Take {:.4f} seconds to search the permutation sequence.".format(
        duration))

    # In the new version of Exhaustive_Search function, there’s no need to use the find_permutation(result, input_matrix) function
    # to recover the permutation sequence applied to the input_matrix to get the result separately any more.
    # start_time_find_permutation = time.perf_counter()
    # permutation_sequence = find_permutation(result, input_matrix)
    # duration_find_permutation = time.perf_counter() - start_time_find_permutation
    # print("[accelerated_search_for_good_permutation] Take {:.4f} seconds to finish find_permutation function.".format(duration_find_permutation))
    # print("[accelerated_search_for_good_permutation] The permutation sequence is: {:}".format(permutation_sequence))
    # print("[accelerated_search_for_good_permutation] The length of permutation sequence is: {:}".format(len(permutation_sequence)))
    magnitude = sum_after_2_to_4(result)
    deleted = np.sum(result) - magnitude
    print("[accelerated_search_for_good_permutation] The total magnitude using the {} strategy is: {}".format(
        options["strategy"], magnitude))
    print(
        "[accelerated_search_for_good_permutation] The total sum of entries deleted using the {} strategy is: {}".format(
            options["strategy"], deleted))
    return permutation_sequence
