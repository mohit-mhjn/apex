from .permutation_utilities import *


def Simulated_Annealing(result, initial_temp = 1000, room_temp = 0.001,
                                tfactor = 0.99, epochs = 100,
                                threshold = 1e-9, timelimit=10e6):
    """
    Execute Simulated Annealing Heurtisitc on 2x4 sparsity permutation search
    Args:
        result:
        permutation_sequence:
        SA_initial_t:
        SA_room_t:
        SA_tfactor:
        SA_epochs:
        improvement_threshold:

    Returns:
    """
    start_time = time.perf_counter()
    permutation_sequence = [n for n in range(result.shape[1])]
    real_swap_num = 0
    start_time = time.perf_counter()
    temperature = initial_temp
    best_permutation_seq = [n for n in permutation_sequence]
    best_result = result.copy()
    best_magnitude = sum_after_2_to_4(best_result)
    while temperature > room_temp and time.perf_counter() - start_time < timelimit:
        for e in range(epochs):
            src = np.random.randint(result.shape[1])
            dst = np.random.randint(result.shape[1])
            src_group = int(src / 4)
            dst_group = int(dst / 4)
            if src_group == dst_group:  # channel swapping within a stripe does nothing
                continue
            new_sum, improvement = try_swap(result, dst, src)
            # mohit: Always accept if that's a good swap!
            if improvement > threshold:
                result[..., [src, dst]] = result[..., [dst, src]]
                permutation_sequence[src], permutation_sequence[dst] = permutation_sequence[dst], \
                                                                       permutation_sequence[src]
                real_swap_num += 1

                # The good swap could be better the best obtained so far, if so update
                new_magnitude = sum_after_2_to_4(result)
                if new_magnitude > best_magnitude:
                    best_permutation_seq = [n for n in permutation_sequence]
                    best_result = result.copy()
                    best_magnitude = new_magnitude

            # mohit: accept the worse swap with some probability (determined through SA progress)
            elif np.exp(improvement / temperature) > np.random.uniform(0, 1):
                result[..., [src, dst]] = result[..., [dst, src]]
                permutation_sequence[src], permutation_sequence[dst] = permutation_sequence[dst], \
                                                                       permutation_sequence[src]
                real_swap_num += 1
            else:
                continue
            if time.perf_counter() - start_time > timelimit:
                break
        temperature = temperature * tfactor
    duration = time.perf_counter() - start_time
    print("\tFinally swap {} channel pairs until the search termination criteria".format(real_swap_num))
    return best_result, duration, best_permutation_seq
