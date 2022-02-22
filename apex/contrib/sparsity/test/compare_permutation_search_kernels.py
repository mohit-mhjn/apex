# mohit-mhjn (mohitm3@illinois.edu) - experimenting with 2:4 sparsity
from torchvision.models import resnet50
import apex.contrib.sparsity.permutation_search_kernels as psk

if __name__ == "__main__":
    pretrained = resnet50(pretrained=True)
    mode = "debug"  # {'debug', 'compare'}
    strategy_set = ["exhaustive", "progressive channel swap", "progressive channel swap - SA"]
    retrial_per_scenario = 5
    test_matrices = [
        pretrained.layer1[0].conv1.weight,  # 64 x 64
        pretrained.layer1[1].conv1.weight,  # 64 x 256
        pretrained.layer2[1].conv1.weight,  # 128 x 512
        pretrained.layer3[1].conv1.weight,  # 256 x 1024
        pretrained.layer4[1].conv1.weight,  # 512 x 2048
        ]

    if mode == "debug":
        my_matrix_group = test_matrices[0]
        psk.accelerated_search_for_good_permutation(my_matrix_group, options={
            "strategy": "progressive channel swap - SA",
            # "strategy": "progressive channel swap",
            "progressive_search_time_limit": 999,  # Relax this limit for experiment
            "SA_initial_t": 1,
            "SA_room_t": 10e-3,
            "SA_tfactor": 0.90,
            "SA_epochs": 100})

    elif mode == "compare":
        for strategy in strategy_set:
            for my_matrix_group in test_matrices:
                for _ in range(retrial_per_scenario):
                    print("[Compare permutation search kernels]: Attempt {} for matrix "
                          "shape ({},{}) using {} strategy".format(_, my_matrix_group.shape[0],
                                                                    my_matrix_group.shape[1],
                                                                    strategy))
                    psk.accelerated_search_for_good_permutation(my_matrix_group, options={
                        "strategy": strategy,
                        # "strategy": "progressive channel swap",
                        "progressive_search_time_limit": 999,  # Relax this limit for experiment
                        "SA_initial_t": 1,
                        "SA_room_t": 10e-3,
                        "SA_tfactor": 0.90,
                        "SA_epochs": 100})
    else:
        print("[Compare permutation search kernels]: Invalid mode selection!")
        exit(-1)