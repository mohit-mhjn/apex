# mohit-mhjn (mohitm3@illinois.edu) - experimenting with 2:4 sparsity
from torchvision.models import resnet50
import apex.contrib.sparsity.permutation_search_kernels as psk


class EmulateTorchPretrained(object):
    """
    This will emulate torch pretrained to generate fake calls for
    pre_trained_matrix.cpu().detach().numpy()
    pre_trained_matrix.shape
    """

    def __init__(self, matrix):
        self.matrix = matrix

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self.matrix

    @property
    def shape(self):
        return self.matrix.shape


if __name__ == "__main__":
    # All scenario settings configured here >>
    matrix_set = "custom"  # {'custom', 'pretrained'}
    strategy_set = ["exhaustive", "progressive channel swap",  # Current standard
                    "BQP", "BLP", "MDA", "SETPART",  # proposed exact methods
                    "progressive channel swap - SA",  # "CCG"   # proposed heuristics
                    ]
    mode = "debug"  # {'debug', 'compare'}

    if matrix_set == "pretrained":
        pretrained = resnet50(pretrained=True)
        retrial_per_scenario = 5
        test_matrices = [
            pretrained.layer1[0].conv1.weight,  # 64 x 64
            pretrained.layer1[1].conv1.weight,  # 64 x 256
            pretrained.layer2[1].conv1.weight,  # 128 x 512
            pretrained.layer3[1].conv1.weight,  # 256 x 1024
            pretrained.layer4[1].conv1.weight,  # 512 x 2048
        ]

    elif matrix_set == "custom":
        retrial_per_scenario = 1
        matrix_dimensions = [4, 8, 12, 16, 20, 24, 32, 48, 64, 72]
        test_matrices = [
            EmulateTorchPretrained(psk.OptimizationModel.create_input_matrix(n, n))
            for n in matrix_dimensions
        ]
    else:
        raise Exception("Invalid matrix set")

    if mode == "debug":
        test_matrices = [test_matrices[1]]  # slice the test matrices list to find the right set to debug
        strategy_set = [strategy_set[2]]  # select strategy index to debug

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
