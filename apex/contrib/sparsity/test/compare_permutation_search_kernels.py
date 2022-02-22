# mohit-mhjn (mohitm3@illinois.edu) - experimenting with 2:4 sparsity
from torchvision.models import resnet50
import apex.contrib.sparsity.permutation_search_kernels as psk

if __name__=="__main__":
    pretrained = resnet50(pretrained=True)
    my_matrix_group = pretrained.layer3[0].conv3.weight  # Change this for different matrix dimensions
    psk.accelerated_search_for_good_permutation(my_matrix_group, options={
        "strategy": "progressive channel swap - SA",
        # "strategy": "progressive channel swap",
        "progressive_search_time_limit": 3600,  # Relax this limit for experiment
        "SA_initial_t": 1,
        "SA_room_t": 10e-3,
        "SA_tfactor": 0.90,
        "SA_epochs": 100})