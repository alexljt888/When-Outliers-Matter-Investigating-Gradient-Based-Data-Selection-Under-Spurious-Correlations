from spuco.datasets import SpuCoMNIST, SpuriousFeatureDifficulty
from spuco.utils import set_seed

set_seed(42)

dataset = SpuCoMNIST(
    root="./data",
    spurious_feature_difficulty=SpuriousFeatureDifficulty.MAGNITUDE_LARGE,
    classes=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    spurious_correlation_strength=0.95,
    split="train",
    download=True,
)

dataset.initialize()

print("Dataset downloaded successfully!")
print("Number of training examples:", len(dataset))
print("Group sizes:")
for group, indices in dataset.group_partition.items():
    print(group, len(indices))
