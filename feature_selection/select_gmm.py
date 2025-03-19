from statistics import NormalDist

import numpy as np

import data_loader


def select_gmm(k=None):
    # Load all data
    healthy, sick, feature_names = data_loader.load_np_data()

    overlaps = []
    skipped = []
    for i, feature in enumerate(feature_names):
        # print(i, feature)
        # If over half of the signal for this gene is less than 8, skip it
        cur_gene_healthy_zeros = np.mean(healthy[:, i] <= 8)
        cur_gene_sick_zeros = np.mean(sick[:, i] <= 8)

        if cur_gene_sick_zeros > 0.5 or cur_gene_healthy_zeros > 0.5:
            skipped.append((i, feature, 0))
            continue

        # Create the gaussian distributions
        healthy_expected, healthy_std = np.mean(healthy[:, i]), np.std(healthy[:, i])
        sick_expected, sick_std = np.mean(sick[:, i]), np.std(sick[:, i])
        if healthy_std == 0 or sick_std == 0:
            skipped.append((i, feature, 0))
            continue

        # Remove outliers
        cur_gene_healthy = healthy[:, i][np.abs(healthy[:, i] - healthy_expected) < 2 * healthy_std]
        cur_gene_sick = sick[:, i][np.abs(sick[:, i] - sick_expected) < 2 * sick_std]

        # Recreate gaussian dist., then overlap
        healthy_expected, healthy_std = np.mean(cur_gene_healthy), np.std(cur_gene_healthy)
        sick_expected, sick_std = np.mean(cur_gene_sick), np.std(cur_gene_sick)

        overlap = NormalDist(mu=healthy_expected, sigma=healthy_std).overlap(NormalDist(mu=sick_expected, sigma=sick_std))
        overlaps.append((i, feature, overlap))

    overlaps.sort(key=lambda x: x[2])
    overlaps.extend(skipped)
    overlaps = [o[1] for o in overlaps]
    if k is not None:
        return overlaps[:k]
    return overlaps


if __name__ == '__main__':
    print(len(select_gmm()))
    for i in select_gmm():
        print(i)
