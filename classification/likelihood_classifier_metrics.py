import data_loader
from archives.ensemble_classifier import EnsembleClassifier


def accuracy_likelihood_classification():
    train_healthy, train_sick, test_healthy, test_sick = data_loader.load_merged_data()
    # Draw histogram for each gene for train
    for i in range(train_healthy.shape[1]):
        import matplotlib.pyplot as plt
        # plt.hist(train_healthy[:, i], bins=range(int(min(train_healthy[:, i])), int(max(train_healthy[:, i])) + 100, 100), alpha=0.5, label='healthy', density=True)
        # plt.hist(train_sick[:, i], bins=range(int(min(train_sick[:, i])), int(max(train_sick[:, i])) + 100, 100), alpha=0.5, label='sick', density=True)
        plt.hist(train_healthy[:, i], bins=100, alpha=0.5, label='healthy', density=True)
        plt.hist(train_sick[:, i], bins=100, alpha=0.5, label='sick', density=True)
        plt.show()

    classifier = EnsembleClassifier(train_healthy, train_sick)
    healthy_preds = classifier.classify(test_healthy)
    sick_preds = classifier.classify(test_sick)
    print(f"Healthy accuracy: {sum(healthy_preds) / len(healthy_preds)}")
    print(f"Sick accuracy: {1 - sum(sick_preds) / len(sick_preds)}")
    # healthy_scores = [p[0] - p[1] for p in healthy_preds]
    # print(healthy_scores)
    # print(np.mean(healthy_scores))
    # sick_scores = [p[0] - p[1] for p in sick_preds]
    # print(sick_scores)
    # print(np.mean(sick_scores))

    # Histogram both
    # import matplotlib.pyplot as plt
    # plt.hist(healthy_scores, bins=range(int(min(healthy_scores)), int(max(healthy_scores)) + 100, 100), alpha=0.5, label='healthy', density=True)
    # plt.hist(sick_scores, bins=range(int(min(sick_scores)), int(max(sick_scores)) + 100, 100), alpha=0.5, label='sick', density=True)
    # plt.show()

    # alpha = 2500
    #
    # healthy_preds = np.asarray([1 if i < alpha else 0 for i in healthy_scores])
    # sick_preds = np.asarray([1 if i < alpha else 0 for i in sick_scores])
    #
    # print(f"Healthy accuracy: {sum(healthy_preds == 0) / len(healthy_preds)}")
    # print(f"Sick accuracy: {sum(sick_preds == 1) / len(sick_preds)}")
    # accuracy = sum(healthy_preds == 0) + sum(sick_preds == 1)
    # print(accuracy / (len(healthy_preds) + len(sick_preds)))
    # print(classifier.classify([classifier.likelihood_estimators[0].means]))
    # print(classifier.classify([classifier.likelihood_estimators[1].means]))


if __name__ == '__main__':
    accuracy_likelihood_classification()
