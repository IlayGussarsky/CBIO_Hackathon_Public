from data_loader import load_np_data
import numpy as np
import config
healthy, sick, feature_names = load_np_data()

# Load genes from feature_selection/gene_orders
gene_order_gmm = np.loadtxt(rf'{config.GENE_ORDERS_PATH}/gene_order_gmm.csv', dtype=str)
gene_order_kstest = np.loadtxt(rf'{config.GENE_ORDERS_PATH}/gene_order_kstest.csv', dtype=str)
gene_order_ttest = np.loadtxt(rf'{config.GENE_ORDERS_PATH}/gene_order_ttest.csv', dtype=str)
gene_order_fnn = np.loadtxt(rf'{config.GENE_ORDERS_PATH}/gene_order_fnn.csv', dtype=str)