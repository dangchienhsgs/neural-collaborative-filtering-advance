import numpy as np

from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import GMFlogistic

model_file = "pretrain/ml-1m_GMF_10_neg_1_hr_0.6470_ndcg_0.3714.h5"
dataset_name = "ml-1m"
mf_dim = 64
layers = [512, 256, 128, 64]

reg_layers = [0, 0, 0, 0]
reg_mf = 0

num_factors = 10
regs = [0, 0]

# Loading data
t1 = time()
dataset = Dataset("Data/" + dataset_name)
train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
num_users, num_items = train.shape
print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
      % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

# Get model
model = GMFlogistic.get_model(num_items=num_items, num_users=num_users, latent_dim=num_factors, regs=regs)
model.load_weights(model_file)

# Evaluate performance
print("K\tHR\tNDCG")
for topK in range(1, 10):
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, 1)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print("%d\t%.4f\t%.4f" % (topK, hr, ndcg))
