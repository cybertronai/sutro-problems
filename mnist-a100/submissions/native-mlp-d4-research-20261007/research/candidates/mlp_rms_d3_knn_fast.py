"""Research candidate: E8 width256 RMS MLP, 1250 updates, native K5 readout.
Requires repository CUDA sources; standalone evaluator qualification pending.
"""
import torch
import persistent_bf16_deep3_rmsnorm_input32 as model
import learned_ridge_head as readout
CONFIG = dict(width=256, members=8, batch=1024, views=2, steps=1250,
              lr1=.4, lr2=.4, head_lr=.4, momentum=.95, dropout=.1,
              weight_decay=.001, mixup=.2, input_noise=.15, input_scale=2.,
              initialization='data_sample', dictionary_scale=0., data_noise=.25,
              swa_fraction=.25, schedule='cosine', smoothing=.05)
def classify(train_x, train_y, test_x):
    queries = torch.cat((train_x, test_x))
    model.classify(train_x, train_y, queries, **CONFIG)
    state = model.extension().debug_last_state()
    features = state[30].reshape(8, -1, 256)[:, :len(queries)].contiguous().reshape(-1, 256)
    return readout.extension().knn(features, train_y, 8, 5, 100.).argmax(1)
