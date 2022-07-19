# Aggregated Multi-output Gaussian Processes | 2022
# Yusuke Tanaka

import pdb
import os
import time
import math
import json
import argparse
import torch
import matplotlib.pyplot as plt

from utils import *

dpi = 100
DTYPE = torch.float64
rootpi = math.sqrt(math.pi)
th = 1e-6
eps = 1e-8


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--cuda_no', default='0', type=str)
    parser.add_argument('--latent_process', default=1, type=int, help='number of latent GPs')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--max_epoch', default=50000, type=int, help='maximum iteration steps')
    parser.add_argument('--print_every', default=1000, type=int, help='number of iterations for prints')
    parser.add_argument('--Exp', default='Exp', type=str, help='Directory for experiment')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    return parser.parse_args()

def init_params():
    Beta = torch.tensor([2e-1]*L, dtype=DTYPE).requires_grad_(True)
    Sigma = torch.tensor([1e-2]*S, dtype=DTYPE).requires_grad_(True)
    W = torch.empty(S,L).uniform_(1e-4, 2e-4).type(DTYPE).requires_grad_(True)
    return Beta, Sigma, W

def calcu_loss():
    C = calcu_C()
    loss = (0.5 * torch.mv(torch.linalg.solve(C, y.reshape([N,1])).t(), y)
            + 0.5 * torch.logdet(C))
    return loss

def calcu_C():
    def base(id):
        b = 0
        for i in range(id):
            b += len(p2r[d2p[datasets[i]]])
        return b

    C = torch.zeros([CN,CN], dtype=DTYPE)
    for id1, id2 in list(itertools.combinations_with_replacement(range(S), 2)):
        p1 = d2p[datasets[id1]]
        p2 = d2p[datasets[id2]]
        R1 = len(p2r[p1])
        R2 = len(p2r[p2])

        dists = pp2dists[(p1,p2)]
        Z1 = dists[:,0] / Beta.reshape([L,1]); Z2 = dists[:,1] / Beta.reshape([L,1])
        Z3 = dists[:,2] / Beta.reshape([L,1]); Z4 = dists[:,3] / Beta.reshape([L,1])
        V1 = dists[:,4]; V2 = dists[:,5]

        G1 = (Z1*rootpi*torch.erf(Z1)+torch.exp(-Z1**2))*Beta.reshape([L,1])**2/2
        G2 = (Z2*rootpi*torch.erf(Z2)+torch.exp(-Z2**2))*Beta.reshape([L,1])**2/2
        G3 = (Z3*rootpi*torch.erf(Z3)+torch.exp(-Z3**2))*Beta.reshape([L,1])**2/2
        G4 = (Z4*rootpi*torch.erf(Z4)+torch.exp(-Z4**2))*Beta.reshape([L,1])**2/2
        weight = (W[id1] * W[id2]).reshape([1,L])
        c = torch.mm(weight, (G1+G2-G3-G4)/(V1*V2)).reshape([R1,R2])
        b1 = base(id1)
        b2 = base(id2)
        if id1 == id2:
            C[b1:b1+R1, b2:b2+R2] = c + Sigma[id1]**2 * torch.eye(R1, dtype=DTYPE)
        else:
            C[b1:b1+R1, b2:b2+R2] = c
            C[b2:b2+R2, b1:b1+R1] = c.t()
    C += eps
    C = C[:,ava_ids]
    C = C[ava_ids,:]
    return C

def torch_opt():
    optim = torch.optim.Adam([Beta, Sigma, W], lr=args.learn_rate)
    losses = []
    t0 = time.time()
    for step in range(args.max_epoch):
        loss = calcu_loss()
        loss.backward(); optim.step(); optim.zero_grad()
        losses.append(loss.item())
        
        if step % args.print_every == 0:
            print("step {}, time {:.2e}, loss {:.4e}".format(step, time.time()-t0, loss.item()))
            t0 = time.time()

        if step > 0 and abs( (losses[len(losses)-1] - losses[len(losses)-2])
                             / losses[len(losses)-2] ) < th:
            break
    return losses

def prediction(target_id):
    C = calcu_C()
    H = torch.zeros([grids.shape[0],CN], dtype=DTYPE)
    counter = 0
    for id in range(S):
        weight = (W[target_id] * W[id]).reshape([1,L])
        p = d2p[datasets[id]]
        for r in p2r[p]:
            dists = pr2dists[(p,r)]
            Z1 = dists[:,0] / Beta.reshape([L,1])
            Z2 = dists[:,1] / Beta.reshape([L,1])
            V = dists[:,2]
            G = (torch.erf(Z1) + torch.erf(Z2)) * rootpi * Beta.reshape([L,1]) / 2
            h = torch.mm(weight, G/V).t()
            H[:,counter] = h.squeeze()
            counter += 1
    H = H[:,ava_ids]
    HC = torch.linalg.solve(C,H.T)
    return torch.mm(H, torch.linalg.solve(C, y.reshape([N,1])))

def aggregation(p, pred):
    agg_pred = []
    for r in p2r[p]:
        left, right = r[0], r[1]
        ids = np.where( (grids >= left) & (grids <= right))[0]
        agg_pred.append([r[0],r[1],torch.mean(pred[ids]).item()])
    return np.array(agg_pred)

def output_params():
    csv_write(save_dir + '/Beta.csv', Beta.numpy())
    csv_write(save_dir + '/Sigma.csv', Sigma.numpy())
    names = np.array(datasets).reshape(len(datasets),1)
    csv_write(save_dir + '/W.csv', np.hstack([names, W.numpy()]))

def visualization():
    fig = plt.figure(figsize=(24, 12), facecolor='white', dpi=dpi)
    fig.suptitle(target_city)
    ylabels = datasets
    for i in range(S):
        dataset, partition = filenames[i]
        i_dir = 'data/' + target_city
        i_file = glob.glob(i_dir + '/' + partition + '/' + dataset + '/*.csv')[0]
        Dat = csv_read(i_file)
        Dat = np.array(Dat)
        Dat[np.where(Dat == 'NA')] = 0

        #X
        s_times = np.double(Dat[:,1])
        e_times = np.double(Dat[:,2])
        ids = [i+1 for i in range(len(s_times)-1)]
        s_times[ids] += 1e-5
        s_times = s_times.reshape([len(s_times),1])
        e_times = e_times.reshape([len(e_times),1])
        tmp = np.hstack([s_times,e_times])
        X = tmp.reshape(tmp.shape[0]*tmp.shape[1])

        #Y
        Y = np.double(Dat[:,3])
        Y, mean, std = normalization(torch.tensor(Y))
        Y = Y.numpy()
        Y = Y.reshape(Y.shape[0],1)
        Y = np.hstack([Y,Y])
        Y = Y.reshape(Y.shape[0]*Y.shape[1])

        #vis
        fig.add_subplot(S, 1, i+1, frameon=True)
        plt.plot(X, Y, color='steelblue', alpha=0.8, lw=0.8)
        plt.plot(grids, preds[:,i], color='crimson', alpha=0.8, lw=0.8)
        plt.ylabel(ylabels[i])
        plt.xlim([0,1])
        plt.tight_layout()
    plt.savefig(save_dir + '/pred.pdf', format='pdf')
    plt.close()


if __name__ == "__main__":

    # setting
    args = get_args()
    torch.manual_seed(args.seed)
    input_file = args.Exp + '/input.csv'
    save_dir = args.Exp + '/result/A-MoGP/' + str(args.latent_process)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None

    # input
    target_city, datasets, S, partitions, P, filenames, d2p = read_input_file(input_file)
    ## training data
    data_dir = 'data/' + target_city
    y, N, CN, means, stds, p2r, ava_ids, target_ava_ids = input_data(data_dir, filenames)
    ## test data
    target_partition = 'week'
    target_data = datasets[0]
    filename = 'data/' + target_city + '/' + target_partition + '/' + target_data + '/*.csv'
    test_true, grids, test_ava_ids = input_target_data(filename, target_partition, p2r)
    pp2dists, pr2dists = input_dists(partitions, p2r, grids)
    
    # inference
    L = args.latent_process
    eps = torch.eye(CN, dtype=DTYPE) * eps
    Beta, Sigma, W = init_params()
    losses = torch_opt()
    Beta = Beta.detach(); Sigma = Sigma.detach(); W = W.detach()
    output_params()

    # prediction (grids)
    target_id = [i for i in range(S) if target_data == datasets[i]][0]
    pred = prediction(target_id)
    pred = inv_normalization(pred, means[target_data], stds[target_data])

    # MAPE (train)
    p = d2p[target_data]
    agg_pred_train = aggregation(p, pred)
    agg_pred_train = agg_pred_train[target_ava_ids]
    true = inv_normalization(y[0:len(p2r[p])].clone(), means[target_data], stds[target_data]).numpy()
    est = agg_pred_train[:,2]
    train_err = MAPE(true, est)

    # MAPE (test)
    p = target_partition
    agg_pred_test = aggregation(p, pred)
    agg_pred_test = agg_pred_test[test_ava_ids]
    est = agg_pred_test[:,2]
    test_err = MAPE(test_true, est)

    # save
    path = save_dir + '/result.csv'
    csv_write(path, np.array(['train_err',train_err,'test_err',test_err]))
    print("train_err {:.2e}, test_err {:.2e}".format(train_err, test_err))
    
    path = save_dir + '/setting.json'
    with open(path, 'w') as f:
        json.dump(vars(args), f)

    # visualization
    preds = torch.zeros([grids.shape[0],len(datasets)], dtype=DTYPE)
    for target_id in range(S):
        pred = prediction(target_id)
        preds[:,target_id] = pred.squeeze()
    visualization()
