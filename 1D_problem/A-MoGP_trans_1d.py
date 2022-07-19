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

def preparation():
    c2Epsilon = defaultdict(str)
    all_datasets = []
    for city in cities:
        c2Epsilon[city] = (torch.eye(c2CN[city], dtype=DTYPE) * eps)
        all_datasets.extend(c2datasets[city])
    total_S = len(all_datasets)
    uniq_datasets = list(set(all_datasets))
    uniq_S = len(uniq_datasets)

    c2ids_for_W = defaultdict(str)
    for city in cities:
        c2ids_for_W[city] = [uniq_datasets.index(c2datasets[city][i]) for i in range(c2S[city])]
    return c2Epsilon, total_S, uniq_S, uniq_datasets, c2ids_for_W

def init_params():
    Beta = torch.tensor([2e-1]*L, dtype=DTYPE).requires_grad_(True)
    c2eta_p = torch.tensor([[1e-1]*L for i in range(uniq_S)]).requires_grad_(True)
    c2W_p = torch.empty(uniq_S,L).uniform_(1e-4,2e-4).type(DTYPE).requires_grad_(True)
    c2Sigma = torch.tensor([], dtype=DTYPE)
    c2eta_v = torch.tensor([], dtype=DTYPE)#valiational dist. variance
    c2W_v = torch.tensor([], dtype=DTYPE)#valiational dist. mean
    for city in cities:
        c2Sigma = torch.cat([c2Sigma, torch.tensor([1e-2]*c2S[city], dtype=DTYPE)], 0)
        c2eta_v = torch.cat([c2eta_v, torch.tensor([[1e-1]*L for i in range(c2S[city])])])
        c2W_v = torch.cat([c2W_v, torch.empty(c2S[city],L).uniform_(1e-4, 2e-4).type(DTYPE)], 0)
    c2Sigma = c2Sigma.requires_grad_(True)
    c2eta_v = c2eta_v.requires_grad_(True)
    c2W_v = c2W_v.requires_grad_(True)
    return Beta, c2Sigma, c2eta_p, c2W_p, c2eta_v, c2W_v

def extract_params(city, S):
    eta_p = c2eta_p[c2ids_for_W[city]]
    W_p = c2W_p[c2ids_for_W[city]]
    Sigma = c2Sigma[S:S+c2S[city]]
    eta_v = c2eta_v[S:S+c2S[city]]
    W_v = c2W_v[S:S+c2S[city]]
    S += c2S[city]
    return eta_p, W_p, Sigma, eta_v, W_v, S

def calcu_loss():
    S = 0
    loss = 0
    for city in cities:
        eta_p, W_p, Sigma, eta_v, W_v, S = extract_params(city, S)
        W = W_v + torch.normal(0, 1, size=(c2S[city], L), dtype=DTYPE) * torch.sqrt(eta_v**2)
        C = calcu_C(city, Sigma, W)
        KL = calcu_KL(W_p, eta_p, W_v, eta_v)
        loss += ( 0.5 * torch.mv(torch.linalg.solve(C, c2y[city].reshape([c2N[city],1])).t(), c2y[city])
                  + 0.5 * torch.logdet(C) + KL )
    return loss

def calcu_KL(W_p, eta_p, W_v, eta_v):
    KL = torch.sum(torch.log(eta_p**2) - torch.log(eta_v**2)
                   + ( (eta_v**2) ** 2 + (W_v - W_p) ** 2) / (2 * (eta_p**2) ** 2))
    return KL

def calcu_C(city, Sigma, W):
    def base(id):
        b = 0
        for i in range(id):
            b += len(cp2r[city][cd2p[city][c2datasets[city][i]]])
        return b

    C = torch.zeros([c2CN[city],c2CN[city]], dtype=DTYPE)
    for id1, id2 in list(itertools.combinations_with_replacement(range(c2S[city]), 2)):
        p1 = cd2p[city][c2datasets[city][id1]]
        p2 = cd2p[city][c2datasets[city][id2]]
        R1 = len(cp2r[city][p1])
        R2 = len(cp2r[city][p2])

        dists = cpp2dists[(city,p1,p2)]
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
    C = C[:,c2ava_ids[city]]
    C = C[c2ava_ids[city],:]
    return C

def torch_opt():
    optim = torch.optim.Adam([Beta,c2Sigma,c2eta_p,c2W_p,c2eta_v,c2W_v], lr=args.learn_rate)
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

def prediction(pred_city, target_id):
    def make_S():
        S = 0
        for city in cities:
            if city == pred_city:
                break
            else:
                S += c2S[city]
        return S
    #------
    S = make_S()
    city = pred_city
    eta_p, W_p, Sigma, eta_v, W_v, S = extract_params(city, S)
    CN, S = c2CN[city], c2S[city]
    d2p, p2r, datasets = cd2p[city], cp2r[city], c2datasets[city]
    y, grids = c2y[city], c2grids[city]
    #-----

    T = 100
    for t in range(T):
        W = W_v + torch.normal(0, 1, size=(c2S[city], L), dtype=DTYPE) * torch.sqrt(eta_v**2)
        C = calcu_C(city, Sigma, W)

        H = torch.zeros([grids.shape[0],CN], dtype=DTYPE)
        counter = 0
        for id in range(S):
            weight = (W[target_id] * W[id]).reshape([1,L])
            p = d2p[datasets[id]]
            for r in p2r[p]:
                dists = cpr2dists[(city,p,r)]
                Z1 = dists[:,0]/Beta.reshape([L,1])
                Z2 = dists[:,1]/Beta.reshape([L,1])
                V = dists[:,2]
                G = (torch.erf(Z1) + torch.erf(Z2)) * rootpi * Beta.reshape([L,1]) / 2
                h = torch.mm(weight, G/V).t()
                H[:,counter] = h.squeeze()
                counter += 1
        H = H[:,c2ava_ids[city]]
        if t == 0:
            pred = torch.mm(H, torch.linalg.solve(C, y.reshape([c2N[city],1])))
        else:
            pred += torch.mm(H, torch.linalg.solve(C, y.reshape([c2N[city],1])))

    return pred/float(T)

def aggregation(p, pred):
    agg_pred = []
    for r in p2r[p]:
        left, right = r[0], r[1]
        ids = np.where( (grids >= left) & (grids <= right))[0]
        agg_pred.append([r[0],r[1],torch.mean(pred[ids]).item()])
    return np.array(agg_pred)

def output_params():
    names = []
    for city in cities:
        for name in c2datasets[city]:
            names.append([city, name])
            
    csv_write(save_dir + '/Beta.csv', Beta.numpy())            
    csv_write(save_dir + '/c2Sigma.csv', np.hstack([names, c2Sigma.numpy().reshape([-1,1])]))
    csv_write(save_dir + '/c2W_v.csv', np.hstack([names, c2W_v.numpy()]))
    csv_write(save_dir + '/c2eta_v.csv', np.hstack([names, c2eta_v.numpy()]))
    csv_write(save_dir + '/c2W_p.csv', np.hstack([np.array(uniq_datasets).reshape([-1, 1]),
                                                  c2W_p.numpy()]))
    csv_write(save_dir + '/c2eta_p.csv', np.hstack([np.array(uniq_datasets).reshape([-1, 1]),
                                                    c2eta_p.numpy()]))

def visualization(city):
    fig = plt.figure(figsize=(24, 12), facecolor='white', dpi=dpi)
    fig.suptitle(city)
    N = len(datasets)
    ylabels = datasets
    for i in range(S):
        dataset, partition = c2filenames[city][i]
        i_dir = 'data/' + city
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
        fig.add_subplot(N, 1, i+1, frameon=True)
        plt.plot(X, Y, color='steelblue', alpha=0.8, lw=0.8)
        plt.plot(c2grids[city], preds[:,i].numpy(), color='crimson', alpha=0.8, lw=0.8)

        plt.ylabel(ylabels[i])
        plt.xlim([0,1])
        plt.tight_layout()
    plt.savefig(save_dir + '/pred_' + city + '.pdf', format='pdf')
    plt.close()

    
if __name__ == "__main__":

    # setting
    args = get_args()
    torch.manual_seed(args.seed)
    input_file = args.Exp + '/input_trans.csv'
    save_dir = args.Exp + '/result/A-MoGP_trans/' + str(args.latent_process)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None

    # input
    cities, c2datasets, c2S, c2partitions, c2P, c2filenames, cd2p = read_input_file_trans(input_file)
    target_city = cities[0]
    ## training data
    data_dir = 'data/'
    c2y, c2N, c2CN, cd2means, cd2stds, cp2r, c2ava_ids, c2grids, target_ava_ids = input_data_trans(data_dir, cities, c2filenames)
    ## test data
    target_partition = 'week'
    target_data = c2datasets[target_city][0]
    filename = 'data/' + target_city + '/' + target_partition + '/' + target_data + '/*.csv'
    test_true, test_ava_ids = input_target_data_trans(filename, target_city, target_partition, cp2r)
    cpp2dists, cpr2dists = input_dists_trans(cities, c2partitions, cp2r, c2grids)

    # inference
    L = args.latent_process
    c2Epsilon, total_S, uniq_S, uniq_datasets, c2ids_for_W = preparation()
    Beta, c2Sigma, c2eta_p, c2W_p, c2eta_v, c2W_v = init_params()
    losses = torch_opt()
    Beta = Beta.detach(); c2Sigma = c2Sigma.detach(); c2eta_p = c2eta_p.detach()
    c2W_p = c2W_p.detach(); c2eta_v = c2eta_v.detach(); c2W_v = c2W_v.detach()
    output_params()
    
    # prediction (grids)
    target_id = [i for i in range(c2S[target_city]) if target_data == c2datasets[target_city][i]][0]
    pred = prediction(target_city, target_id)
    pred = inv_normalization(pred, cd2means[target_city][target_data], cd2stds[target_city][target_data])

    #---
    city = target_city
    d2p, p2r, datasets = cd2p[city], cp2r[city], c2datasets[city]
    y, means, stds = c2y[city], cd2means[city], cd2stds[city]
    y = c2y[city]
    N, S = c2N[city], c2S[city]
    grids = c2grids[city]
    #---
    
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
    for city in cities:
        preds = torch.zeros([c2grids[city].shape[0],len(c2datasets[city])], dtype=DTYPE)
        for target_id in range(c2S[city]):
            pred = prediction(city, target_id)
            preds[:,target_id] = pred.squeeze()
        visualization(city)
