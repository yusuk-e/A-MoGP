# Aggregated Multi-output Gaussian Processes | 2022
# Yusuke Tanaka

import pdb
import os
import time
import json
import argparse
import torch

from utils import *

DTYPE = torch.float64
th = 1e-5
eps = 1e-8

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--cuda_no', default='0', type=str)
    parser.add_argument('--latent_process', default=1, type=int, help='number of latent GPs')
    parser.add_argument('--g_scale', default=0.5, type=float, help='grid_size (ratio to 1km)')
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
        c2Epsilon[city] = (torch.eye(c2N[city], dtype=DTYPE) * eps).to(device)
        all_datasets.extend(c2datasets[city])
    total_S = len(all_datasets)
    uniq_datasets = list(set(all_datasets))
    uniq_S = len(uniq_datasets)

    c2ids_for_W = defaultdict(str)
    for city in cities:
        c2ids_for_W[city] = [uniq_datasets.index(c2datasets[city][i]) for i in range(c2S[city])]
    return c2Epsilon, total_S, uniq_S, uniq_datasets, c2ids_for_W

def init_params():
    Beta = torch.tensor([1]*L, dtype=DTYPE).to(device).requires_grad_(True)
    c2eta_p = torch.tensor([[1e-1]*L for i in range(uniq_S)]).to(device).requires_grad_(True)
    c2W_p = torch.empty(uniq_S,L).uniform_(1e-4, 2e-4).type(DTYPE).to(device).requires_grad_(True)#prior mean
    c2Sigma = torch.tensor([], dtype=DTYPE)
    c2eta_v = torch.tensor([], dtype=DTYPE)#valiational dist. variance
    c2W_v = torch.tensor([], dtype=DTYPE)#valiational dist. mean
    for city in cities:
        c2Sigma = torch.cat([c2Sigma, torch.tensor([1e-1]*c2S[city], dtype=DTYPE)], 0)
        c2eta_v = torch.cat([c2eta_v, torch.tensor([[1e-1]*L for i in range(c2S[city])])])
        c2W_v = torch.cat([c2W_v, torch.empty(c2S[city],L).uniform_(1e-4, 2e-4).type(DTYPE)], 0)
    c2Sigma = c2Sigma.to(device).requires_grad_(True)
    c2eta_v = c2eta_v.to(device).requires_grad_(True)
    c2W_v = c2W_v.to(device).requires_grad_(True)
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
        W = W_v + torch.normal(0, 1, size=(c2S[city], L), dtype=DTYPE).to(device) * eta_v**2
        C = calcu_C(city, Sigma, W)
        KL = calcu_KL(W_p, eta_p, W_v, eta_v)
        loss += ( 0.5 * torch.mv(torch.linalg.solve(C, c2y[city].reshape([c2N[city],1])).T,
                                c2y[city]) + 0.5 * torch.logdet(C) + KL )
    return loss

def calcu_C(city, Sigma, W):
    def base(id):
        b = 0
        for i in range(id):
            b += len(p2r[d2p[datasets[i]]])
        return b

    N, S, D = c2N[city], c2S[city], c2D[city]
    d2p, p2r, datasets = cd2p[city], cp2r[city], c2datasets[city]
    C = torch.zeros([N,N], dtype=DTYPE).to(device)
    cov = torch.exp(-0.5 / Beta.reshape([L,1])**2 * c2norm[city])
    for id1, id2 in list(itertools.combinations_with_replacement(range(S), 2)):
        p1 = d2p[datasets[id1]]
        p2 = d2p[datasets[id2]]
        R1 = len(p2r[p1])
        R2 = len(p2r[p2])
        aggM = cpp2aggM[city][(p1,p2)]
        weight = (W[id1] * W[id2]).reshape([1,L])
        tmp = torch.mm(weight, cov).t()
        c = torch.sparse.mm(aggM, tmp).reshape([R1,R2])
        b1 = base(id1)
        b2 = base(id2)
        if id1 == id2:
            C[b1:b1+R1, b2:b2+R2] = c + Sigma[id1]**2 * torch.eye(R1, dtype=DTYPE).to(device)
        else:
            C[b1:b1+R1, b2:b2+R2] = c
            C[b2:b2+R2, b1:b1+R1] = c.t()
    return C + eps

def calcu_KL(W_p, eta_p, W_v, eta_v):
    KL = torch.sum(torch.log(eta_p**2) - torch.log(eta_v**2) + ( (eta_v**2) ** 2 + (W_v - W_p) ** 2)
                   / (2 * (eta_p**2) ** 2))
    return KL

def torch_opt():
    optim = torch.optim.Adam([Beta, c2Sigma, c2eta_p, c2W_p, c2eta_v, c2W_v], lr=args.learn_rate)
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

def prediction():
    def make_S():
        S = 0
        for city in cities:
            if city == target_city:
                break
            else:
                S += c2S[city]
        return S

    S = make_S()
    eta_p, W_p, Sigma, eta_v, W_v, S = extract_params(city, S)
    N, S, D, G = c2N[city], c2S[city], c2D[city], c2G[city]
    norm, pr2aggM = c2norm[city], cpr2aggM[city]
    target_id = [i for i in range(S) if target_data == datasets[i]][0]

    T = 100
    for t in range(T):
        W = W_v + torch.normal(0, 1, size=(c2S[city], L), dtype=DTYPE).to(device) * eta_v**2
        C = calcu_C(city, Sigma, W)

        H = torch.zeros([G,N], dtype=DTYPE).to(device)
        cov = torch.exp(-0.5 / Beta.reshape([L,1])**2 * norm)
        counter = 0
        for id in range(S):
            weight = (W[target_id] * W[id]).reshape([1,L])
            tmp = torch.mm(weight, cov).t()
            p = d2p[datasets[id]]
            for r in p2r[p]:
                aggM = pr2aggM[(p,r)]
                h = torch.mm(aggM, tmp).reshape([1,G])
                H[:,counter] = h
                counter += 1
        if t == 0:
            pred = torch.mm(H, torch.linalg.solve(C, y.reshape([N,1])))
        else:
            pred += torch.mm(H, torch.linalg.solve(C, y.reshape([N,1])))
    return pred/float(T)

def prediction_org():
    def make_S():
        S = 0
        for city in cities:
            if city == target_city:
                break
            else:
                S += c2S[city]
        return S

    S = make_S()
    eta_p, W_p, Sigma, eta_v, W_v, S = extract_params(city, S)
    N, S, D, G = c2N[city], c2S[city], c2D[city], c2G[city]
    norm, pr2aggM = c2norm[city], cpr2aggM[city]
    target_id = [i for i in range(S) if target_data == datasets[i]][0]

    
    W = W_v
    C = calcu_C(city, Sigma, W)

    H = torch.zeros([G,N], dtype=DTYPE).to(device)
    cov = torch.exp(-0.5 / Beta.reshape([L,1])**2 * norm)
    counter = 0
    for id in range(S):
        weight = (W[target_id] * W[id]).reshape([1,L])
        tmp = torch.mm(weight, cov).t()
        p = d2p[datasets[id]]
        for r in p2r[p]:
            aggM = pr2aggM[(p,r)]
            h = torch.mm(aggM, tmp).reshape([1,G])
            H[:,counter] = h
            counter += 1
    return torch.mm(H, torch.solve(y.reshape([N,1]), C)[0])
    #return torch.mm(H, torch.linalg.solve(C, y.reshape([-1,1])))
    
def aggregation(p, pred):
    agg_pred = []
    for r in p2r[p]:
        grid_ids = pr2g[p][r]
        agg_pred.append([r, torch.mean(pred[grid_ids]).item()])
    return np.array(agg_pred)

def visualization(p, values, filename, cname):
    i_file = glob.glob('boundary/' + target_city + '/input/' + p + '/*.shp')[0]
    df = shapefile_read(i_file)
    df = epsg_conv(df, 3857)
    df = shapedata_add_col(df, values, target_city, p)
    vmin = min(df['values'].values)
    vmax = max(df['values'].values)
    o_file = save_dir + '/' + filename + '_' + p + '.pdf'
    mapping(o_file, df, 'values', vmin, vmax, cname)

def output_params():
    names = []
    for city in cities:
        for name in c2datasets[city]:
            names.append([city, name])

    csv_write(save_dir + '/Beta.csv', Beta.cpu().detach().numpy())
    csv_write(save_dir + '/c2Sigma.csv', np.hstack([names, c2Sigma.cpu().detach().numpy().reshape([-1,1])]))
    csv_write(save_dir + '/c2W_v.csv', np.hstack([names, c2W_v.cpu().detach().numpy()]))
    csv_write(save_dir + '/c2eta_v.csv', np.hstack([names, c2eta_v.cpu().detach().numpy()]))
    csv_write(save_dir + '/c2W_p.csv', np.hstack([np.array(uniq_datasets).reshape([-1,1]), c2W_p.cpu().detach().numpy()]))
    csv_write(save_dir + '/c2eta_p.csv', np.hstack([np.array(uniq_datasets).reshape([-1,1]), c2eta_p.cpu().detach().numpy()]))
    
def input_target_data(filename):
    i_file = glob.glob(filename)[0]
    Dat = csv_read(i_file)
    for d in Dat:
        region = str(d[0])
        if target_partition not in cp2r[target_city]:
            cp2r[target_city][target_partition] = [region]
        else:
            if region not in cp2r[target_city][target_partition]:
                cp2r[target_city][target_partition].extend([region])
    return np.array(Dat)[:,1].astype(float)
    

if __name__ == '__main__':

    # setting
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda:' + args.cuda_no if torch.cuda.is_available() else 'cpu')
    input_file = args.Exp + '/input_trans.csv'
    save_dir = args.Exp + '/result/A-MoGP_trans/' + str(args.latent_process)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None

    # input
    cities, c2datasets, c2S, c2partitions, c2P, c2filenames, cd2p = read_input_file_trans(input_file)
    target_city = cities[0]
    ## training data
    c2y, c2N, cd2means, cd2stds, cp2r = input_data_trans(cities, c2filenames, device)
    c2norm, c2D = input_norm_trans(args.g_scale, cities, device)
    cpp2aggM, cpr2aggM, c2G = input_aggM_trans(args.g_scale, cities, c2partitions, cp2r, c2D, device)
    pr2g = pkl_read('boundary/' + target_city + '/output/' + str(args.g_scale) + '/pr2grid_ids.pkl')
    ## test data
    target_data = c2datasets[target_city][0]
    target_partition = select_partition(target_city, target_data)
    filename = 'data/' + target_city + '/' + target_data + '/' + target_partition + '/*.csv'
    test_true = input_target_data(filename)
    
    # inference
    L = args.latent_process
    c2Epsilon, total_S, uniq_S, uniq_datasets, c2ids_for_W = preparation()
    Beta, c2Sigma, c2eta_p, c2W_p, c2eta_v, c2W_v = init_params()
    losses = torch_opt()
    output_params()

    #---
    city = target_city
    d2p, p2r, datasets = cd2p[city], cp2r[city], c2datasets[city]
    y, means, stds = c2y[city], cd2means[city], cd2stds[city]
    N, S, D, G = c2N[city], c2S[city], c2D[city], c2G[city]
    #---
    
    # prediction (grids)
    pred = prediction()
    pred = inv_normalization(pred, means[target_data], stds[target_data])
    
    # MAPE (train)
    p = d2p[target_data]
    agg_pred_train = aggregation(p, pred)
    train_true = inv_normalization(y[0:len(p2r[p])].clone().cpu(),
                             means[target_data], stds[target_data]).numpy()
    est = agg_pred_train[:,1].astype(float)
    train_err = MAPE(train_true, est)

    # MAPE (test)
    p = target_partition
    agg_pred_test = aggregation(p, pred)
    est = agg_pred_test[:,1].astype(float)
    test_err = MAPE(test_true, est)

    # save
    path = save_dir + '/result.csv'
    csv_write(path, np.array(['train_err',train_err,'test_err',test_err]))
    print("train_err {:.2e}, test_err {:.2e}".format(train_err, test_err))
    
    path = save_dir + '/setting.json'
    with open(path, 'w') as f:
        json.dump(vars(args), f)

    # visualization
    p = d2p[target_data]
    visualization(p, np.stack([agg_pred_train[:,0], train_true]).T, 'true', 'Blues')
    visualization(p, agg_pred_train, 'prediction', 'Blues')
    p = target_partition
    visualization(p, np.stack([agg_pred_test[:,0], test_true]).T, 'true', 'Blues')
    visualization(p, agg_pred_test, 'prediction', 'Blues')

