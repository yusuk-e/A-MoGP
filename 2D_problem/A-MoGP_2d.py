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

def init_params():
    Beta = torch.tensor([1]*L, dtype=DTYPE).to(device).requires_grad_(True)
    Sigma = torch.tensor([1e-1]*S, dtype=DTYPE).to(device).requires_grad_(True)
    W = torch.empty(S,L).uniform_(1e-4, 2e-4).type(DTYPE).to(device).requires_grad_(True)
    return Beta, Sigma, W

def calcu_loss():
    C = calcu_C()
    loss = 0.5 * torch.mv( torch.linalg.solve(C, y.reshape([-1,1])).T, y) + 0.5 * torch.logdet(C)
    return loss

def calcu_C():
    def base(id):
        b = 0
        for i in range(id):
            b += len(p2r[d2p[datasets[i]]])
        return b

    C = torch.zeros([N,N], dtype=DTYPE).to(device)
    cov = torch.exp(-0.5 / Beta.reshape([L,1]) ** 2 * norm)
    for id1, id2 in list(itertools.combinations_with_replacement(range(S), 2)):
        p1 = d2p[datasets[id1]]
        p2 = d2p[datasets[id2]]
        R1 = len(p2r[p1])
        R2 = len(p2r[p2])
        aggM = pp2aggM[(p1,p2)]
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
    H = torch.zeros([G,N], dtype=DTYPE).to(device)
    cov = torch.exp(-0.5 / Beta.reshape([L,1]) ** 2 * norm)
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
    return torch.mm(H, torch.linalg.solve(C, y.reshape([-1,1])))

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
    csv_write(save_dir + '/Beta.csv', Beta.cpu().detach().numpy())
    csv_write(save_dir + '/Sigma.csv', Sigma.cpu().detach().numpy())
    names = np.array(datasets).reshape(len(datasets),1)
    csv_write(save_dir + '/W.csv', np.hstack([names, W.cpu().detach().numpy()]))

def input_target_data(filename):
    i_file = glob.glob(filename)[0]
    Dat = csv_read(i_file)
    for d in Dat:
        region = str(d[0])
        if target_partition not in p2r:
            p2r[target_partition] = [region]
        else:
            if region not in p2r[target_partition]:
                p2r[target_partition].extend([region])
    return np.array(Dat)[:,1].astype(float)
    

if __name__ == '__main__':

    # setting
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda:' + args.cuda_no if torch.cuda.is_available() else 'cpu')
    input_file = args.Exp + '/input.csv'
    save_dir = args.Exp + '/result/A-MoGP/' + str(args.latent_process)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None

    # input
    target_city, datasets, S, partitions, P, filenames, d2p = read_input_file(input_file)
    ## training data
    data_dir = 'data/' + target_city
    y, N, means, stds, p2r = input_data(data_dir, filenames, device)
    norm_dir = 'boundary/' + target_city + '/output/' + str(args.g_scale)
    norm, D = input_norm(norm_dir + '/norm.npy', device)
    aggM_dir = 'boundary/' + target_city + '/output/' + str(args.g_scale)
    pp2aggM, pr2aggM, G = input_aggM(aggM_dir + '/rr_count.df',
                                     aggM_dir + '/rp_count.df', partitions, p2r, D, device)
    pr2g = pkl_read('boundary/' + target_city + '/output/' + str(args.g_scale) + '/pr2grid_ids.pkl')
    ## test data
    target_data = datasets[0]
    target_partition = select_partition(target_city, target_data)
    filename = 'data/' + target_city + '/' + target_data + '/' + target_partition + '/*.csv'
    test_true = input_target_data(filename)
    
    # inference
    L = args.latent_process
    eps = (torch.eye(N, dtype=DTYPE) * eps).to(device)
    Beta, Sigma, W = init_params()
    losses = torch_opt()
    output_params()

    # prediction (grids)
    target_id = [i for i in range(S) if target_data == datasets[i]][0]
    pred = prediction(target_id)
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
