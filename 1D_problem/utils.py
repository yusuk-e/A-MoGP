# Aggregated Multi-output Gaussian Processes | 2022
# Yusuke Tanaka

import pdb
import csv
import glob
import itertools
from collections import defaultdict
import numpy as np
import torch
DTYPE = torch.float64

def csv_read(file):
    f = open(file)
    csvReader = csv.reader(f)
    D = []
    for row in csvReader:
        D.append(row)
    return D

def csv_write(file, D):
    f = open(file,'w')
    csvWriter = csv.writer(f,lineterminator='\n')
    if np.ndim(D) == 1:
        csvWriter.writerow(D)
    elif np.ndim(D) == 2:
        for i in range(np.shape(D)[0]):
            line = D[i]
            csvWriter.writerow(line)
    f.close()
    
def read_input_file(file):
    datasets = []
    partitions = []
    filenames = []
    d2p = {}
    Dat = csv_read(file)
    for city, dataset, partition in Dat:
        if dataset not in datasets:
            datasets.append(dataset)
        if partition not in partitions:
            partitions.append(partition)
        filenames.append([dataset, partition])
        d2p[dataset] = partition
    return city, datasets, len(datasets), partitions, len(partitions), filenames, d2p

def read_input_file_trans(file):
    cities = []
    c2datasets = defaultdict(str)
    c2partitions = defaultdict(str)
    c2filenames = defaultdict(str)
    cd2p = defaultdict(lambda: defaultdict(str))
    c2S = defaultdict(str)
    c2P = defaultdict(str)
    Dat = csv_read(file)
    for city, dataset, partition in Dat:
        if city not in cities:
            cities.append(city)
        if len(c2datasets[city]) == 0:
            c2datasets[city] = [dataset]
        else:
            c2datasets[city].append(dataset)
        if len(c2partitions[city]) == 0:
            c2partitions[city] = [partition]
        else:
            if partition not in c2partitions[city]:
                c2partitions[city].append(partition)
        if len(c2filenames[city]) == 0:
            c2filenames[city] = [[dataset, partition]]
        else:
            c2filenames[city].append([dataset, partition])
        cd2p[city][dataset] = partition
    for city in cities:
        c2P[city] = len(c2partitions[city])
        c2S[city] = len(c2datasets[city])
    return cities, c2datasets, c2S, c2partitions, c2P, c2filenames, cd2p

def input_data(i_dir, filenames):
    y = torch.tensor([], dtype=DTYPE)
    p2regions = {}
    means = {}
    stds = {}
    ava_ids = []
    counter = 0
    flag = 0
    target_ava_ids = []
    for dataset, partition in filenames:
        i_file = glob.glob(i_dir + '/' + partition + '/' + dataset + '/*.csv')[0]
        Dat = csv_read(i_file)
        ys = []
        for d in Dat:
            region = (float(d[1]),float(d[2]))
            if partition not in p2regions:
                p2regions[partition] = [region]
            else:
                if region not in p2regions[partition]:
                    p2regions[partition].extend([region])
            if d[3] != 'NA':
                ava_ids.append(counter)
                ys.append(float(d[3]))
                if flag == 0:
                    target_ava_ids.append(counter)
            counter += 1
        ys = torch.tensor(ys, dtype=DTYPE)
        ys, mean, std = normalization(ys)
        y = torch.cat([y, ys], 0)
        means[dataset] = mean.item()
        stds[dataset] = std.item()
        flag = 1
    return y, y.size()[0], counter, means, stds, p2regions, ava_ids, target_ava_ids

def input_data_trans(d_dir, cities, c2filenames):
    c2y = defaultdict(str)
    cp2regions = defaultdict(lambda: defaultdict(str))
    cd2means = defaultdict(lambda: defaultdict(str))
    cd2stds = defaultdict(lambda: defaultdict(str))
    c2N = defaultdict(str)
    c2CN = defaultdict(str)
    c2ava_ids = defaultdict(str)
    c2grids = defaultdict(str)
    flag = 0
    target_ava_ids = []
    for city in cities:
        i_dir = d_dir + '/' + city
        y_c = torch.tensor([], dtype=DTYPE)
        filenames = c2filenames[city]
        ava_ids = []
        counter = 0
        for dataset, partition in filenames:
            i_file = glob.glob(i_dir + '/' + partition  + '/' + dataset + '/*.csv')[0]
            Dat = csv_read(i_file)
            ys = []
            for d in Dat:
                region = (float(d[1]),float(d[2]))
                if partition not in cp2regions[city]:
                    cp2regions[city][partition] = [region]
                else:
                    if region not in cp2regions[city][partition]:
                        cp2regions[city][partition].extend([region])
                if d[3] != 'NA':
                    ava_ids.append(counter)
                    ys.append(float(d[3]))
                    if flag == 0:
                        target_ava_ids.append(counter)
                counter += 1
            ys = torch.tensor(ys, dtype=DTYPE)
            ys, mean, std = normalization(ys)
            y_c = torch.cat([y_c, ys], 0)
            cd2means[city][dataset] = mean.item()
            cd2stds[city][dataset] = std.item()
            flag = 1
        c2y[city] = y_c
        c2N[city] = y_c.size()[0]
        c2ava_ids[city] = ava_ids
        c2CN[city] = counter
        
        start = np.min(np.array(Dat)[:,1].astype(float))
        end = np.max(np.array(Dat)[:,2].astype(float))
        G = int((end-start)/1e-4)
        c2grids[city] = np.linspace(start, end, G)
    return c2y, c2N, c2CN, cd2means, cd2stds, cp2regions, c2ava_ids, c2grids, target_ava_ids

def input_target_data(filename, target_partition, p2r):
    i_file = glob.glob(filename)[0]
    Dat = csv_read(i_file)
    yt = []
    ava_ids = []
    for d_id, d in enumerate(Dat):
        region = (float(d[1]),float(d[2]))
        if target_partition not in p2r:#for validation
            p2r[target_partition] = [region]
        else:
            if region not in p2r[target_partition]:
                p2r[target_partition].extend([region])
        if d[3] != 'NA':
            ava_ids.append(d_id)
            yt.append(float(d[3]))
    start = np.min(np.array(Dat)[:,1].astype(float))
    end = np.max(np.array(Dat)[:,2].astype(float))
    G = int((end-start)/1e-4)
    grids = np.linspace(start, end, G)
    return np.array(yt), grids, ava_ids

def input_target_data_trans(filename, target_city, target_partition, cp2r):
    i_file = glob.glob(filename)[0]
    Dat = csv_read(i_file)
    yt = []
    ava_ids = []
    for d_id, d in enumerate(Dat):
        region = (float(d[1]),float(d[2]))
        if target_partition not in cp2r[target_city]:#for validation
            cp2r[target_city][target_partition] = [region]
        else:
            if region not in cp2r[target_city][target_partition]:
                cp2r[target_city][target_partition].extend([region])
        if d[3] != 'NA':
            ava_ids.append(d_id)
            yt.append(float(d[3]))
    return np.array(yt), ava_ids

def normalization(y):
    mean = torch.mean(y)
    std = torch.std(y)
    for i in range(len(y)):
        y[i] = (y[i] - mean) / float(std)
    return y, mean, std

def inv_normalization(f, mean, std):
    for i in range(len(f)):
        f[i] = f[i] * std + mean
    return f

def input_dists(partitions, p2r, grids):
    grids = torch.tensor(grids)
    pp2dists = {}
    for p1, p2 in list(itertools.product(partitions, partitions)):
        regions1 = p2r[p1]
        regions2 = p2r[p2]
        dists = []
        for r1, r2 in list(itertools.product(regions1, regions2)):
            left_r1 = r1[0]; right_r1 = r1[1]
            left_r2 = r2[0]; right_r2 = r2[1]
            Z1 = right_r1 - left_r2; Z2 = right_r2 - left_r1
            Z3 = right_r1 - right_r2; Z4 = left_r1 - left_r2
            period1 = right_r1 - left_r1; period2 = right_r2 - left_r2
            dists.append([Z1,Z2,Z3,Z4,period1,period2])
        pp2dists[(p1,p2)] = torch.tensor(dists)

    pr2dists = {}
    for p in partitions:
        for r in p2r[p]:
            left = r[0]; right = r[1]
            Z1 = right - grids; Z2 = grids - left
            period = torch.tensor([right-left]*grids.shape[0])
            dists = torch.stack([Z1,Z2,period]).t()
            pr2dists[(p,r)] = dists
    return pp2dists, pr2dists

def input_dists_trans(cities, c2partitions, cp2r, c2grids):
    cpp2dists = {}
    cpr2dists = {}
    for city in cities:
        partitions = c2partitions[city]
        for p1, p2 in list(itertools.product(partitions, partitions)):
            regions1 = cp2r[city][p1]
            regions2 = cp2r[city][p2]
            dists = []
            for r1, r2 in list(itertools.product(regions1, regions2)):
                left_r1 = r1[0]; right_r1 = r1[1]
                left_r2 = r2[0]; right_r2 = r2[1]
                Z1 = right_r1 - left_r2; Z2 = right_r2 - left_r1
                Z3 = right_r1 - right_r2; Z4 = left_r1 - left_r2
                period1 = right_r1 - left_r1; period2 = right_r2 - left_r2
                dists.append([Z1,Z2,Z3,Z4,period1,period2])
            cpp2dists[(city,p1,p2)] = torch.tensor(dists)

        for p in partitions:
            for r in cp2r[city][p]:
                left = r[0]; right = r[1]
                grids = torch.tensor(c2grids[city])
                Z1 = right - grids; Z2 = grids - left
                period = torch.tensor([right-left]*grids.shape[0])
                dists = torch.stack([Z1,Z2,period]).t()
                cpr2dists[(city,p,r)] = dists
    return cpp2dists, cpr2dists

def MAPE(true, est):
    S = 0
    errs = []
    for i in range(len(est)):
        t = true[i]
        e = est[i]
        err = np.abs((t - e) / float(t + 1e-5))
        S += err
        errs.append(err)
    ave_err = S / float(len(est))
    std_dev = np.std(errs)
    std_err = np.std(errs) / np.sqrt(len(est))
    return ave_err#, std_dev, std_err, errs
