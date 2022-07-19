# Aggregated Multi-output Gaussian Processes | 2022
# Yusuke Tanaka

import pdb
import csv
from collections import defaultdict
import glob
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import torch
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

sns.set()
sns.set_style("white")
sns.set_context("paper", 1.5, {"lines.linewidth": 1.5})
sns.set_palette("deep")
myblue = 'navy'
dpi=100
DTYPE = torch.float64


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def pkl_read(file):
    f = open(file, 'rb')
    D = pickle.load(f)
    f.close()
    return D

def pkl_write(file, D):
    f = open(file,'wb')
    pickle.dump(D,f,protocol=4)
    f.close()

def df_read(file):
    D = pd.read_pickle(file,compression='xz')
    return D

def df_write(file, D):
    D.to_pickle(file,compression='xz')

def shapefile_read(file):
    df = gpd.read_file(file)
    return df

def epsg_conv(df, value):
    df = df.to_crs(epsg=value)
    return df

def boundary_plot(file, df):
    df.boundary.plot(figsize=(10,10), edgecolor='black', linewidth=0.3)
    plt.savefig(file, format='png', dpi=dpi)
    plt.close()

def point_plot(file, df, D):
    points = pd.DataFrame({'x': D[:,0], 'y': D[:,1]})
    points_geo = gpd.GeoDataFrame(geometry=gpd.points_from_xy(points.x, points.y))
    points_geo.crs = "EPSG:4326"
    points_geo = epsg_conv(points_geo, 3857)
    base = df.boundary.plot(figsize=(10,10), edgecolor='black', linewidth=0.3)
    points_geo.plot(ax=base, marker='o', color=myblue, markersize=0.1)
    plt.savefig(file, format='png', dpi=dpi)
    plt.close()

def define_r_name(c, p):
    if c == 'CHI':
        if p == 'Community':
            r_name = 'area_num_1'
        elif p == 'Precinct':
            r_name = 'dist_num'
        elif p == 'Side':
            r_name = 'side'
    elif c == 'NYC':
        if p == 'Borough':
            r_name = 'boro_code'
        elif p == 'Community':
            r_name = 'boro_cd'
        elif p == 'Precinct':
            r_name = 'precinct'
        elif p == 'UHF42':
            r_name = 'UHFCODE'
        elif p == 'Zip':
            r_name = 'ZIPCODE'
    return r_name

def conv_Borough_region(r):
    if r == 'staten island':
        code = 5
    elif r == 'bronx':
        code = 2
    elif r == 'queens':
        code = 4
    elif r == 'brooklyn':
        code = 3
    elif r== 'manhattan':
        code = 1
    else:
        print('error')
    return code

def mapping(file, df, col, vmin, vmax, color):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cax.tick_params(labelsize=25)
    df.plot(
        edgecolor='silver',
        linewidth=0.5,
        column=col, 
        vmin=vmin,
        vmax=vmax,
        cmap=color, 
        ax=ax, 
        legend=True, 
        cax=cax)
    plt.savefig(file, format='pdf', dpi=dpi, bbox_inches='tight',pad_inches = 0.2)
    plt.close()

def shapedata_add_col(df, D, c, p):
    r_name = define_r_name(c, p)
    D = np.array(D)
    d = np.array(D[:,0]).astype(int)
    codes = df[r_name]
    vec = []
    for code in codes:
        id = np.where(d == int(code))[0][0]
        vec.append(D[id,1])
    df['values'] = np.array(vec).astype(float)
    return df

def bound(df):
    X = []
    Y = []
    for i, s in df.iterrows():
        polygon = unify_polygons(s)
        for p in polygon:
            lons, lats = zip(*list(p.exterior.coords))
            X.extend(lons)
            Y.extend(lats)
    return min(X), max(X), min(Y), max(Y)

def grid(xmin, xmax, ymin, ymax, alpha):
    xstep = alpha * 0.010966404715491394#1km
    ystep = alpha * 0.0090133729745762#1km
    x = xmin
    y = ymin
    grid = []
    col = row = 0
    cr_ids = []
    while(1):        
        if ymin <= y and y <= ymax:
            grid.append([x,y])
            cr_ids.append([col,row])
            y += ystep
            row += 1
        else:
            if xmin <= x and x <= xmax:
                x += xstep
                y = ymin
                col += 1
                row = 0
            else:
                break
    return np.array(grid), np.array(cr_ids)

def join_grid(df, grid):
    new_grid = []
    grid_ids = []
    for i, s in df.iterrows():
        polygon = unify_polygons(s)
        sub_grid_ids, sub_grid = extract_grid(grid, polygon)
        for i in range(len(sub_grid)):
            new_grid.append(sub_grid[i])
            grid_ids.append(sub_grid_ids[i])
    return np.array(new_grid), grid_ids

def unify_polygons(s):
    polygons = s['geometry']
    if np.size(polygons) > 1:
        polygon = [x for x in polygons]
    else:
        polygon = [polygons]
    return polygon

def extract_grid(grid, polygon):
    points = grid.astype(float)
    geometry = [Point(point) for point in points]
    df = pd.DataFrame([])
    crs = {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84'}
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    df = pd.DataFrame([])
    crs = {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84'}
    gd = gpd.GeoDataFrame(df, crs=crs, geometry=polygon)
    join = gpd.sjoin(gdf, gd, how="inner", op='intersects')
    sub_grid_ids = []
    sub_grid = []
    for i, s in join.iterrows():
        sub_grid_ids.append(i)
        sub_grid.append([s['geometry'].x, s['geometry'].y])
    return sub_grid_ids, sub_grid

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

def input_data(i_dir, filenames, device):
    y = torch.tensor([], dtype=DTYPE)
    p2regions = {}
    means = {}
    stds = {}
    for dataset, partition in filenames:
        i_file = glob.glob(i_dir + '/' + dataset + '/' + partition + '/*.csv')[0]
        Dat = csv_read(i_file)
        ys = []
        for d in Dat:
            region = str(d[0])
            if partition == 'Borough':
                region = str(conv_Borough_region(region))
            if partition not in p2regions:
                p2regions[partition] = [region]
            else:
                if region not in p2regions[partition]:
                    p2regions[partition].extend([region])
            ys.append(float(d[1]))
        ys = torch.tensor(ys, dtype=DTYPE)
        ys, mean, std = normalization(ys)
        y = torch.cat([y, ys], 0)
        means[dataset] = mean.item()
        stds[dataset] = std.item()
    y = y.to(device)
    return y, y.size()[0], means, stds, p2regions

def input_data_trans(cities, c2filenames, device):
    c2y = defaultdict(str)
    cp2regions = defaultdict(lambda: defaultdict(str))
    cd2means = defaultdict(lambda: defaultdict(str))
    cd2stds = defaultdict(lambda: defaultdict(str))
    c2N = defaultdict(str)
    for city in cities:
        i_dir = 'data/' + city
        y_c = torch.tensor([], dtype=DTYPE)
        filenames = c2filenames[city]
        for dataset, partition in filenames:
            i_file = glob.glob(i_dir + '/' + dataset + '/' + partition + '/*.csv')[0]
            Dat = csv_read(i_file)
            ys = []
            for d in Dat:
                region = str(d[0])
                if partition == 'Borough':
                    region = str(conv_Borough_region(region))
                if partition not in cp2regions[city]:
                    cp2regions[city][partition] = [region]
                else:
                    if region not in cp2regions[city][partition]:
                        cp2regions[city][partition].extend([region])
                ys.append(float(d[1]))
            ys = torch.tensor(ys, dtype=DTYPE)
            ys, mean, std = normalization(ys)
            y_c = torch.cat([y_c, ys], 0)
            cd2means[city][dataset] = mean.item()
            cd2stds[city][dataset] = std.item()
        y_c = y_c.to(device)
        c2y[city] = y_c
        c2N[city] = y_c.size()[0]
    return c2y, c2N, cd2means, cd2stds, cp2regions

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

def input_norm(i_file, device):
    norm = np.load(i_file)
    norm = torch.from_numpy(norm)
    norm = norm.to(device)
    return norm, norm.size()[0]

def input_norm_trans(g_scale, cities, device):
    c2norm = defaultdict(str)
    c2D = defaultdict(str)
    for city in cities:
        i_file = 'boundary/' + city + '/output/' + str(g_scale) + '/norm.npy'
        norm = np.load(i_file)
        norm = torch.from_numpy(norm)
        c2norm[city] = norm.to(device)
        c2D[city] = norm.size()[0]
    return c2norm, c2D

def input_aggM(i_file, i_file2, partitions, p2r, D, device):
    rr_count = df_read(i_file)
    pp2aggM = {}
    for p1, p2 in list(itertools.product(partitions, partitions)):
        sub_df = rr_count[ (rr_count['p1']==p1) & (rr_count['p2']==p2) ]
        counter = 0
        columns = []
        rows = []
        v = []
        sub_df_values = sub_df.values
        for i in range(sub_df.shape[0]):
            p1, p2, r1, r2, ids, counts = sub_df_values[i]
            columns.extend(ids)
            rows.extend([counter] * len(ids))
            v.extend(counts / sum(counts))
            counter += 1      

        ids = torch.tensor([rows, columns])
        v = torch.tensor(v, dtype=DTYPE)
        size = torch.Size([counter,D])
        aggM = torch.sparse.FloatTensor(ids.to(device), v.to(device), size)
        pp2aggM[(p1,p2)] = aggM

    rp_count = df_read(i_file2)
    G = len(set(rp_count['g_id'].values))
    pr2aggM = {}
    for p in partitions:
        for r in p2r[p]:
            sub_df = rp_count[ (rp_count['p']==p) & (rp_count['r']==r) ]
            counter = 0
            columns = []
            rows = []
            v = []
            sub_df_values = sub_df.values
            for i in range(sub_df.shape[0]):
                p, r, g_id, ids, counts = sub_df_values[i]
                columns.extend(ids)
                rows.extend([counter] * len(ids))
                v.extend(counts / sum(counts))
                counter += 1
            ids = torch.tensor([rows, columns])
            v = torch.tensor(v, dtype=DTYPE)
            size = torch.Size([G,D])
            pr2aggM[(p,r)] = torch.sparse.FloatTensor(ids.to(device), v.to(device), size)
    return pp2aggM, pr2aggM, G

def input_aggM_trans(g_scale, cities, c2partitions, cp2r, c2D, device):
    cpp2aggM = defaultdict(str)
    for city in cities:
        i_file = 'boundary/' + city + '/output/' + str(g_scale) + '/rr_count.df'
        rr_count = df_read(i_file)
        pp2aggM = {}
        partitions = c2partitions[city]
        for p1, p2 in list(itertools.product(partitions, partitions)):
            sub_df = rr_count[ (rr_count['p1']==p1) & (rr_count['p2']==p2) ]
            counter = 0
            columns = []
            rows = []
            v = []
            sub_df_values = sub_df.values
            for i in range(sub_df.shape[0]):
                p1, p2, r1, r2, ids, counts = sub_df_values[i]
                columns.extend(ids)
                rows.extend([counter] * len(ids))
                v.extend(counts / sum(counts))
                counter += 1
            ids = torch.tensor([rows, columns])
            v = torch.tensor(v, dtype=DTYPE)
            size = torch.Size([counter,c2D[city]])
            aggM = torch.sparse.FloatTensor(ids.to(device), v.to(device), size)
            pp2aggM[(p1,p2)] = aggM
        cpp2aggM[city] = pp2aggM

    cpr2aggM = defaultdict(str)
    c2G = defaultdict(str)
    for city in cities:
        i_file2 = 'boundary/' + city + '/output/' + str(g_scale) + '/rp_count.df'
        rp_count = df_read(i_file2)
        G = len(set(list(rp_count['g_id'].values)))
        pr2aggM = {}
        for p in c2partitions[city]:
            for r in cp2r[city][p]:
                sub_df = rp_count[ (rp_count['p']==p) & (rp_count['r']==r) ]
                counter = 0
                columns = []
                rows = []
                v = []
                sub_df_values = sub_df.values
                for i in range(sub_df.shape[0]):
                    p, r, g_id, ids, counts = sub_df_values[i]
                    columns.extend(ids)
                    rows.extend([counter] * len(ids))
                    v.extend(counts / sum(counts))
                    counter += 1
                ids = torch.tensor([rows, columns])
                v = torch.tensor(v, dtype=DTYPE)
                size = torch.Size([G,c2D[city]])
                pr2aggM[(p,r)] = torch.sparse.FloatTensor(ids.to(device), v.to(device), size)
        cpr2aggM[city] = pr2aggM
        c2G[city] = G
    return cpp2aggM, cpr2aggM, c2G

def select_partition(city, dataname):
    if city == 'CHI':
        if dataname == 'poverty_rate':
            out = 'Community'
        elif dataname == 'unemployment':
            out = 'Community'
        else:
            print('err')
    elif city == 'NYC':
        if dataname == 'mean_commute':
            out = 'Community'
        elif dataname == 'poverty_rate':
            out = 'Community'
        elif dataname == 'PM25':
            out = 'UHF42'
        elif dataname == 'unemployment':
            out = 'Community'
        elif dataname == 'population':
            out = 'Community'
        elif dataname == 'recycle':
            out = 'Community'
        elif dataname == 'crime':
            out = 'Precinct'
        else:
            print('err')
    return out

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
