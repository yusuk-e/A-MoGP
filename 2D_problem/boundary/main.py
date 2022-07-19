# Aggregated Multi-output Gaussian Processes | 2022
# Yusuke Tanaka

import pdb
import glob
import numpy as np
import pandas as pd
import argparse
import itertools
import os, sys
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import calcu
import utils

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--g_scale', default=0.5, type=float, help='grid_size (ratio to 1km)')
    return parser.parse_args()

def make_grid(df, alpha):
    def extract():
        new = []
        max_row = np.max(cr_ids[:,1])
        for id in grid_ids:            
            new.append([cr_ids[id,0], max_row-cr_ids[id,1]])
        return new
    xmin, xmax, ymin, ymax = utils.bound(df)
    grid, cr_ids = utils.grid(xmin, xmax, ymin, ymax, alpha)
    xs, ys = np.unique(grid[:,0]), np.unique(grid[:,1])
    grid, grid_ids = utils.join_grid(df, grid)
    cr_ids = extract()
    return np.array(grid), {'xs':xs, 'ys':ys, 'cr_ids':cr_ids}

def grid_main():
    i_dirs = ['CHI/input/Precinct', 'NYC/input/Borough']
    for i_dir in i_dirs:
        i_file = glob.glob(i_dir + '/*.shp')[0]
        df = utils.shapefile_read(i_file)
        grid, cr_ids = make_grid(df, alpha)
        o_dir = i_dir.split('/')[0] + '/output/' + str(alpha)
        utils.check_dir(o_dir)
        o_file = o_dir + '/grid'
        np.save(o_file, grid)
        o_file = o_dir + '/cr_ids.pkl'
        utils.pkl_write(o_file, cr_ids)

def make_r2grid_ids(grid, df, p, r):
    G = {}
    for i, s in df.iterrows():
        r_id = str(int(s[r]))
        polygon = utils.unify_polygons(s)
        sub_grid_ids, sub_grid = utils.extract_grid(grid, polygon)
        G[r_id] = sub_grid_ids
    return G

def grid_ids():
    pr2grid_ids = {}
    i_file = city + '/output/' + str(alpha) + '/grid.npy'
    grid = np.load(i_file)
    for partition in H[city]:
        i_file = glob.glob(city + '/input/' + partition + '/*.shp')[0]
        df = utils.shapefile_read(i_file)
        df = utils.epsg_conv(df, 4326)
        r_name = utils.define_r_name(city, partition)
        r2grid_ids = make_r2grid_ids(grid, df, partition, r_name)
        pr2grid_ids[partition] = r2grid_ids
    o_dir = city + '/output/' + str(alpha)
    utils.check_dir(o_dir)
    o_file = o_dir + '/pr2grid_ids.pkl'
    utils.pkl_write(o_file, pr2grid_ids)

def vis():
    i_file = city + '/output/' + str(alpha) + '/grid.npy'
    grid = np.load(i_file)
    for partition in H[city]:
        i_file = glob.glob(city + '/input/' + partition + '/*.shp')[0]
        df = utils.shapefile_read(i_file)
        df = utils.epsg_conv(df, 3857)

        o_dir = city + '/output/' + str(alpha) + '/' + partition
        utils.check_dir(o_dir)
        utils.boundary_plot(o_dir + '/boundary.png', df)
        utils.point_plot(o_dir + '/grid.png', df, grid)

def conv_euclid(grid):
    x = 0.010966404715491394#1km
    y = 0.0090133729745762#1km
    grid[:,0] = grid[:,0] / x
    grid[:,1] = grid[:,1] / y
    return grid

def norm():
    i_file = city + '/output/' + str(alpha) + '/grid.npy'
    grid = np.load(i_file)
    grid = conv_euclid(grid)
    norm = calcu.make_norm(grid)
    o_file = city + '/output/' + str(alpha) + '/norm'
    np.save(o_file, norm)

def rr_count():
    i_file = city + '/output/' + str(alpha) + '/grid.npy'
    grid = np.load(i_file)
    grid = conv_euclid(grid)
    i_file = city + '/output/' + str(alpha) + '/norm.npy'
    norm = np.load(i_file)
    i_file = city + '/output/' + str(alpha) + '/pr2grid_ids.pkl'
    pr2grid_ids = utils.pkl_read(i_file)
    lines = []
    partitions = H[city]
    for p1, p2 in list(itertools.combinations_with_replacement(partitions, 2)):
        print(p1,p2)
        for r1 in pr2grid_ids[p1]:
            gs1 = np.array(pr2grid_ids[p1][r1]).astype(int)
            for r2 in pr2grid_ids[p2]:
                gs2 = np.array(pr2grid_ids[p2][r2]).astype(int)
                count = calcu.make_rr_count(grid, gs1, gs2, norm)
                ids = np.where(count > 0)[0]
                values = count[ids]
                lines.append([p1,p2,r1,r2,ids,values])
                if p1 != p2:
                    lines.append([p2,p1,r2,r1,ids,values])
    rr_count = pd.DataFrame(lines, columns=['p1','p2','r1','r2','ids','counts'])
    o_dir = city + '/output/' + str(alpha)
    utils.check_dir(o_dir)
    o_file = o_dir + '/rr_count.df'
    utils.df_write(o_file, rr_count)

def rp_count():
    i_file = city + '/output/' + str(alpha) + '/grid.npy'
    grid = np.load(i_file)
    grid = conv_euclid(grid)
    i_file = city + '/output/' + str(alpha) + '/norm.npy'
    norm = np.load(i_file)
    i_file = city + '/output/' + str(alpha) + '/pr2grid_ids.pkl'
    pr2grid_ids = utils.pkl_read(i_file)
    lines = []
    for g_id in range(len(grid)):
        g = np.array(grid[g_id])
        partitions = H[city]
        for p in partitions:
            for r in pr2grid_ids[p]:
                gs = np.array(pr2grid_ids[p][r]).astype(int)
                count = calcu.make_rp_count(grid, g, gs, norm)
                ids = np.where(count > 0)[0]
                values = count[ids]
                lines.append([p,r,g_id,ids,values])
    rp_count = pd.DataFrame(lines, columns=['p','r','g_id','ids','counts'])
    o_dir = city + '/output/' + str(alpha)
    utils.check_dir(o_dir)
    o_file = o_dir + '/rp_count.df'
    utils.df_write(o_file, rp_count)


if __name__ == '__main__':

    args = get_args()
    alpha = args.g_scale # Ratio to 1 km
    H = {'NYC': ['Borough', 'Community', 'Precinct', 'UHF42'],
         'CHI': ['Side', 'Community', 'Precinct']}

    grid_main()
    for city in H:
        grid_ids()
        vis()
        norm()
        rr_count()#region-region
        rp_count()#region-grid
