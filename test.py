import imread
import numpy as np
import h5py
from skimage.segmentation import quickshift, find_boundaries
from scipy.ndimage.filters import minimum_filter, maximum_filter
from fast64counter import ValueCountInt64, WeightedCountInt64
from heapq import heapify, heappush, heappop

import os
import sys
import time
import gc
import operator

from collections import defaultdict

import numpy as np
from scipy.ndimage.measurements import label as ndimage_label
import h5py

import cplex

# import overlaps


##################################################
# Parameters
##################################################
size_compensation_factor = 0.9
chunksize = 128  # chunk size in the HDF5

# NB - both these functions should accept array arguments
# weights for segments
def segment_worth(area):
    return area ** size_compensation_factor
# weights for links
def link_worth(area1, area2, area_overlap):
    min_area = np.minimum(area1, area2)
    max_fraction = area_overlap / np.maximum(area1, area2)
    return max_fraction * (min_area ** size_compensation_factor)

def build_tree(im):
    # im = boundary probability

    print "QS"
    segs = quickshift(im, sigma=2.0, return_tree=False, convert2lab=False, max_dist=10)
    orig_segs = segs.copy()
    print "done"

    # Find all neighboring regions and the values between them
    plus = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]).astype(np.bool)
    lo = minimum_filter(segs, footprint=plus).astype(np.int32)
    hi = maximum_filter(segs, footprint=plus).astype(np.int32)

    counter = ValueCountInt64()
    counter.add_values_pair32(lo.ravel(), hi.ravel())
    loc, hic, edge_counts = counter.get_counts_pair32()
    weighter = WeightedCountInt64()
    weighter.add_values_pair32(lo.ravel(), hi.ravel(), im.astype(np.float32).ravel())
    low, hiw, edge_weights = weighter.get_weights_pair32()

    max_regions = 2 * segs.max() + 1 #  number of regions is max() + 1
    counts = np.zeros((max_regions, max_regions), dtype=np.uint64)
    weights = np.zeros((max_regions, max_regions), dtype=float)
    counts[loc, hic] = edge_counts
    weights[low, hiw] = edge_weights
    # zero diagonal
    counts[np.arange(max_regions), np.arange(max_regions)] = 0

    next_region = segs.max() + 1
    parents = {}
    # set up heap
    heap = [(weights[l, h] / counts[l, h], l, h) for l, h in zip(*np.nonzero(counts))]
    heapify(heap)

    # successively merge regions
    while heap:
        w, lo, hi = heappop(heap)
        if (lo in parents) or (hi in parents):
            continue
        print next_region, max_regions
        parents[lo] = next_region
        parents[hi] = next_region
        counts[next_region, :] = counts[lo, :] + counts[hi, :]
        weights[next_region, :] = weights[lo, :] + weights[hi, :]
        counts[:, next_region] = counts[next_region, :]
        weights[:, next_region] = weights[next_region, :]
        for idx in range(next_region):
            if idx in parents:
                continue
            if counts[idx, next_region] > 0 and (idx not in parents):
                heappush(heap, (weights[idx, next_region] / counts[idx, next_region], idx, next_region))
        segs[segs == lo] = next_region
        segs[segs == hi] = next_region
        next_region += 1
    print "done"
    return orig_segs, parents


def load_probs(f):
    return np.clip(1.0 - h5py.File(f)['probabilities'][...], 0, 1).astype(np.float32)[:512, :512]

def offset_segmentation(seg, offset):
    labels, parents = seg
    return labels + offset, {c + offset : p + offset for c, p in parents.iteritems()}

def exclusions(tree, node):
    while node in tree:
        yield node
        node = tree[node]
    yield node

if __name__ == '__main__':
    segmentations = [build_tree(load_probs(f)) for f in sys.argv[1:]]

    # make each label unique
    for idx, seg in enumerate(segmentations):
        if idx > 0:
            prev_tree = segmentations[idx - 1][1]
            offset = max(prev_tree.values()) + 1
            segmentations[idx] = offset_segmentation(seg, offset)

    # build unified parent and area tables
    num_regions = max(segmentations[-1][1].values()) + 1
    all_areas = np.zeros(num_regions, np.int64)
    all_parents = -1 * np.ones(num_regions, np.int64)
    for regions, parents in segmentations:
        areacounter = ValueCountInt64()
        areacounter.add_values_32(regions.astype(np.int32).ravel())
        keys, areas = areacounter.get_counts()
        all_areas[keys] = areas
        all_parents[parents.keys()] = parents.values()

    for idx in range(num_regions):
        if all_parents[idx] >= 0:
            all_areas[all_parents[idx]] += all_areas[idx]segmentations

    print "Num regions:", num_regions

    model = cplex.Cplex()
    model.variables.add(obj = segment_worth(all_areas),
                        lb = [0] * num_segments,
                        ub = [1] * num_segments,
                        types = ["B"] * num_segments)

    for idx in range(num_segments):
        exclusions = [idx]
        p = all_parents[idx]
        while p != -1:
            exclusions.append(p)
            p = all_parents[p]
        if len(exclusions) > 1:
            model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = [int(i) for i in excls],
                                                                      val = [1] * len(excls))],
                                         senses = "L",
                                         rhs = [1])
    print "added exclusions"

    # Add slice-to-slice links and link constraints.
    links_up = defaultdict(list)
    links_down = defaultdict(list)
    for ((regions1, tree1), (regions2, tree2)) in zip(segmentations, segmentations[1:]):
        for r1, r2, weight in find_tree_overlaps(regions1, tree1, regions2, tree2, link_worth):
            linkidx = model.variables.get_num()
            model.variables.add(obj = [weight], lb = [0], ub = [1], types = "B", names = ['link_%d_%d' % (r1, r2)])
            links_up[r1].append(linkidx)
            linkd_down[r2].append(linkidx)
    for segidx, linklist in links_up.iteritems():
        model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = [int(segidx)] + linklist,
                                                                  val = [1] + [-1] * len(linklist))],
                                     senses = "G",
                                     rhs = [0])

    for segidx, linklist in links_down.iteritems():
        model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = [int(segidx)] + linklist,
                                                                  val = [1] + [-1] * len(linklist))],
                                     senses = "G",
                                     rhs = [0])


    model.objective.set_sense(model.objective.sense.maximize)
    model.parameters.threads.set(1)
    model.parameters.mip.tolerances.mipgap.set(0.02)  # 2% tolerance
    # model.parameters.emphasis.memory.set(1)  # doesn't seem to help
    model.parameters.emphasis.mip.set(1)
