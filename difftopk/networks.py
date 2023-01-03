import torch
import math
import numpy as np
import diffsort


def sparse_bitonic_network(n):
    IDENTITY_MAP_FACTOR = .5
    num_blocks = math.ceil(np.log2(n))
    assert n <= 2 ** num_blocks
    network = []

    for block_idx in range(num_blocks):
        for layer_idx in range(block_idx + 1):
            m = 2 ** (block_idx - layer_idx)

            split_a, split_b = np.zeros((n, 2**num_blocks)), np.zeros((n, 2**num_blocks))
            combine_min, combine_max = np.zeros((2**num_blocks, n)), np.zeros((2**num_blocks, n))
            count = 0

            alpha_idxs = []
            one_minus_alpha_idxs = []
            remain_idxs = []
            alpha_mask = []

            for i in range(0, 2**num_blocks, 2*m):
                for j in range(m):
                    ix = i + j
                    a, b = ix, ix + m

                    # Cases to handle n \neq 2^k: The top wires are discarded and if a comparator considers them, the
                    # comparator is ignored.
                    if a >= 2**num_blocks-n and b >= 2**num_blocks-n:
                        split_a[count, a], split_b[count, b] = 1, 1
                        if (ix // 2**(block_idx + 1)) % 2 == 1:
                            a, b = b, a
                        combine_min[a, count], combine_max[b, count] = 1, 1
                        if (ix // 2**(block_idx + 1)) % 2 == 0:
                            assert b > a
                            alpha_idxs.append([a - (2 ** num_blocks - n), a - (2 ** num_blocks - n)])
                            alpha_idxs.append([b - (2 ** num_blocks - n), b - (2 ** num_blocks - n)])
                            one_minus_alpha_idxs.append([a - (2 ** num_blocks - n), b - (2 ** num_blocks - n)])
                            one_minus_alpha_idxs.append([b - (2 ** num_blocks - n), a - (2 ** num_blocks - n)])
                        else:
                            assert b < a
                            alpha_idxs.append([a - (2 ** num_blocks - n), b - (2 ** num_blocks - n)])
                            alpha_idxs.append([b - (2 ** num_blocks - n), a - (2 ** num_blocks - n)])
                            one_minus_alpha_idxs.append([a - (2 ** num_blocks - n), a - (2 ** num_blocks - n)])
                            one_minus_alpha_idxs.append([b - (2 ** num_blocks - n), b - (2 ** num_blocks - n)])
                        alpha_mask.append([count])
                        count += 1
                    elif a < 2**num_blocks-n and b < 2**num_blocks-n:
                        pass
                    elif a >= 2**num_blocks-n and b < 2**num_blocks-n:
                        split_a[count, a], split_b[count, a] = 1, 1
                        combine_min[a, count], combine_max[a, count] = IDENTITY_MAP_FACTOR, IDENTITY_MAP_FACTOR
                        remain_idxs.append([a - (2 ** num_blocks - n), a - (2 ** num_blocks - n)])
                        count += 1
                    elif a < 2**num_blocks-n and b >= 2**num_blocks-n:
                        split_a[count, b], split_b[count, b] = 1, 1
                        combine_min[b, count], combine_max[b, count] = IDENTITY_MAP_FACTOR, IDENTITY_MAP_FACTOR
                        remain_idxs.append([b - (2 ** num_blocks - n), b - (2 ** num_blocks - n)])
                        count += 1
                    else:
                        assert False

            alpha_idxs = np.array(alpha_idxs)
            one_minus_alpha_idxs = np.array(one_minus_alpha_idxs)
            remain_idxs = np.array(remain_idxs)
            alpha_mask = np.array(alpha_mask)

            split_a = split_a[:count, 2 ** num_blocks - n:]
            split_b = split_b[:count, 2 ** num_blocks - n:]
            combine_min = combine_min[2**num_blocks-n:, :count]
            combine_max = combine_max[2**num_blocks-n:, :count]
            network.append((split_a, split_b, combine_min, combine_max, alpha_idxs, one_minus_alpha_idxs, remain_idxs, alpha_mask))

    return network


def sparse_splitter_selection_network(n, k):
    def split(lanes):  # create a splitter cascade
        """Create a splitter cascade for a given set of lanes.
        lanes   list of lane indices to which to apply the splitter cascade
        returns the layers needed for the splitter cascade as a list of
                layer descriptions, each of which is a list of swaps,
                which are pairs of lane indices"""
        n = len(lanes)  # get number of lanes
        if n < 2: return []  # check for at least two lanes
        lyrcnt = int(math.ceil(math.log2(n)))  # get number of layers
        layers = [None] * lyrcnt  # create layer array
        for i in range(lyrcnt):  # traverse the layers
            k = 2 ** (lyrcnt - i - 1)  # get lane section size
            layers[i] = [(lanes[j], lanes[j + k])  # collect swaps
                         for o in range(0, n, k + k)
                         for j in range(o, min(o + k, n - k))]
        return layers  # return created swap layers

    def lyrext(layers, ext, off):  # extend layers of a selection network
        """Extend sorting or selection network layers by new (partial) layers.
        layers  existing layers of the sorting or selection network
        ext     new layers by which to extend the existing ones
        off     offset at which to apply the new (partial) layers"""
        layers.extend([[] for _ in range(off + len(ext) - len(layers))])
        for i, lyr in enumerate(ext): layers[off + i].extend(lyr)

    def splitx(rank, minrks, rsplit, layers, nxtlyr):
        """Split a subset of lanes that are identified by a ge value.
        rank    lane selection value (minimum lane rank)
        minrks  current minimum ranks per lane
        rsplit  list of minimum ranks resulting from a splitter cascade
        layers  previously constructed layers
        nxtlyr  smallest layer index, each lane may be operated on again"""
        sel = [l for l, r in enumerate(minrks) if r == rank]
        if len(sel) < 2: return 0  # get and check lane set
        ext = split(sel)  # split the lane set
        off = max(nxtlyr[l] for l in sel)
        lyrext(layers, ext, off)  # add split cascade layers for lane set
        off += len(ext)  # get next layer after split cascade
        for j, l in enumerate(sel):  # traverse the affected lanes
            minrks[l] += rsplit[j]  # update minimum lane ranks
            nxtlyr[l] = off  # and next layer indices
        return 1  # return that layers were added

    def topk(n, k=5):  # create top-k selection network
        """Create a network for selecting the top k elements from n elements.
        n       number of elements to select from
        k       number of top elements to select
        returns the selection network as a list of layer descriptions,
                each of which is a list of swaps, which are pairs of lane indices"""
        lanes = list(range(n))  # create list of lane indices
        rsplit = [2 ** y - 1 for y in [bin(i).count('1') for i in lanes]]
        minrks = [0] * n  # minimum ranks resulting from a splitter cascade
        nxtlyr = [0] * n  # start minimum ranks and next layers to zero
        layers = []  # initialize network layers
        rkend = -1  # and end of rank range
        addlyr = +1  # default: execute loop
        while addlyr:  # while layers were added
            addlyr = 0  # clear added layer flag
            for i in range(k - 1, rkend, -1):
                addlyr |= splitx(i, minrks, rsplit, layers, nxtlyr)
            rkend += 1  # each loop increases the minimum rank
        return layers  # return the constructed layers

    layers = topk(n, k)
    layers = [[(n - b - 1, n - a - 1) for a, b in lyr] for lyr in layers]

    IDENTITY_MAP_FACTOR = .5
    network = []

    for layer_idx, layer in enumerate(layers):
        split_a, split_b = np.zeros((n-len(layer), n)), np.zeros((n-len(layer), n))
        combine_min, combine_max = np.zeros((n, n-len(layer))), np.zeros((n, n-len(layer)))
        count = 0

        alpha_idxs = []
        one_minus_alpha_idxs = []
        remain_idxs = []
        alpha_mask = []

        for a, b in layer:
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = 1, 1
            alpha_idxs.append([a, a])
            alpha_idxs.append([b, b])
            one_minus_alpha_idxs.append([a, b])
            one_minus_alpha_idxs.append([b, a])
            alpha_mask.append([count])
            count += 1

        for a in set(range(n))-set([x for s in layer for x in s]):
            split_a[count, a], split_b[count, a] = 1, 1
            combine_min[a, count], combine_max[a, count] = IDENTITY_MAP_FACTOR, IDENTITY_MAP_FACTOR
            remain_idxs.append([a, a])
            count += 1

        alpha_idxs = np.array(alpha_idxs)
        one_minus_alpha_idxs = np.array(one_minus_alpha_idxs)
        remain_idxs = np.array(remain_idxs)
        alpha_mask = np.array(alpha_mask)

        assert count == n-len(layer), (count, n, len(layer))
        network.append((split_a, split_b, combine_min, combine_max, alpha_idxs, one_minus_alpha_idxs, remain_idxs, alpha_mask))

    print(len(network), 'layers', n, k)

    return network


def get_sparse_net(net):
    s_net = []
    for l in net:
        ss = []
        for m in l[:4]:
            m = m.T
            m_i = torch.nonzero(m)
            m_v = m[m != 0]
            s = torch.sparse_coo_tensor(m_i.T, m_v, m.shape).coalesce()
            ss.append(s)
        for m in l[4:]:
            ss.append(m)
        s_net.append(ss)
    return s_net


def get_sorting_network(type, n, device, k=None):
    def matrix_to_torch(m):
        return [[torch.from_numpy(matrix).float().to(device) for matrix in matrix_set] for matrix_set in m]

    if type == 'bitonic':
        return matrix_to_torch(diffsort.networks.bitonic_network(n))
    elif type == 'sparse_bitonic':
        return get_sparse_net(matrix_to_torch(sparse_bitonic_network(n)))
    elif type == 'sparse_splitter_selection':
        return get_sparse_net(matrix_to_torch(sparse_splitter_selection_network(n, k)))
    # elif type == 'sparse_bitonic_topk':
    #     return get_sparse_net(matrix_to_torch(sparse_bitonic_network_topk(n, k)))
    elif type == 'odd_even':
        return matrix_to_torch(diffsort.networks.odd_even_network(n))
    else:
        raise NotImplementedError('Sorting network `{}` unknown.'.format(type))
