import torch
from ortools.graph.python import linear_sum_assignment
import numpy as np

# CSA code needs integer weights.  Use this multiplier to convert
# floating-point weights to integers.
# multiplier = 100
#
# The degree of outlier connections.
# degree = 6

def match_edge_maps(bmap1: torch.Tensor, bmap2: torch.Tensor, maxDist: float, outlierCost: float, multiplier=100, degree=6):
    width, height = bmap1.shape[1], bmap1.shape[0]
    class Edge:
        def __init__(self):
            self.i = 0
            self.j = 0
            self.w = 0.0

    class Pixel:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # Initialize match1 and match2 arrays
    match1 = [[Pixel(-1, -1) for _ in range(width)] for _ in range(height)]
    match2 = [[Pixel(-1, -1) for _ in range(width)] for _ in range(height)]

    # Radius of search window
    r = int(maxDist)

    # Figure out which nodes are matchable
    matchable1 = [[False for _ in range(width)] for _ in range(height)]
    matchable2 = [[False for _ in range(width)] for _ in range(height)]
    for y1 in range(height):
        for x1 in range(width):
            if not bmap1[y1][x1]:
                continue
            for v in range(-r, r+1):
                for u in range(-r, r+1):
                    d2 = u*u + v*v
                    if d2 > maxDist*maxDist:
                        continue
                    x2 = x1 + u
                    y2 = y1 + v
                    if x2 < 0 or x2 >= width:
                        continue
                    if y2 < 0 or y2 >= height:
                        continue
                    if not bmap2[y2][x2]:
                        continue
                    matchable1[x1][y1] = True
                    matchable2[x2][y2] = True

    # Count the number of nodes on each side of the match
    n1, n2 = 0, 0
    nodeToPix1 = []
    nodeToPix2 = []
    pixToNode1 = [[-1 for _ in range(width)] for _ in range(height)]
    pixToNode2 = [[-1 for _ in range(width)] for _ in range(height)]
    for x in range(width):
        for y in range(height):
            pixToNode1[x][y] = -1
            pixToNode2[x][y] = -1
            pix = Pixel(x, y)
            if matchable1[x][y]:
                pixToNode1[x][y] = n1
                nodeToPix1.append(pix)
                n1 += 1
            if matchable2[x][y]:
                pixToNode2[x][y] = n2
                nodeToPix2.append(pix)
                n2 += 1

    # cost matrix size
    n = n1+n2
    costs: np.ndarray = np.zeros(n, n)

    # Weight of outlier connections
    ow = int(outlierCost * multiplier)

    # Scratch array for outlier edges
    outliers = [0] * dmax

    # Construct the list of edges between pixels within maxDist
    real_edges = []
    for x1 in range(width):
        for y1 in range(height):
            if not matchable1[x1][y1]:
                continue
            for u in range(-r, r+1):
                for v in range(-r, r+1):
                    d2 = u*u + v*v
                    if d2 > maxDist*maxDist:
                        continue
                    x2 = x1 + u
                    y2 = y1 + v
                    if x2 < 0 or x2 >= width:
                        continue
                    if y2 < 0 or y2 >= height:
                        continue
                    if not matchable2[x2][y2]:
                        continue
                    i = pixToNode1[x1][y1]
                    j = pixToNode2[x2][y2]
                    w = (d2)**0.5
                    assert i >= 0 and i < n1
                    assert j >= 0 and j < n2
                    assert w < outlierCost
                    real_edges.append([i, j, int(round(w * multiplier))])
                    # e = Edge()
                    # e.i = pixToNode1[x1][y1]
                    # e.j = pixToNode2[x2][y2]
                    # e.w = (d2)**0.5
                    # assert e.i >= 0 and e.i < n1
                    # assert e.j >= 0 and e.j < n2
                    # assert e.w < outlierCost
                    # edges.append(e)

    # The cardinality of the match is n
    n = n1 + n2
    nmin = min(n1, n2)
    nmax = max(n1, n2)

    # Compute the degree of various outlier connections
    d1 = max(0, min(degree, n1-1))  # from map1
    d2 = max(0, min(degree, n2-1))  # from map2
    d3 = min(degree, min(n1, n2))  # between outliers
    dmax = max(d1, d2, d3)

    assert n1 == 0 or (d1 >= 0 and d1 < n1)
    assert n2 == 0 or (d2 >= 0 and d2 < n2)
    assert d3 >= 0 and d3 <= nmin

    # Count the number of edges
    m = 0
    m += len(edges)  # real connections
    m += d1 * n1  # outlier connections
    m += d2 * n2  # outlier connections
    m += d3 * nmax  # outlier-outlier connections
    m += n  # high-cost perfect match overlay

    # If the graph is empty, then there's nothing to do
    if m == 0:
        return 0
    
    # Construct the input graph for the assignment problem
    edges = np.zeros((m, 3), dtype=np.int32)
    # igraph = [[0] * 3 for _ in range(m)]
    # real edges
    count = len(real_edges)
    edges[:count] = real_edges
    # for a in range(len(edges)):
    #     i = edges[a].i
    #     j = edges[a].j
    #     assert i >= 0 and i < n1
    #     assert j >= 0 and j < n2
    #     igraph[count][0] = i
    #     igraph[count][1] = j
    #     igraph[count][2] = int(round(edges[a].w * multiplier))
    #     count += 1

    # outliers edges for map1, exclude diagonal
    for i in range(n1):
        outliers = torch.sort(torch.randperm(n1-1)[:d1])[0] # TODO: replace with np.random.choice, need benchmark for final decision
        for a in range(d1):
            j = outliers[a]
            if j >= i:
                j += 1
            assert i != j
            assert j >= 0 and j < n1
            edges[count] = [i, n2+j, ow]
            # igraph[count][0] = i
            # igraph[count][1] = n2 + j
            # igraph[count][2] = ow
            count += 1
    # outliers edges for map2, exclude diagonal
    for j in range(n2):
        outliers = torch.sort(torch.randperm(n2-1)[:d2])[0]
        for a in range(d2):
            i = outliers[a]
            if i >= j:
                i += 1
            assert i != j
            assert i >= 0 and i < n2
            edges[count] = [n1+i, j, ow]
            # igraph[count][0] = n1 + i
            # igraph[count][1] = j
            # igraph[count][2] = ow
            count += 1
    # outlier-to-outlier edges
    for i in range(nmax):
        outliers = torch.sort(torch.randperm(nmin)[:d3])[0]
        for a in range(d3):
            j = outliers[a]
            assert j >= 0 and j < nmin
            if n1 < n2:
                assert i >= 0 and i < n2
                assert j >= 0 and j < n1
                edges[count] = [n1+i, n2+j, ow]
                # igraph[count][0] = n1 + i
                # igraph[count][1] = n2 + j
            else:
                assert i >= 0 and i < n1
                assert j >= 0 and j < n2
                costs[n1+j, n2+i] = ow
                edges[count] = [n1+j, n2+i, ow]
                # igraph[count][0] = n1 + j
                # igraph[count][1] = n2 + i
            # igraph[count][2] = ow
            count += 1
    # perfect match overlay (diagonal)
    costs[count:count+n1] = [[i, n2+i, ow*multiplier] for i in range(n1)]
    count += n1
    costs[count:count+n2] = [[n1+i, i, ow*multiplier] for i in range(n2)]
    # for i in range(n1):
    #     igraph[count][0] = i
    #     igraph[count][1] = n2 + i
    #     igraph[count][2] = ow * multiplier
    #     count += 1
    # for i in range(n2):
    #     igraph[count][0] = n1 + i
    #     igraph[count][1] = i
    #     igraph[count][2] = ow * multiplier
    #     count += 1
    assert count == m
    edges[:, 1] += n

    # Check all the edges, and set the values up for CSA
    # for i in range(m):
    #     assert igraph[i][0] >= 0 and igraph[i][0] < n
    #     assert igraph[i][1] >= 0 and igraph[i][1] < n
    #     igraph[i][0] += 1
    #     igraph[i][1] += 1 + n

    # Solve the assignment problem
    assignment = linear_sum_assignment.SimpleLinearSumAssignment()
    assignment.add_arcs_with_cost(edges[:, 0], edges[:, 1], edges[:, 2])
    status = assignment.solve()
    if status != assignment.OPTIMAL:
        raise ValueError("No optimal solution found")
    # CSA implementation is not provided, you need to find a suitable library or implement it yourself

    # Check the solution
    raise NotImplementedError("CSA implementation is not ready yet.")
    matched = np.zeros((assignment.num_arcs(), ))
    # Count the number of high-cost edges from the perfect match overlay that were used in the match
    overlayCount = 0
    for a in range(assignment.num_arcs()):
        i = assignment.left_node(a)
        j = assignment.right_node(a)
        c = assignment.cost(a)
        assert i >= 0 and i < n
        assert j >= 0 and j < n
        assert c >= 0
        # edge from high-cost perfect match overlay
        if c == ow * multiplier:
            overlayCount += 1
        # skip outlier edges
        if i >= n1:
            continue
        if j >= n2:
            continue
        # for edges between real nodes, check the edge weight
        pix1 = nodeToPix1[i]
        pix2 = nodeToPix2[j]
        dx = pix1.x - pix2.x
        dy = pix1.y - pix2.y
        w = int(round((dx*dx + dy*dy)**0.5 * multiplier))
        assert w == c

    # Print a warning if any of the edges from the perfect match overlay were used
    if overlayCount > 5:
        print(f"WARNING: The match includes {overlayCount} outlier(s) from the perfect match overlay.")

    # Compute match arrays
    for a in range(n):
        i = assignment.left_node(a)
        j = assignment.right_node(a)
        if i < n1 and j < n2:
            pix1 = nodeToPix1[i]
            pix2 = nodeToPix2[j]
            match1[pix1.x][pix1.y] = pix2
            match2[pix2.x][pix2.y] = pix1
    m1 = [[0 for _ in range(width)] for _ in range(height)]
    m2 = [[0 for _ in range(width)] for _ in range(height)]
    for x in range(width):
        for y in range(height):
            if bmap1[y][x]:
                if match1[x][y] != Pixel(-1, -1):
                    m1[y][x] = match1[x][y].x * height + match1[x][y].y + 1
            if bmap2[y][x]:
                if match2[x][y] != Pixel(-1, -1):
                    m2[y][x] = match2[x][y].x * height + match2[x][y].y + 1

    # Compute the match cost
    cost = 0
    for x in range(width):
        for y in range(height):
            if bmap1[y][x]:
                if match1[x][y] == Pixel(-1, -1):
                    cost += outlierCost
                else:
                    dx = x - match1[x][y].x
                    dy = y - match1[x][y].y
                    cost += 0.5 * ((dx*dx + dy*dy)**0.5)
            if bmap2[y][x]:
                if match2[x][y] == Pixel(-1, -1):
                    cost += outlierCost
                else:
                    dx = x - match2[x][y].x
                    dy = y - match2[x][y].y
                    cost += 0.5 * ((dx*dx + dy*dy)**0.5)

    # Return the match cost
    return m1, m2, cost

# # Call the match function with the appropriate arguments
# result = match(width, height, maxDist, bmap1, bmap2, outlierCost, degree, multiplier)