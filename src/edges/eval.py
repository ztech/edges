import torch
from skimage import morphology
from match import match_edge_maps

eps = 2.2204e-16  # eps in MATLAB


def correspond_pixels(
    img1: torch.Tensor, img2: torch.Tensor, maxDist=0.0075, outlierCost=100
):
    # check arguments
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must be the same size")
    if maxDist < 0:
        raise ValueError("maxDist must be >= 0")
    if outlierCost <= 1:
        raise ValueError("outlierCost must be >1")

    # do the computation
    idiag = torch.sqrt(torch.tensor([img1.shape[0] ** 2 + img1.shape[1] ** 2]))
    oc = outlierCost * maxDist * idiag
    m1, m2, cost = match_edge_maps(img1, img2, maxDist * idiag, oc)

    return m1, m2, cost, oc


def eval(
    img: torch.Tensor,
    gt: list[torch.Tensor],
    thrs: int | list[float] = 99,
    maxDist=0.0075,
    thin=1,
):
    if isinstance(thrs, int):
        n_thrs = thrs
        thrs = torch.linspace(1 / (n_thrs + 1), 1 - 1 / (n_thrs + 1), n_thrs)
    else:
        n_thrs = len(thrs)
        thrs = torch.tensor(thrs)

    # evaluate edge result at each threshold
    cntR = torch.zeros(n_thrs)
    sumR = torch.zeros(n_thrs)
    cntP = torch.zeros(n_thrs)
    sumP = torch.zeros(n_thrs)
    visual = torch.zeros((img.shape[0], img.shape[1], 3, n_thrs))
    for k in range(n_thrs):
        # threshhold and thin E
        img_thrs = img >= max(eps, thrs[k])
        if thin:
            img_thrs = morphology.thin(img_thrs, thin)
        # compare to each ground truth in turn and accumualte
        matchE = torch.zeros_like(img)
        matchG = torch.zeros_like(img)
        allG = torch.zeros_like(img)

        for g in range(len(gt)):
            matchE1, matchG1 = correspond_pixels(img_thrs, gt[g], maxDist)
            matchE = matchE | matchE1 > 0
            matchG = matchG + int(matchG1 > 0)
            allG = allG + gt[g]
        # compute recall (summed over each gt image)
        cntR[k] = matchG.sum()
        sumR[k] = allG.sum()
        # compute precision (edges can match any gt image)
        cntP[k] = torch.count_nonzero(matchE)
        sumP[k] = torch.count_nonzero(img_thrs)
        # optinally create visualization of matches
        cs = torch.tensor([[0, -1, -1], [-1, -0.3, -1], [-0.3, -0.2, 0]])
        FP = img_thrs - matchE
        TP = matchE
        FN = (allG - matchG) / len(gt)
        for g in range(3):
            visual[:, :, g, k] = torch.clamp(
                1 + FN * cs[0, g] + TP * cs[1, g] + FP * cs[2, g], 0
            )
        visual[1:, :, :, k] = torch.min(visual[1:, :, :, k], visual[:-1, :, :, k])
        visual[:, 1:, :, k] = torch.min(visual[:, 1:, :, k], visual[:, :-1, :, k])

    return thrs, cntR, sumR, cntP, sumP, visual


# % get additional parameters
# dfs={ 'out','', 'thrs',99, 'maxDist',.0075, 'thin',1 };
# [out,thrs,maxDist,thin] = getPrmDflt(varargin,dfs,1);
# if(any(mod(thrs,1)>0)), K=length(thrs); thrs=thrs(:); else
#   K=thrs; thrs=linspace(1/(K+1),1-1/(K+1),K)'; end

# % load edges (E) and ground truth (G)
# if(all(ischar(E))), E=double(imread(E))/255; end
# G=load(G); G=G.groundTruth; n=length(G);
# for g=1:n, G{g}=double(G{g}.Boundaries); end

# % evaluate edge result at each threshold
# Z=zeros(K,1); cntR=Z; sumR=Z; cntP=Z; sumP=Z;
# if(nargout>=6), V=zeros([size(E) 3 K]); end
# for k = 1:K
#   % threshhold and thin E
#   E1 = double(E>=max(eps,thrs(k)));
#   if(thin), E1=double(bwmorph(E1,'thin',inf)); end
#   % compare to each ground truth in turn and accumualte
#   Z=zeros(size(E)); matchE=Z; matchG=Z; allG=Z;
#   for g = 1:n
#     [matchE1,matchG1] = correspondPixels(E1,G{g},maxDist);
#     matchE = matchE | matchE1>0;
#     matchG = matchG + double(matchG1>0);
#     allG = allG + G{g};
#   end
#   % compute recall (summed over each gt image)
#   cntR(k) = sum(matchG(:)); sumR(k) = sum(allG(:));
#   % compute precision (edges can match any gt image)
#   cntP(k) = nnz(matchE); sumP(k) = nnz(E1);
#   % optinally create visualization of matches
#   if(nargout<6), continue; end; cs=[1 0 0; 0 .7 0; .7 .8 1]; cs=cs-1;
#   FP=E1-matchE; TP=matchE; FN=(allG-matchG)/n;
#   for g=1:3, V(:,:,g,k)=max(0,1+FN*cs(1,g)+TP*cs(2,g)+FP*cs(3,g)); end
#   V(:,2:end,:,k) = min(V(:,2:end,:,k),V(:,1:end-1,:,k));
#   V(2:end,:,:,k) = min(V(2:end,:,:,k),V(1:end-1,:,:,k));
# end

# % if output file specified write results to disk
# if(isempty(out)), return; end; fid=fopen(out,'w'); assert(fid~=1);
# fprintf(fid,'%10g %10g %10g %10g %10g\n',[thrs cntR sumR cntP sumP]');
# fclose(fid);
