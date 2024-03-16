import torch
from torch.linalg import det

def cof1(M,index):
    zs = M[:index[0]-1,:index[1]-1]
    ys = M[:index[0]-1,index[1]:]
    zx = M[index[0]:,:index[1]-1]
    yx = M[index[0]:,index[1]:]
    s = torch.cat((zs,ys),axis=1)
    x = torch.cat((zx,yx),axis=1)
    return det(torch.cat((s,x),axis=0))
 
def alcof(M,index):
    return pow(-1,index[0]+index[1])*cof1(M,index)
 
def adj(M):
    result = torch.zeros((M.shape[0],M.shape[1]))
    for i in range(1,M.shape[0]+1):
        for j in range(1,M.shape[1]+1):
            result[j-1][i-1] = alcof(M,[i,j])
    return result
 
# def invmat(M):
#     return 1.0/det(M)*adj(M)

def invmat(M):
    inverse = M
    for i in range (M.shape[0]):
        m = M[i,:,:].squeeze(0)
        mi = 1.0/det(m)*adj(m)
        inverse[i,:,:] = mi
        # mn = torch.cat(mn, mi)
        # mn.append(mi)
    return inverse # 1.0/det(M)*adj(M)

# M = torch.FloatTensor([[1,2,-1],[2,3,4],[3,1,2]])
# print(invmat(M))
# print(torch.inverse(M))