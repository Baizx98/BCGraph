import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.datasets import Reddit,Planetoid
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import NeighborSampler
import time
from collections import deque
import collections
import quiver
from quiver.utils import  CSRTopo
from quiver.pyg import GraphSageSampler

import numpy as np
import math
import matplotlib.pyplot as plt
import powerlaw

'''
dataset = Reddit('/data/Reddit/')
data = dataset[0]

train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=1024,
                                           shuffle=True,
                                           drop_last=True) # Quiver
csr_topo = quiver.CSRTopo(data.edge_index) # Quiver
quiver_sampler = GraphSageSampler(csr_topo, sizes=[25, 10], device=0, mode='GPU') # Quiver


'''
root = "/data/wyj/products/"
dataset = PygNodePropPredDataset('ogbn-products', root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]
train_idx = split_idx['train']


train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=1024,
                                           shuffle=True,
                                           drop_last=True)

csr_topo = quiver.CSRTopo(data.edge_index)
quiver_sampler = GraphSageSampler(csr_topo, sizes=[15, 10, 5], device=0, mode='GPU')



'''      
dataset='Cora'
path = '/data/wyj/'
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

train_idx = torch.arange(data.num_nodes, dtype=torch.long)
train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=256,
                                           shuffle=True,
                                           drop_last=True)

csr_topo = quiver.CSRTopo(data.edge_index)
quiver_sampler = GraphSageSampler(csr_topo, sizes=[10, 10], device=0)
'''

class DLinkedNode:
    def __init__(self, key=0):
        self.key = key
        self.prev = None
        self.next = None
class FIFOCache:
    def __init__(self, capacity: int):
        self.cache = dict()
        # 使用伪头部和伪尾部节点    
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        return key

    def put(self, key: int) -> None:
        
        node = DLinkedNode(key)
            # 添加进哈希表
        self.cache[key] = node
        self.addToTail(node)
        self.size += 1
        if self.size > self.capacity:
                # 如果超出容量，删除双向链表的尾部节点
            removed = self.removeHead()
                # 删除哈希表中对应的项
            self.cache.pop(removed.key)
            self.size -= 1

    def addToTail(self, node):
        self.tail.prev.next=node
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev=node
    
    def removeHead(self):
        node = self.head.next
        self.head.next=node.next
        node.next.prev=self.head
        return node
    
class LRUCache:

    def __init__(self, capacity: int):
        self.cache = dict()
        # 使用伪头部和伪尾部节点    
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def compute(self,key:int)->int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        return node.key
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # 如果 key 存在，先通过哈希表定位，再移到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.key

    def put(self, key: int) -> None:
        if key not in self.cache:
            # 如果 key 不存在，创建一个新的节点
            node = DLinkedNode(key)
            # 添加进哈希表
            self.cache[key] = node
            # 添加至双向链表的头部
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                # 如果超出容量，删除双向链表的尾部节点
                removed = self.removeTail()
                # 删除哈希表中对应的项
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            # 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node = self.cache[key]
            self.moveToHead(node)
    
    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node
    
class Node:
    def __init__(self, key, val, pre=None, nex=None, freq=0):
        self.pre = pre
        self.nex = nex
        self.freq = freq
        self.val = val
        self.key = key
        
    def insert(self, nex):
        nex.pre = self
        nex.nex = self.nex
        self.nex.pre = nex
        self.nex = nex
    
def create_linked_list():
    head = Node(0, 0)
    tail = Node(0, 0)
    head.nex = tail
    tail.pre = head
    return (head, tail)

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.minFreq = 0
        self.freqMap = collections.defaultdict(create_linked_list)
        self.keyMap = {}

    def delete(self, node):
        if node.pre:
            node.pre.nex = node.nex
            node.nex.pre = node.pre
            if node.pre is self.freqMap[node.freq][0] and node.nex is self.freqMap[node.freq][-1]:
                self.freqMap.pop(node.freq)
        return node.key
        
    def increase(self, node):
        node.freq += 1
        self.delete(node)
        self.freqMap[node.freq][-1].pre.insert(node)
        if node.freq == 1:
            self.minFreq = 1
        elif self.minFreq == node.freq - 1:
            head, tail = self.freqMap[node.freq - 1]
            if head.nex is tail:
                self.minFreq = node.freq
    def compute(self,key:int)->int:
        if key in self.keyMap:
            return key
        return -1
    
    def get(self, key: int) -> int:
        if key in self.keyMap:
            self.increase(self.keyMap[key])
            return self.keyMap[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        if self.capacity != 0:
            if key in self.keyMap:
                node = self.keyMap[key]
                node.val = value
            else:
                node = Node(key, value)
                self.keyMap[key] = node
                self.size += 1
            if self.size > self.capacity:
                self.size -= 1
                deleted = self.delete(self.freqMap[self.minFreq][0].nex)
                self.keyMap.pop(deleted)
            self.increase(node)


class TEST:
    def __init__(self,static_cache_size:int=0, dynamic_cache_size:int=0,csr_topo: CSRTopo = None):
        self.static_cache_size= int(static_cache_size)
        self.dynamic_cache_size= int(dynamic_cache_size)
        self.csr_topo=csr_topo
        self.node_count=csr_topo.indptr.shape[0]-1
        self.feature_order = None

        self.i=0
        self.hit_num=0
        self.dynamic_hit_num=0
        self.total_num=0

        self.fifo=FIFOCache(self.dynamic_cache_size)     #fifo

        self.lru=LRUCache(self.dynamic_cache_size)

        self.lfu=LFUCache(self.dynamic_cache_size)
        '''self.last_tensor=deque(maxlen=self.dynamic_cache_size)
        tlist=range(self.static_cache_size,self.static_cache_size+self.dynamic_cache_size)
        self.last_tensor.extend(tlist)'''
        self.last_tensor=None
    def isneeddynamic(self):
        figure_name='ogbn_products'
        degree=self.csr_topo.indptr[1:]-self.csr_topo.indptr[:-1]
        ldegree=degree.tolist()
       
       
        fig0=plt.figure()
        fit=powerlaw.Fit(ldegree)
        #下面代码来自于https://github.com/keflavich/plfit/blob/master/plfit/plfit.py
        self.data=degree.numpy()
        self._ks=fit.power_law.D
        xmin = fit.xmin
        alpha = fit.power_law.alpha

        niter = 0
        ntail = sum(self.data >= xmin)
        ntot = len(self.data)
        nnot = ntot-ntail              # n(<xmin)
        pnot = nnot/float(ntot)        # p(<xmin)
        nonpldata = self.data[self.data<xmin]
        nrandnot = sum( np.random.rand(ntot) < pnot ) # randomly choose how many to sample from <xmin
        nrandtail = ntot - nrandnot         # and the rest will be sampled from the powerlaw

        ksv = []
        for i in range(niter):
            # first, randomly sample from power law
            # with caveat!
            nonplind = np.floor(np.random.rand(nrandnot)*nnot).astype('int')
            fakenonpl = nonpldata[nonplind]
            randarr = np.random.rand(nrandtail)
            fakepl = randarr**(1/(1-alpha)) * xmin
            fakedata = np.concatenate([fakenonpl,fakepl])
            newfit=powerlaw.Fit(fakedata)
            #print(f'拟合得到幂律分布幂指数:{fit.power_law.alpha},   d:{newfit.power_law.D}')
            ksv.append(newfit.power_law.D)

        ksv = np.array(ksv)
        p = (ksv*1.5>self._ks).sum() / float(niter)
        self._pval = p
        self._ks_rand = ksv
        print(f'p:({niter})={(p):.3f}')
        

        print(f'拟合得到幂律分布幂指数:{fit.power_law.alpha}')#拟合得到幂律分布幂指数alpha
        print(f'图数据degree与拟合幂律函数之间Kolmogorov-Smirnov距离D:{fit.power_law.D}')#数据ldegerr和拟合之间的Kolmogorov-Smirnov距离D
        ax0=fit.plot_pdf(color = 'b', linewidth = 2)
        fit.power_law.plot_pdf(color = 'g', linestyle = 'dashdot', ax = ax0)
        #fit.plot_ccdf(color = 'r', linewidth = 2, ax = ax0)
        #fit.power_law.plot_ccdf(color = 'g', linestyle = 'dashdot', ax = ax0)
        ax0.title.set_text(figure_name)
        ax0.set_ylabel("P(X)")
        ax0.set_xlabel("degree")
        power_fit_root ='/data/wyj/debug0/pictures/'+figure_name+'_power_fit.png'
        fig0.savefig(power_fit_root)

        cdegree=collections.Counter(ldegree)

        plt.figure()
        
        plt.scatter(cdegree.keys(),cdegree.values(), c='purple', alpha=0.2, edgecolors="grey")
        
        plt.title(figure_name)
        plt.xlabel("degree")
        plt.ylabel("num")
        root ='/data/wyj/debug0/pictures/'+figure_name+'_degree.png'
        plt.savefig(root)

    def begin_compute_missrate(self):  #与get_miss_rete()配合使用，计算命中率
        self.hit_num=0
        self.dynamic_hit_num=0
        self.total_num=0
        self.i=0
    def reindex(self,gpu_portion:int =0):
        adj_csr=self.csr_topo
        node_count = self.node_count
        total_range = torch.arange(node_count, dtype=torch.long)
        perm_range = torch.randperm(int(node_count * gpu_portion))
   
        degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
        _, prev_order = torch.sort(degree, descending=True)
        new_order = torch.zeros_like(prev_order)
        prev_order[:int(node_count * gpu_portion)] = prev_order[perm_range]
        new_order[prev_order] = total_range
        self.feature_order= new_order
    def sample_reindex(self,norder:None, gpu_portion:int=0):
        node_count = self.node_count #计算稀疏矩阵节点数
        total_range = torch.arange(node_count, dtype=torch.long) #tensor([0,1,2,3,……,node_count-1])
        perm_range = torch.randperm(int(node_count * gpu_portion)) #torch.randperm(n) 随机打乱tensor([2,0,1,8,……])
        degree = norder 
        _, prev_order = torch.sort(degree, descending=True)  #降序排列
        new_order = torch.zeros_like(prev_order)
        prev_order[:int(node_count * gpu_portion)] = prev_order[perm_range]#排好序后，前x个节点随机打乱
        new_order[prev_order] = total_range
        self.feature_order=new_order
    def get_miss_rate(self):
        hit_rate = float(self.hit_num) / self.total_num
        dynamic_hit_rate=float(self.dynamic_hit_num)/self.total_num
        print(f'lru:total_num:{self.total_num},static_hit_rate:{(hit_rate):.4f},dynamic_hit_rate:{(dynamic_hit_rate):.4f},hit_rate:{(hit_rate+dynamic_hit_rate):.4f}')
    def __getitem__(self, node_idx: torch.Tensor):
        self.i+=1
        if self.feature_order is not None:
            node_idx = self.feature_order[node_idx]
            self.total_num += node_idx.size()[0]
            temp=torch.nonzero(node_idx<self.static_cache_size) 
            self.hit_num+=temp.shape[0]  #计算静态命中率
            
            if self.dynamic_cache_size!=0:  #计算LFU动态缓存命中率
                dtemp=torch.nonzero(node_idx>self.static_cache_size)
                dylist=node_idx[dtemp].view(-1)
                dylist=dylist.tolist()            
                for i in dylist:
                    if self.lfu.compute(i)!=-1:
                        self.dynamic_hit_num+=1
                for i in dylist:
                    self.lfu.put(i,0)

            '''
            if self.dynamic_cache_size!=0:  #计算LRU动态缓存命中率
                dtemp=torch.nonzero(node_idx>self.static_cache_size)
                dylist=node_idx[dtemp].view(-1)
                dylist=dylist.tolist()           
                for i in dylist:
                    if self.lru.compute(i)!=-1:
                        self.dynamic_hit_num+=1
                for i in dylist:
                    self.lru.put(i)
            '''
            '''
            if self.i%2==1 and self.dynamic_cache_size!=0 and self.last_tensor is not None: #fill the same id between 2 minibatch
                print('begin-------------------------------------------------')
                self.total_num += node_idx.size()[0]
                temp=torch.nonzero(node_idx<self.static_cache_size) 
                self.hit_num+=temp.shape[0]  #计算静态命中率

                print(f"self.last_tensor.size:{self.last_tensor.shape}")
                dtemp=torch.nonzero(node_idx>self.static_cache_size)
                dylist=node_idx[dtemp].view(-1)
                dylist=dylist.tolist()            #本次访问的node_idx中超出静态缓存的部分
                oldlist=self.last_tensor.tolist()
                set_c=set(dylist)&set(oldlist) 
                self.dynamic_hit_num+=len(set_c) 

            if self.i%2==1:self.last_tensor=node_idx
            else:
                last_node_idx=self.last_tensor
                cpu_node_idx=node_idx
                superset = torch.cat([last_node_idx,cpu_node_idx])
                uniset, count = superset.unique(return_counts=True)
                mask = (count == 2)
                result = uniset.masked_select(mask)
                self.last_tensor=result
                if self.i<=8 and self.i%2==0:
                    for j in range(0,10):
                        begin,end=j/10*self.node_count,(j+1)/10*self.node_count
                        a=(self.last_tensor>=begin)
                        b=(self.last_tensor<end)
                        idx=torch.nonzero(a*b)
                        num=idx.shape[0]                            
                        rate=num*1.0/node_idx.size()[0]

                        fill_rate=num*1.0/self.node_count
                        sum_rate=1.0*node_idx.size()[0]/self.node_count
                        print(f"minibatch_node_num:{node_idx.size()[0]},sum_fill:{sum_rate:.4f},{j*10}%~{j*10+10}%,,same_ratio:{rate:.4f},can_fill_ratio:{fill_rate:.4f}")
                
                second_tensor=self.last_tensor.clone()
                temp=torch.nonzero(self.last_tensor>self.static_cache_size) 
                second_tensor=second_tensor[temp].view(-1)
                print(f'i:{self.i}, num:{second_tensor.size()}')
                
                print('minibatch end ------------------------------------')
                
                fillnum=self.dynamic_cache_size-second_tensor.shape[0]
                if fillnum>0:
                    dtemp=torch.nonzero(node_idx>self.static_cache_size)
                    node_idx_clone=node_idx[dtemp].view(-1)
                    superset = torch.cat([node_idx_clone,second_tensor])
                    uniset, count = superset.unique(return_counts=True)
                    mask = (count == 1)
                    result = uniset.masked_select(mask)
                    result=result[:fillnum]
                    self.last_tensor=torch.cat([second_tensor,result])
                else:
                    self.last_tensor=second_tensor[:self.dynamic_cache_size] 
            '''    
            '''
            if self.dynamic_cache_size!=0:  #计算fifo动态缓存命中率
                #print(f"self.dynamic_hit_num:{self.dynamic_hit_num}")
                #print(f"dself.fifo.size:{self.fifo.size}")
                dtemp=torch.nonzero(node_idx>self.static_cache_size)
                dylist=node_idx[dtemp].view(-1)
                dylist=dylist.tolist()           
                for i in dylist:
                    if self.fifo.get(i)!=-1:
                        self.dynamic_hit_num+=1
                for i in dylist:
                    if self.fifo.get(i)==-1:
                        self.fifo.put(i)
            '''
            ''' 
            if self.dynamic_cache_size!=0:      #缓存上一个minibatch的访问节点特征
                thislist=node_idx.tolist()            
                oldlist=list(self.last_tensor)
                set_c=set(thislist)&set(oldlist) 
                #print(f'len_dynamic_hit:{len(set_c)}')
                self.dynamic_hit_num+=len(set_c)      
        ma= int(0.3*self.node_count)
        a=(node_idx>self.static_cache_size)
        b=(node_idx<ma)
        idx=torch.nonzero(a*b)
        num=idx.shape[0]
        fill_rate=num*1.0/self.node_count
        #print(f'fill_rate:{fill_rate:.4f}')
        dylist=node_idx[idx].view(-1).tolist()
        self.last_tensor.extend(dylist)   
        #print(f'last_tensor_size:{len(self.last_tensor)}')   
        '''
    
norder=None
node_count=csr_topo.indptr.shape[0] - 1
print(data.x.size(),"  ",data.x.size()[0],"  ",node_count)
torch.cuda.synchronize()
start=time.time()
def node_order(iter):
    i=0
    new_nodeorder=torch.zeros(data.x.size()[0],dtype=torch.long)
    while i<iter:
        for seeds in train_loader: 
            i+=1
            if i%20==0:print(i)
            if i>=iter:break
            n_id, batch_size, _ = quiver_sampler.sample(seeds)
            new_nodeorder[n_id]+=1
    return new_nodeorder
#norder=node_order(200)
torch.cuda.synchronize()
end=time.time()
print(f'create new order time:{(end-start):.4f}s')



cache_size=data.x.size()[0]*0.5
dynamic_cache_size=data.x.size()[0]*0
print(f'dynamic_cache_size:{dynamic_cache_size}')
x=TEST(cache_size,dynamic_cache_size,csr_topo)
x.isneeddynamic()
x.reindex(0)



def test(epoch):
    x.begin_compute_missrate()
    for seeds in train_loader:
        n_id, batch_size, adjs = quiver_sampler.sample(seeds)
        x[n_id]
    x.get_miss_rate()

'''
for i in range(1,7):
    cache_size=data.x.size()[0]*0.1*i
    print(f'cache_size:{cache_size}')
    x=TEST(cache_size,0,csr_topo)
    x.sample_reindex(norder,0)
    for j in range(0,2):
        print('')
        print(f'Run {i*j:02d}:')
        test(i*j)
'''
'''
for i in range(0,1):
        print('')
        print(f'Run {i:02d}:')
        test(i)
'''
'''
for i in range(1,7):
    dynamic_cache_size=data.x.size()[0]*0.1*i
    print(f'dynamic_cache_size:{dynamic_cache_size}')
    x=TEST(cache_size,dynamic_cache_size,csr_topo)
    x.reindex(0)
    for j in range(1,2):
        print('')
        print(f'Run {i*j:02d}:')
        test(i*j)
'''  