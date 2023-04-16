#pragma once
#include <stdio.h>
#include <torch/extension.h>
#include <iostream>
#include <map>
#include <stdint.h>
#include <vector>
#define WARP_SIZE 32

/*
CAS是英文单词CompareAndSwap的缩写，中文意思是：比较并替换。CAS需要有3个操作数：内存地址V，旧的预期值A，即将要更新的目标值B。

CAS指令执行时，当且仅当内存地址V的值与预期值A相等时，将内存地址V的值修改为B，否则就什么都不做。整个比较并替换的操作是一个原子操作
返回值为未修改前的Destination位置的初始值
*/
// Insert the key/values  into the hashtable
__device__ void gpu_hashtable_insert(KeyValue* hashtable, const KeyValue& kvs)
{
    unsigned long long key=kvs.key;
    uint64_t value=kvs.value;
    uint64_t slot=hash(key);
    while(true)
    {
        uint64_t prev=atomicCAS(&hashtable[slot].key,kEmpty,key);
        if(prev==kEmpty || prev==key)
        {
            hashtable[slot].value=value;
            return;
        }
        slot=(slot+1)&(kHashTableCapacity-1);
    }
}
__device__ uint64_t gpu_hashtable_find(KeyValue*hashtable, const uint64_t & key)
{
    uint64_t slot=hash(key);
    while(true)
    {
        if(hashtable[slot].key==key)
        {
            return hashtable[slot].value;
        }
        if(hashtable[slot].key==kEmpty)
        {
            return kEmpty;
        }
        slot=(slot+1) &(kHashTableCapacity-1);
    }
}

__device__ void gpu_hashtable_erase(KeyValue*hashtable, const uint64_t & key)
{
    uint64_t slot=hash(key);
    while(true)
    {
        if(hashtable[slot].key==key)
        {
            hashtable[slot].key=kEmpty;
            hashtable[slot].value=kEmpty;
            return;
        }
        if(hashtable[slot].key==kEmpty)
        {
            return;
        }
        slot=(slot+1) &(kHashTableCapacity-1);
    }
}

__device__ int find(const int64_t *offsets, const int device_count,
                    const int64_t index)
{
    if(index < 0){
    	return -1;
    }
    int i = 1;
    for (i = 1; i <= device_count; i++) {
        if (index < offsets[i]) { return i - 1; }
    }
    return -1;
}
//将tensor内的node id对应的特征聚合到dynamic cache；并将nid-idx键值对插入哈希表
__global__ void tensor_gpu_hashtable_insert(KeyValue* hashtable, const int64_t *indices,int indice_length, char **dev_ptrs, const int64_t *offsets,
                                     const int device_count,int*mflag,int*global_block,const int stride)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // each warp take charge of one-feature copy
    unsigned int warp_id = tid / WARP_SIZE;
    unsigned int warp_step = step / WARP_SIZE;

    unsigned int warp_start = warp_id;
    unsigned int thread_start = tid % WARP_SIZE;

    int64_t dev_index = 0;
    int64_t dev_offset = 0;
    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;
    char *dev_ptr;
    unsigned int local_start = thread_start;

    char* dynamic_dev_ptr=dev_ptrs[device_count-2];
    while (warp_start < indice_length) {
        local_start = thread_start;
        //dev_index =device_count-3;
        dev_index = find(offsets, device_count, indices[warp_start]);
        dev_ptr=dev_ptrs[dev_index];
        dev_offset = indices[warp_start] - offsets[dev_index];
        src_copy_start = dev_offset * stride;
        dst_copy_start = warp_start * stride;
        for (; local_start < stride; local_start += WARP_SIZE) {
            dynamic_dev_ptr[dst_copy_start + local_start] =
                dev_ptr[src_copy_start + local_start];
        }
        bool blocked=true;  //每个mini_batch都要申请、初始化、传入array[],长度为nid_length
        while(blocked && mflag[warp_start]==-1){
            if (0 == atomicCAS(&global_block[warp_start], 0, 1)) {
                KeyValue tkv=KeyValue{(uint64_t)indices[warp_start],(uint64_t)warp_start};
                //KeyValue tkv=KeyValue{(uint64_t)indices[warp_start],(uint64_t)0};
                gpu_hashtable_insert(hashtable,tkv);
                mflag[warp_start]=warp_start;
                atomicExch(&global_block[warp_start], 0);
                blocked=false;
            }
        }
        warp_start += warp_step;
    }
}
//最初append时，填充dynamic cache
__global__ void res_to_dynamic_gather(char **dev_ptrs,const int device_count, char*res,
                                     const int *array, int* insertnum,
                                     const int stride,int* exchange_begin)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // each warp take charge of one-feature copy
    unsigned int warp_id = tid / WARP_SIZE;
    unsigned int warp_step = step / WARP_SIZE;

    unsigned int warp_start = warp_id;
    unsigned int thread_start = tid % WARP_SIZE;

    unsigned int local_start = thread_start;
    const unsigned int beginidx=(unsigned int)*exchange_begin;
    int indice_length=*insertnum;

    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;
    int nidx=0;
    char* dynamic_dev_ptr=dev_ptrs[device_count-2];
    while(warp_start<indice_length)
    {
        nidx=array[warp_start];
        src_copy_start = nidx * stride;
        dst_copy_start = (warp_start+beginidx) * stride;
        for (; local_start < stride; local_start += WARP_SIZE) {
            dynamic_dev_ptr[dst_copy_start + local_start] =
            res[src_copy_start + local_start];
        }
        warp_start += warp_step;
    }
}

__global__ void quiver_tensor_gather_static(char **dev_ptrs, const int64_t *offsets,
                                     const int device_count,
                                     const int64_t *indices, int indice_length,
                                     char *res, const int stride,
                                     const int *access_book,
                                     const int ignore_access_book,int *minibatch,int*mflag,int*global_block)
{

    //
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // each warp take charge of one-feature copy
    unsigned int warp_id = tid / WARP_SIZE;
    unsigned int warp_step = step / WARP_SIZE;

    unsigned int warp_start = warp_id;
    unsigned int thread_start = tid % WARP_SIZE;

    int64_t dev_index = 0;
    int64_t dev_offset = 0;
    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;
    char *dev_ptr;
    unsigned int local_start = thread_start;
    
    while (warp_start < indice_length) {
            local_start = thread_start;
            dev_index = find(offsets, device_count, indices[warp_start]);
        // we only copy data from reachable device
            if (dev_index != -1 && (ignore_access_book || access_book[dev_index])) {
                dev_ptr = dev_ptrs[dev_index];
                dev_offset = indices[warp_start] - offsets[dev_index];
                src_copy_start = dev_offset * stride;
                dst_copy_start = warp_start * stride;
                for (; local_start < stride; local_start += WARP_SIZE) {
                    res[dst_copy_start + local_start] =
                    dev_ptr[src_copy_start + local_start];
                }

                bool blocked=true;  //每个mini_batch都要申请、初始化、传入array[],长度为nid_length
                while(access_book[dev_index]==1 && blocked && mflag[warp_start]==-1){
                    if (0 == atomicCAS(&global_block[warp_start], 0, 1)) {
                        minibatch[warp_start]=1;
                        mflag[warp_start]=warp_start;
                        atomicExch(&global_block[warp_start], 0);
                        blocked=false;
                    }
                }
            }
            warp_start += warp_step;
    }
        
}
__global__ void quiver_tensor_gather(char **dev_ptrs, const int64_t *offsets,
                                     const int device_count,
                                     const int64_t *indices, int indice_length,
                                     char *res, const int stride,
                                     const int *access_book,
                                     const int ignore_access_book,const int dynamic_cache_size,
                                     KeyValue* hashtable,
                                     int *dynamic_minibatch,int*mflag,int*global_block,int*static_minibatch)
{

    //
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // each warp take charge of one-feature copy
    unsigned int warp_id = tid / WARP_SIZE;
    unsigned int warp_step = step / WARP_SIZE;

    unsigned int warp_start = warp_id;
    unsigned int thread_start = tid % WARP_SIZE;

    int64_t dev_index = 0;
    int64_t dev_offset = 0;
    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;
    char *dev_ptr;
    unsigned int local_start = thread_start;

        char* dynamic_dev_ptr=dev_ptrs[device_count-2];
        while (warp_start < indice_length) {
            local_start = thread_start;
            dev_index = find(offsets, device_count, indices[warp_start]);
            int64_t dynamic_idx=(int64_t)gpu_hashtable_find(hashtable,(uint64_t)indices[warp_start]);
            if(dynamic_idx!= -1){
                src_copy_start=dynamic_idx*stride;
                dst_copy_start=warp_start*stride;
                for (; local_start < stride; local_start += WARP_SIZE) { 
                    res[dst_copy_start + local_start] =dynamic_dev_ptr[src_copy_start + local_start];
                }
                
                bool blocked=true; 
                while( blocked && mflag[warp_start]==-1){
                    if (0 == atomicCAS(&global_block[warp_start], 0, 1)) {
                        dynamic_minibatch[warp_start]=1;
                        mflag[warp_start]=warp_start;
                        atomicExch(&global_block[warp_start], 0);
                        blocked=false;
                    }
                }
                
            }
            else if (dev_index !=-1 && (ignore_access_book || access_book[dev_index])){
                dev_ptr = dev_ptrs[dev_index];
                dev_offset = indices[warp_start] - offsets[dev_index];
                src_copy_start = dev_offset * stride;   //该node feature开始位置所在的地址
                dst_copy_start = warp_start * stride;
                for (; local_start < stride; local_start += WARP_SIZE) { 
                    res[dst_copy_start + local_start] =dev_ptr[src_copy_start + local_start];
                }
                
                if(access_book[dev_index]==2){
                    bool blocked=true; 
                    while( blocked && mflag[warp_start]==-1){
                    if (0 == atomicCAS(&global_block[warp_start], 0, 1)) {
                        /*KeyValue tkv=KeyValue{(uint64_t)warp_start,(uint64_t)indices[warp_start]};
                        if(gpu_hashtable_find(ptable,(uint64_t)warp_start)==kEmpty)gpu_hashtable_insert(ptable,tkv);*/
                        static_minibatch[warp_start]=1;
                        mflag[warp_start]=warp_start;
                        atomicExch(&global_block[warp_start], 0);
                        blocked=false;
                    }
                    }
                }
            }
            warp_start += warp_step;
        }
}

__global__ void
quiver_tensor_gather_aligned(float **dev_ptrs, const int64_t *offsets,
                             const int device_count, const int64_t *indices,
                             int indice_length, float *res, const int stride)
{

    //
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_thread = gridDim.x * blockDim.x;

    unsigned int warp_start = thread_id;
    // unsigned int warp_end = (thread_id + 1) * WARP_SIZE;
    // unsigned int thread_local = thread_id % WARP_SIZE;

    int64_t dev_index = 0;
    int64_t dev_offset = 0;
    float *dev_ptr;
    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;
    unsigned int output_index, output_offset;

    while (warp_start < indice_length * stride) {
        output_index = warp_start / stride;
        output_offset = warp_start % stride;
        dev_index = find(offsets, device_count, indices[output_index]);
        dev_ptr = dev_ptrs[dev_index];
        dev_offset = indices[output_index] - offsets[dev_index];

        src_copy_start = dev_offset * stride + output_offset;
        dst_copy_start = output_index * stride + output_offset;
        res[dst_copy_start] = dev_ptr[src_copy_start];
        warp_start += num_thread;
    }
}
