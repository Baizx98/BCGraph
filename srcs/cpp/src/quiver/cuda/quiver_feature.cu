#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <quiver/common.hpp>
#include <quiver/quiver.cu.hpp>
#include <quiver/linearprobing.cu.hpp>
#include <quiver/shard_tensor.cu.hpp>
#include <torch/extension.h>

#include <atomic>
#include <iostream>
#include <string>
#include <torch/csrc/utils/python_numbers.h>
#include <unordered_map>

#include <algorithm>
#include <cmath>
#include <vector>

namespace quiver
{
#define CHECK_CPU(x) AT_ASSERTM(!x.device().is_cuda(), #x " must be CPU tensor")
class ShardTensorItem
{
  public:
    int device;
    cudaIpcMemHandle_t mem_handle;
    std::vector<int> shape;
    // for now we assume it is all float
    int element_size;
    ShardTensorItem(int device_, cudaIpcMemHandle_t mem_handle_,
                    std::vector<int> shape_)
        : device(device_), mem_handle(mem_handle_), shape(shape_)
    {
    }
    ShardTensorItem(){

    };
    std::tuple<int, int, py::bytes, std::vector<int>> share_ipc()
    {
        auto _handle = PyBytes_FromStringAndSize((char *)&mem_handle,
                                                 CUDA_IPC_HANDLE_SIZE);
        auto bytes_obj = py::reinterpret_steal<py::object>((PyObject *)_handle);
        return std::make_tuple(device, element_size, bytes_obj, shape);
    }
    void from_ipc(std::tuple<int, int, std::string, std::vector<int>> ipc_data)
    {

        device = std::get<0>(ipc_data);
        element_size = std::get<1>(ipc_data);
        shape = std::get<3>(ipc_data);
        auto handle = std::get<2>(ipc_data);
        auto ipc_handle =
            reinterpret_cast<const cudaIpcMemHandle_t *>(handle.c_str());

        mem_handle = *ipc_handle;
    }
};

class ShardTensor
{
  public:
    ShardTensor(int device) : device_(device), inited_(false), device_count_(0)
    {

        offset_list_.push_back(0);
    }
    ~ShardTensor()
    {
        cudaFree(pHashTable);
    }
    size_t get_tensor_bytes(torch::Tensor tensor)
    {
        // assume it's float
        int dim = tensor.dim();
        size_t total_bytes = element_size;
        for (int index = 0; index < dim; index++) {
            total_bytes *= tensor.sizes()[index];
        }
        return total_bytes;
    }
    std::vector<int> get_tensor_shape(torch::Tensor tensor)
    {
        std::vector<int> shape;
        int dim = tensor.dim();
        for (int index = 0; index < dim; index++) {
            shape.push_back(tensor.sizes()[index]);
        }
        return shape;
    }

    void append(ShardTensorItem item)
    {
        cudaSetDevice(device_);
        if (!inited_) {
            shape_.resize(item.shape.size());
            shape_[0] = 0;
            auto tensor_sizes = item.shape;
            for (int index = 1; index < shape_.size(); index++) {
                shape_[index] = tensor_sizes[index];
            }
            inited_ = true;
        }
        offset_list_.push_back(offset_list_[offset_list_.size() - 1] +
                               item.shape[0]);

        // Check accessbility
        if (item.device >= 0) {
            // TODO

            int access_i_j, access_j_i;
            cudaDeviceCanAccessPeer(&access_i_j, device_, item.device);
            cudaDeviceCanAccessPeer(&access_j_i, item.device, device_);
            if ((access_i_j && access_j_i) || device_ == item.device) {
                access_book.push_back(1);
                // printf("%d <-> %d support peer access \n", device_,
                // item.device);
            } else {
                access_book.push_back(0);
                // printf("%d <-> %d dont support peer access \n", device_,
                // item.device);
            }

        } else {
            access_book.push_back(1);
            // printf("%d <-> CPU support peer access \n", device_);
        }
        // get dev_ptr that can be accessed from this process
        void *ptr = NULL;
        tensor_devices_.push_back(item.device);
        if (!access_book[access_book.size() - 1]) {
            cudaSetDevice(item.device);
            cudaIpcOpenMemHandle(&ptr, item.mem_handle,
                                 cudaIpcMemLazyEnablePeerAccess);
            cudaSetDevice(device_);
            // printf("WARNING: Tensor from device %d can NOT be accessed in
            // kernel launched on device %d \n", item.device, device_);
        } else {
            cudaIpcOpenMemHandle(&ptr, item.mem_handle,
                                 cudaIpcMemLazyEnablePeerAccess);
        }

        //
        dev_ptrs_.push_back(ptr);
        element_size = item.element_size;
        shape_[0] += item.shape[0];
        device_count_ += 1;
        cudaCheckError();
    }

    void append(torch::Tensor &tensor, int target_device, bool is_dynamic_cache=false)
    {
        CHECK_CPU(tensor);
        // for now, we assume tensor is added ordered
        if (!inited_) {
            shape_.resize(tensor.dim());
            shape_[0] = 0;
            auto tensor_sizes = tensor.sizes();
            for (int index = 1; index < shape_.size(); index++) {
                shape_[index] = tensor_sizes[index];
            }
            inited_ = true;
        }
        element_size = tensor.element_size();
        tensor_shapes_.push_back(get_tensor_shape(tensor));

        offset_list_.push_back(offset_list_[offset_list_.size() - 1] +
                               tensor.sizes()[0]);

        void *ptr = NULL;
        size_t data_size = get_tensor_bytes(tensor);
        tensor_devices_.push_back(target_device);
        if (target_device >= 0) {
            // if target_device >= 0, it means we use p2p
            // printf("LOG >>> Malloc Data On Device %d With %ulld Bytes\n",
            // target_device, data_size);
             if(is_dynamic_cache)
            {
                dynamic_cache_size=tensor.sizes()[0];
                pHashTable=create_hashtable();   //创建动态缓存哈希表
                cudaCheckError();

                cudaSetDevice(target_device); 
                cudaMalloc(&ptr, data_size);     //申请动态缓存空间

                //dynamic_dev_ptr=(char*)ptr;
                if(cpu_tensor_beginid!=-1 && cpu_tensor_dptr!=nullptr)
                {
                    printf("begin fill dynamic cache--------,cpu_tensor_beginid:%d,ptr:%p    \n",cpu_tensor_beginid,cpu_tensor_dptr);
                    cudaMemcpy(ptr,cpu_tensor_dptr,data_size,cudaMemcpyHostToDevice);
                    std::vector<KeyValue> insert_kvs ;
                    for(uint64_t i=0;i<dynamic_cache_size;i++)
                    {
                        uint64_t rand0=cpu_tensor_beginid+i;
                        uint64_t rand1=i;
                        insert_kvs.push_back(KeyValue{rand0,rand1});
                    }
                    insert_hashtable(pHashTable,insert_kvs.data(),(uint32_t)dynamic_cache_size);
                }
                cudaSetDevice(device_); 
                access_book.push_back(3);
            }
            else{
            cudaSetDevice(target_device);
            cudaMalloc(&ptr, data_size);
            cudaMemcpy(ptr, tensor.data_ptr(), data_size,
                       cudaMemcpyHostToDevice);
            cudaSetDevice(device_);

            // decide access book

            int access_i_j, access_j_i;
            cudaDeviceCanAccessPeer(&access_i_j, device_, target_device);
            cudaDeviceCanAccessPeer(&access_j_i, target_device, device_);
            if ((access_i_j && access_j_i) || device_ == target_device) {
                access_book.push_back(1);
                // printf("%d <-> %d support peer access \n", device_,
                // target_device);
            } else {
                access_book.push_back(0);
                // printf("%d <-> %d dont support peer access \n", device_,
                // target_device);
            }
            }

        } else {
            cudaSetDevice(device_);
            // if target_device < 0, it means we use Zero-Copy
            quiverRegister(tensor.data_ptr(), data_size,
                           cudaHostRegisterMapped);
            cudaHostGetDevicePointer(&ptr, (void *)tensor.data_ptr(), 0);
            cpu_tensor_dptr=(void*)tensor.data_ptr();
            cpu_tensor_beginid=(int)offset_list_[offset_list_.size() - 1];
            access_book.push_back(2);
            //printf("append(cpu),cpu_tensor_dptr:%p\n",cpu_tensor_dptr);
            // printf("%d <-> CPU support peer access \n", device_);
        }

        dev_ptrs_.push_back(ptr);

        shape_[0] += tensor.size(0);
        device_count_ += 1;
    }

    std::tuple<char **, int64_t *, int *> get_device_pointers(int device)
    {
        auto iter = device_pointers_map.find(device);
        if (iter == device_pointers_map.end()) {
            char **buffers_device;
            int64_t *offset_device;
            int *access_book_device;

            // Copy buffers Device
            cudaMalloc((void ***)&buffers_device,
                       sizeof(float *) * device_count_);
            cudaMemcpy(buffers_device, &dev_ptrs_[0],
                       sizeof(float *) * dev_ptrs_.size(),
                       cudaMemcpyHostToDevice);
            cudaCheckError();

            // copy offset
            cudaMalloc((void **)&offset_device,
                       sizeof(int64_t) * offset_list_.size());
            cudaMemcpy(offset_device, &offset_list_[0],
                       sizeof(int64_t) * offset_list_.size(),
                       cudaMemcpyHostToDevice);
            cudaCheckError();

            cudaMalloc((void **)&access_book_device,
                       sizeof(int) * access_book.size());
            cudaMemcpy(access_book_device, &access_book[0],
                       sizeof(int) * access_book.size(),
                       cudaMemcpyHostToDevice);
            cudaCheckError();
            device_pointers_map.emplace(
                device, std::make_tuple(buffers_device, offset_device,
                                        access_book_device));
            iter = device_pointers_map.find(device);
        }
        return iter->second;
    }
    void begin_compute_missrate()
    {
        printf("begining---------------------------------------------------------------\n");
        total_num=0;
        static_get_num=0;
        dynamic_get_num=0;
        report=0;
    }
    void get_miss_rate()
    {
       
        float sget_rate=(float)static_get_num/(float)total_num;
        float dget_rate=(float)dynamic_get_num/(float)total_num;
        if(dynamic_cache_size==0)
        {
            printf("static_cache_hit: total_num:%d,  static_hit_num:%d,  static_hit_rate:%.4f \n",total_num, static_get_num, sget_rate);
        }
        else{
            printf("dynamic_cache_hit: total_num:%d,  dynamic_hit_num:%d,  dynamic_hit_rate:%.4f \n",total_num, dynamic_get_num, dget_rate);
            printf("miss:  total_num:%d,  miss_num:%d,  miss_rate:%.4f ,total_hit_rate:%.4f\n",total_num, static_get_num, sget_rate,(1.0-sget_rate));
        //printf("total:miss_rate:%.4f,  hit_rate:%.4f\n",(1.0-sget_rate-dget_rate),(sget_rate+dget_rate));
        }
        printf("this epoch end---------------------------------------------------------\n");
    }
    void dynamic_cache(torch::Tensor &indices)
    {
        int current_device = 0;
        cudaGetDevice(&current_device);
        char **buffers_device;
        int64_t *offset_device;
        int *access_book_device;

        auto val = get_device_pointers(current_device);
        buffers_device = std::get<0>(val);
        offset_device = std::get<1>(val);
        access_book_device = std::get<2>(val);

        cudaMemset(pHashTable, 0xff, sizeof(KeyValue) * kHashTableCapacity);
        int t=min(dynamic_cache_size,(int)indices.numel());
        int hflag[t];
        std::memset(hflag,-1,sizeof(int)*t);
        int*dflag;
        cudaMalloc((void **)&dflag, t*sizeof(int));
        cudaMemcpy(dflag, hflag, t*sizeof(int), cudaMemcpyHostToDevice);

        int*global_block;
        std::memset(hflag,0,sizeof(int)*t);
        cudaMalloc((void **) &global_block, t*sizeof(int));
        cudaMemcpy(global_block, hflag, t*sizeof(int), cudaMemcpyHostToDevice);
        int nblockSize = 0;
        int nnumBlocks = 0;
        cudaOccupancyMaxPotentialBlockSize(&nnumBlocks, &nblockSize,
                                           tensor_gpu_hashtable_insert);
        tensor_gpu_hashtable_insert<<<nnumBlocks,nblockSize>>>(pHashTable,indices.data_ptr<int64_t>(),t, buffers_device, offset_device, offset_list_.size(),dflag,global_block,stride_in_bytes(0));
        cudaCheckError();

        cudaFree(dflag);
        cudaFree(global_block);
        //an illegal memory access was encountered    删除的时候是直接把那块地址的内容都删了？？？
        /*cudaFree(buffers_device);
        cudaFree(offset_device);
        cudaFree(access_book_device);*/
    }
    torch::Tensor operator[](torch::Tensor &indices)
    {
        /*
        __global__ void quiver_tensor_gather(const int64_t** dev_ptrs, const
        int64_t* offsets, const int device_count, const int64_t* indices, int
        indice_length, const float* res, const int item_byte_size){
        torch::zeros((100,100),torch::KF32);
        */
        ++report;
        int current_device = 0;
        cudaGetDevice(&current_device);
        auto stream = at::cuda::getCurrentCUDAStream();

        std::vector<int64_t> res_shape(shape_);
        res_shape[0] = indices.numel();
        // decide Tensor

        auto options = torch::TensorOptions();
        if(element_size == 2){
            options = options.dtype(torch::kFloat16).device(torch::kCUDA, current_device);
        }else if(element_size == 4){
            options = options.dtype(torch::kFloat32).device(torch::kCUDA, current_device);
        }

                    
        auto res = torch::empty(res_shape, options);
        cudaCheckError();

        // Device Data
        // for(int index = 0; index < offset_list_.size(); index++){
        //    std::cout<<"offset " << offset_list_[index]<<std::endl;
        //    std::cout<<"access_book[index] " << access_book[index]<<std::endl;
        //}

        char **buffers_device;
        int64_t *offset_device;
        int *access_book_device;

        auto val = get_device_pointers(current_device);
        buffers_device = std::get<0>(val);
        offset_device = std::get<1>(val);
        access_book_device = std::get<2>(val);

        int blockSize = 0;
        int numBlocks = 0;
        
        // std::cout<<"LOG >>> "<<" numBlocks "<< numBlocks <<" blockSize
        // "<<blockSize<<std::endl;
        int ignore_access_book = 0;
        if (current_device != device_) { ignore_access_book = 1; }

        if(dynamic_cache_size==0){
            cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize,
                                           quiver_tensor_gather_static);
            const int t=indices.sizes()[0];
            int hflag[t];
            std::memset(hflag,-1,sizeof(int)*t);
            int*dflag;
            cudaMalloc((void **)&dflag, t*sizeof(int));
            cudaMemcpy(dflag, hflag, t*sizeof(int), cudaMemcpyHostToDevice);

            int*global_block;
            int* minibatch;
            std::memset(hflag,0,sizeof(int)*t);
            cudaMalloc((void **) &global_block, t*sizeof(int));
            cudaMemcpy(global_block, hflag, t*sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void **) &minibatch, t*sizeof(int));
            cudaMemcpy(minibatch, hflag, t*sizeof(int), cudaMemcpyHostToDevice);
        
            /*int*static_get_rate;
            int state=0;
            cudaMalloc((void **)&static_get_rate,
                       sizeof(int));
            cudaMemcpy(static_get_rate, &state,
                       sizeof(int),cudaMemcpyHostToDevice);*/
            quiver_tensor_gather_static<<<numBlocks, blockSize, 0, stream>>>(
                buffers_device, offset_device, offset_list_.size(),
                indices.data_ptr<int64_t>(), indices.numel(), (char*)res.data_ptr(),
                stride_in_bytes(0), access_book_device, ignore_access_book,minibatch, dflag, global_block);
            cudaCheckError();
            
            cudaMemcpy(hflag,minibatch,sizeof(int)*t,cudaMemcpyDeviceToHost);
            total_num+=indices.sizes()[0];
            for(auto x:hflag)
            {
                static_get_num+=x;
            }
            //static_get_num+=accumulate(hflag,hflag+t,0);
            //cudaFree(static_get_rate);
            cudaFree(dflag);
            cudaFree(global_block);
            cudaFree(minibatch);
        }
        else if(dynamic_cache_size!=0){
            cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize,
                                           quiver_tensor_gather);
            const int t=indices.sizes()[0];
            int hflag[t];
            std::memset(hflag,-1,sizeof(int)*t);
            int*dflag;
            cudaMalloc((void **)&dflag, t*sizeof(int));
            cudaMemcpy(dflag, hflag, t*sizeof(int), cudaMemcpyHostToDevice);

            int*global_block;
            int* dynamic_minibatch;
            std::memset(hflag,0,sizeof(int)*t);
            cudaMalloc((void **) &global_block, t*sizeof(int));
            cudaMemcpy(global_block, hflag, t*sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void **) &dynamic_minibatch, t*sizeof(int));
            cudaMemcpy(dynamic_minibatch, hflag, t*sizeof(int), cudaMemcpyHostToDevice);

            int* static_minibatch;
            cudaMalloc((void **) &static_minibatch, t*sizeof(int));
            cudaMemcpy(static_minibatch, hflag, t*sizeof(int), cudaMemcpyHostToDevice);

          
            quiver_tensor_gather<<<numBlocks, blockSize, 0, stream>>>(
                buffers_device, offset_device, offset_list_.size(),
                indices.data_ptr<int64_t>(), indices.numel(), (char*)res.data_ptr(),
                stride_in_bytes(0), access_book_device, ignore_access_book,
                dynamic_cache_size, pHashTable,dynamic_minibatch, dflag, global_block,static_minibatch);

            cudaCheckError();
            cudaDeviceSynchronize();
            if(report%2){
                cudaMemcpy(hflag,static_minibatch,sizeof(int)*t,cudaMemcpyDeviceToHost);
                total_num+=t;
                int tsnum=0;
                for(auto x:hflag)tsnum+=x;
                static_get_num+=tsnum;

                int tflag[t];
                std::memset(tflag,0,sizeof(int)*t);
                cudaMemcpy(tflag,dynamic_minibatch,sizeof(int)*t,cudaMemcpyDeviceToHost);
                int tnum=0;
                for(auto x:tflag)tnum+=x;
                dynamic_get_num+=tnum;
            }
            //std::vector<KeyValue> oldkvs = iterate_hashtable(pHashTable);
            //printf("minibatch_ids:%d ,miss_num:%d, dynamic_hit_num:%d , phashtable_entrys:%d,dynamic_cache_size:%d\n",t,tsnum,tnum,(int)oldkvs.size(),dynamic_cache_size);
            
            cudaFree(dflag);
            cudaFree(global_block);
            cudaFree(dynamic_minibatch);
            cudaFree(static_minibatch);


            //实时更新动态缓存的内容
            /*std::vector<KeyValue> kvs = iterate_hashtable(ptable);
            int exchange_begin=max(0,dynamic_cache_size-(int)kvs.size());
            int insertnum=min(dynamic_cache_size,(int)kvs.size());
            int array[insertnum];
            std::vector<KeyValue> insert_kvs ;
            if((int)kvs.size()<dynamic_cache_size)
            {
                partern_erase_hashtable(pHashTable,(uint64_t)exchange_begin);
                //从res的array[i]位置拿到数据，移动到dynamic_cache的rand1位置
                for(uint64_t i=0;i<(uint64_t)kvs.size();i++)
                {
                    array[i]=kvs[i].key;
                    uint64_t rand0=kvs[i].value;
                    uint64_t rand1=i+(uint64_t)exchange_begin;
                    insert_kvs.push_back(KeyValue{rand0,rand1});
                }
            }
            else
            {
                cudaMemset(pHashTable, 0xff, sizeof(KeyValue) * kHashTableCapacity);
                for(uint64_t i=0;i<insertnum;i++)
                {
                    array[i]=kvs[i].key;
                    uint64_t rand0 =kvs[i].value;
                    uint64_t rand1 = i;
                    //printf("get key:%u,  value:%u\n",rand0,rand1);
                    insert_kvs.push_back(KeyValue{rand0,rand1});
                }   
            }
           // printf("ptable size:%d, need insert entrys:%d is==up miss num?   exchange_begin:%d\n",(int)kvs.size(),(int)insert_kvs.size(),exchange_begin);
            insert_hashtable(pHashTable,insert_kvs.data(),(uint32_t)insertnum);
            
            int*dinsertnum;
            cudaMalloc((void**)&dinsertnum,sizeof(int));
            cudaMemcpy(dinsertnum,&insertnum,sizeof(int),cudaMemcpyHostToDevice);

            int*darray;
            cudaMalloc((void **)&darray,
                    sizeof(int) * insertnum);
            cudaMemcpy(darray, &array[0],
                    sizeof(int) * insertnum,
                    cudaMemcpyHostToDevice);
            int*dexchange_begin;
            cudaMalloc((void **)&dexchange_begin,
                    sizeof(int));
            cudaMemcpy(dexchange_begin, &exchange_begin,
                    sizeof(int) ,
                    cudaMemcpyHostToDevice);
            cudaCheckError();
            int nblockSize = 0;
            int nnumBlocks = 0;
            cudaOccupancyMaxPotentialBlockSize(&nnumBlocks, &nblockSize,
                                           res_to_dynamic_gather);
            res_to_dynamic_gather<<<nnumBlocks,nblockSize>>>(buffers_device,offset_list_.size(),(char*)res.data_ptr(),darray,dinsertnum,stride_in_bytes(0),dexchange_begin);
            cudaCheckError();

            cudaFree(darray);
            cudaFree(ptable);
            cudaFree(dinsertnum);
            cudaFree(dexchange_begin);*/
        }
        return res;
    }

    std::vector<int64_t> shape() const { return shape_; }

    int device() const { return device_; }

    int size(int dim) const
    {
        if (shape_.size() == 0) return 0;
        return shape_[dim];
    }

    int64_t stride(int dim) const
    {
        int64_t res = 1;
        for (int index = dim + 1; index < shape_.size(); index++) {
            res *= shape_[index];
        }
        return res;
    }

    int64_t stride_in_bytes(int dim) const{
        return stride(dim) * element_size;
    }

    int64_t numel() const
    {
        int64_t res = 1;
        for (int index = 0; index < shape_.size(); index++) {
            res *= shape_[index];
        }
        return res;
    }
    std::vector<ShardTensorItem> share_ipc()
    {
        std::vector<ShardTensorItem> res;
        for (int index = 0; index < dev_ptrs_.size(); index++) {
            if (tensor_devices_[index] >= 0) {
                cudaSetDevice(tensor_devices_[index]);
                ShardTensorItem *item = new ShardTensorItem();
                item->device = tensor_devices_[index];
                item->shape = tensor_shapes_[index];
                item->element_size = element_size;
                cudaIpcGetMemHandle(&(item->mem_handle), dev_ptrs_[index]);
                res.push_back(*item);
            }
        }
        return res;
    }

    int device_count() const { return device_count_; }

    void unregister(torch::Tensor &cpu_tensor)
    {

        std::cout << "begin unregister" << std::endl;
        cudaHostUnregister((void *)cpu_tensor.data_ptr<float>());
        std::cout << "end unregister" << std::endl;
    }

  private:
    std::vector<int64_t> offset_list_;
    std::vector<void *> dev_ptrs_;
    std::vector<int> tensor_devices_;
    std::vector<int> access_book;
    std::vector<std::vector<int>> tensor_shapes_;
    std::vector<int64_t> shape_;
    std::unordered_map<int, std::tuple<char **, int64_t *, int *>>
        device_pointers_map;
    int numa_broker_device;
    int device_;
    int device_count_;
    bool inited_;
    int element_size;

    int64_t cpu_tensor_beginid=-1;
    void* cpu_tensor_dptr=nullptr;
    char* dynamic_dev_ptr=nullptr;
    
    KeyValue* pHashTable;
    int dynamic_cache_size=0;

    int total_num=0;
    int static_get_num=0;
    int dynamic_get_num=0;
    int report=0;
};

void init_p2p(std::vector<int> devices)
{
    std::cout << "LOG>>> P2P Access Initilization" << std::endl;

    for (int i = 0; i < devices.size(); i++) {
        int src = devices[i];
        cudaSetDevice(src);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, src);

        // CUDA IPC is only supported on devices with unified addressing
        if (!prop.unifiedAddressing) {
            printf(
                "Device %d does not support unified addressing, skipping...\n",
                i);
            continue;
        }
        // This sample requires two processes accessing each device, so we need
        // to ensure exclusive or prohibited mode is not set
        if (prop.computeMode != cudaComputeModeDefault) {
            printf(
                "Device %d is in an unsupported compute mode for this sample\n",
                i);
            continue;
        }

        for (int j = i + 1; j < devices.size(); j++) {
            int dst = devices[j];
            int access_i_j = 0;
            int access_j_i = 0;
            cudaDeviceCanAccessPeer(&access_i_j, src, dst);
            cudaDeviceCanAccessPeer(&access_j_i, dst, src);
            if (access_i_j && access_j_i) {
                printf("Enable P2P Access Between %d <---> %d \n", src, dst);
                cudaSetDevice(src);
                cudaDeviceEnablePeerAccess(dst, 0);
                cudaCheckError();
                cudaSetDevice(dst);
                cudaDeviceEnablePeerAccess(src, 0);
                cudaCheckError();
            }
        }
    }
}
bool can_device_access_peer(int src_device_index, int dst_device_index)
{
    int access_i_j = 0, access_j_i = 0;
    cudaDeviceCanAccessPeer(&access_i_j, src_device_index, dst_device_index);
    cudaDeviceCanAccessPeer(&access_j_i, dst_device_index, src_device_index);
    return (access_i_j == 1) && (access_j_i == 1);
}

}  // namespace quiver
void register_cuda_quiver_feature(pybind11::module &m)
{
    m.def("init_p2p", &quiver::init_p2p,
          py::call_guard<py::gil_scoped_release>());

    m.def("can_device_access_peer", &quiver::can_device_access_peer,
          py::call_guard<py::gil_scoped_release>());

    py::class_<quiver::ShardTensorItem>(m, "ShardTensorItem")
        .def(py::init<>())
        .def("share_ipc", &quiver::ShardTensorItem::share_ipc)
        .def("from_ipc", &quiver::ShardTensorItem::from_ipc);

    py::class_<quiver::ShardTensor>(m, "ShardTensor")
        //.def(py::init<std::vector<torch::Tensor>, int>())
        .def(py::init<int>())
        .def("__getitem__", &quiver::ShardTensor::operator[],
             py::call_guard<py::gil_scoped_release>())
        .def("unregister", &quiver::ShardTensor::unregister,
             py::call_guard<py::gil_scoped_release>())
        .def("shape", &quiver::ShardTensor::shape,
             py::call_guard<py::gil_scoped_release>())
        .def("numel", &quiver::ShardTensor::numel,
             py::call_guard<py::gil_scoped_release>())
        .def("device", &quiver::ShardTensor::device,
             py::call_guard<py::gil_scoped_release>())
        .def("stride", &quiver::ShardTensor::stride,
             py::call_guard<py::gil_scoped_release>())
        .def("size", &quiver::ShardTensor::size,
             py::call_guard<py::gil_scoped_release>())
        .def("device_count", &quiver::ShardTensor::device_count,
             py::call_guard<py::gil_scoped_release>())
        .def("begin_compute_missrate", &quiver::ShardTensor::begin_compute_missrate,
             py::call_guard<py::gil_scoped_release>())
        .def("get_miss_rate", &quiver::ShardTensor::get_miss_rate,
             py::call_guard<py::gil_scoped_release>())
        .def("dynamic_cache",
             py::overload_cast<torch::Tensor &>(
                 &quiver::ShardTensor::dynamic_cache),
             py::call_guard<py::gil_scoped_release>())
        .def("append",
             py::overload_cast<torch::Tensor &, int, bool>(
                 &quiver::ShardTensor::append),
             py::call_guard<py::gil_scoped_release>())
        .def("append",
             py::overload_cast<quiver::ShardTensorItem>(
                 &quiver::ShardTensor::append),
             py::call_guard<py::gil_scoped_release>())
        .def("share_ipc", &quiver::ShardTensor::share_ipc,
             py::call_guard<py::gil_scoped_release>());
}