from multiprocessing.reduction import ForkingPickler
import quiver

"""
这段代码使用Python的multiprocessing.reduction模块为quiver库中的Feature类和两个sampler类(GraphSageSampler和MixedGraphSageSampler)注册了自定义的序列化和反序列化方法。

其中reduce_feature函数将Feature对象转换为可在进程间传输的句柄，然后在rebuild_feature函数中重新构建该对象。同样地，reduce_pyg_sampler函数和rebuild_pyg_sampler函数将sampler对象转换为可传输的句柄，然后在另一个进程中重新构建该对象。最后，init_reductions函数将这些自定义的序列化和反序列化方法注册到了ForkingPickler中。

这个过程中，序列化和反序列化方法主要是通过Feature类和sampler类的share_ipc()和lazy_from_ipc_handle()方法实现的。share_ipc()方法将对象转换为可在进程间传输的共享内存句柄，lazy_from_ipc_handle()方法则从共享内存句柄中重新构建对象。
"""


def rebuild_feature(ipc_handle):

    feature = quiver.Feature.lazy_from_ipc_handle(ipc_handle)
    return feature


def reduce_feature(feature):

    ipc_handle = feature.share_ipc()
    return (rebuild_feature, (ipc_handle, ))


def rebuild_pyg_sampler(cls, ipc_handle):
    sampler = cls.lazy_from_ipc_handle(ipc_handle)
    return sampler


def reduce_pyg_sampler(sampler):
    ipc_handle = sampler.share_ipc()
    return (rebuild_pyg_sampler, (
        type(sampler),
        ipc_handle,
    ))


def init_reductions():
    ForkingPickler.register(quiver.Feature, reduce_feature)
    ForkingPickler.register(quiver.pyg.GraphSageSampler, reduce_pyg_sampler)
    ForkingPickler.register(
        quiver.pyg.MixedGraphSageSampler, reduce_pyg_sampler)
