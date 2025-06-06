from gpu import barrier
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.host._nvidia_cuda import TMADescriptor, create_tma_descriptor
from gpu.id import block_idx, thread_idx
from gpu.memory import (
    _GPUAddressSpace,
    cp_async_bulk_tensor_shared_cluster_global,
    cp_async_bulk_tensor_global_shared_cta,
    tma_store_fence,
)
from gpu.sync import (
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
    mbarrier_try_wait_parity_shared,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from memory import UnsafePointer, stack_allocation

from utils.index import Index
from utils.static_tuple import StaticTuple

alias GMEM_HEIGHT = 64
alias GMEM_WIDTH = 64
alias BLOCK_SIZE = 32
alias SMEM_HEIGHT = BLOCK_SIZE
alias SMEM_WIDTH = BLOCK_SIZE


@__llvm_arg_metadata(descriptor, `nvvm.grid_constant`)
@__llvm_arg_metadata(descriptor_tr, `nvvm.grid_constant`)
fn kernel_copy_async_tma[
    block_size: Int
](descriptor: TMADescriptor, descriptor_tr: TMADescriptor):
    var shmem = stack_allocation[
        block_size * block_size,
        DType.float32,
        alignment=1024,
        address_space = _GPUAddressSpace.SHARED,
    ]()
    var shmem_tr = stack_allocation[
        block_size * block_size,
        DType.float32,
        alignment=1024,
        address_space = _GPUAddressSpace.SHARED,
    ]()
    var mbar = stack_allocation[
        1, Int64, address_space = _GPUAddressSpace.SHARED
    ]()
    var descriptor_ptr = UnsafePointer(to=descriptor).bitcast[NoneType]()
    var descriptor_tr_ptr = UnsafePointer(to=descriptor_tr).bitcast[NoneType]()

    x = block_idx.x * block_size
    y = block_idx.y * block_size

    col = thread_idx.x % block_size
    row = thread_idx.x // block_size

    # LOAD
    if thread_idx.x == 0:
        mbarrier_init(mbar, 1)
        mbarrier_arrive_expect_tx_shared(mbar, block_size * block_size * 4)
        cp_async_bulk_tensor_shared_cluster_global(
            shmem, descriptor_ptr, mbar, Index(x, y)
        )
    barrier()
    mbarrier_try_wait_parity_shared(mbar, 0, 10000000)

    # COMPUTE
    shmem_tr[col * block_size + row] = shmem[row * block_size + col]

    # FENCE
    barrier()
    tma_store_fence()

    # STORE
    if thread_idx.x == 0:
        cp_async_bulk_tensor_global_shared_cta(
            shmem_tr, descriptor_tr_ptr, Index(y, x)
        )
        cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


def test_tma_tile_copy(ctx: DeviceContext):
    print("== test_tma_tile_copy")
    var gmem_host = UnsafePointer[Float32].alloc(GMEM_HEIGHT * GMEM_WIDTH)
    var gmem_tr_host = UnsafePointer[Float32].alloc(GMEM_HEIGHT * GMEM_WIDTH)
    for i in range(GMEM_HEIGHT * GMEM_WIDTH):
        gmem_host[i] = i % 32

    print("Initial matrix:")
    for row in range(GMEM_HEIGHT):
        for col in range(GMEM_WIDTH):
            idx = row * GMEM_WIDTH + col
            print(String(Int32(gmem_host[idx])).ljust(2), end=" ")
        print()
    print()

    var gmem_dev = ctx.enqueue_create_buffer[DType.float32](
        GMEM_HEIGHT * GMEM_WIDTH
    )
    var gmem_tr_dev = ctx.enqueue_create_buffer[DType.float32](
        GMEM_HEIGHT * GMEM_WIDTH
    )

    ctx.enqueue_copy(gmem_dev, gmem_host)

    var descriptor = create_tma_descriptor[DType.float32, 2](
        gmem_dev,
        (GMEM_HEIGHT, GMEM_WIDTH),
        (GMEM_WIDTH, 1),
        (SMEM_HEIGHT, SMEM_WIDTH),
    )
    var descriptor_tr = create_tma_descriptor[DType.float32, 2](
        gmem_tr_dev,
        (GMEM_WIDTH, GMEM_HEIGHT),
        (GMEM_HEIGHT, 1),
        (SMEM_WIDTH, SMEM_HEIGHT),
    )

    ctx.enqueue_function[kernel_copy_async_tma[BLOCK_SIZE]](
        descriptor,
        descriptor_tr,
        grid_dim=(GMEM_HEIGHT // SMEM_HEIGHT, GMEM_WIDTH // SMEM_WIDTH, 1),
        block_dim=(SMEM_HEIGHT * SMEM_WIDTH, 1, 1),
    )
    ctx.enqueue_copy(gmem_tr_host, gmem_tr_dev)
    ctx.synchronize()

    print("Final matrix:")
    for row in range(GMEM_HEIGHT):
        for col in range(GMEM_WIDTH):
            idx = row * GMEM_WIDTH + col
            print(String(Int32(gmem_tr_host[idx])).ljust(2), end=" ")
        print()
    print()
    gmem_host.free()
    gmem_tr_host.free()


def main():
    with DeviceContext() as ctx:
        test_tma_tile_copy(ctx)
