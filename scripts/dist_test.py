import os
import socket

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    value = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(value)

    print(
        f"host={socket.gethostname()} "
        f"rank={rank}/{world_size} "
        f"gpu={torch.cuda.get_device_name(local_rank)} "
        f"result={value.item()}",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
