"""
TSP-20 快速调参：网格搜索 ucb_c × ucb_tau（DeepACO，固定前 100 张测试地图）。
分段记录与 test.ipynb 一致：[1,10,...,100]，每组打印 total duration 与各 T 的平均代价；无 debug 快照。
控制台与脚本同目录下 tune_tsp20.log 同步输出；Tee 在每次 write 后 flush，避免首轮结果滞留在缓冲里。
"""

import itertools
import sys
import time
from pathlib import Path

import torch

LOG_PATH = "tune_tsp20.log"


class _Tee:
    """将 print 同时写到控制台与 log 文件。"""

    __slots__ = ("_streams",)

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str):
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

from net import Net
from aco import ACO
from utils import load_test_dataset

torch.manual_seed(12345)

EPS = 1e-10
device = "cuda:0" if torch.cuda.is_available() else "cpu"

N_ANTS = 20
N_INSTANCES = 100
T_ACO = [1, 10, 20, 30, 40, 50, 100]

# tsp20：与 presets['tsp20']['schedule'] 一致，除了下面对 c、tau 的扫描
FIXED_SCHEDULE = dict(
    ucb_schedule="exp",
    ucb_c_min=0.05,
    stagnation_k=10,
    boost_gamma=2.0,
    ucb_c_max=2.0,
    boost_cooldown=5,
)

GRID_UCB_C = (0.3, 0.5, 0.8)
GRID_TAU = (10.0, 20.0, 50.0)


@torch.no_grad()
def infer_instance(
    model,
    pyg_data,
    distances,
    t_aco_diff,
    ucb_c: float,
    ucb_tau: float,
    schedule_kw: dict,
):
    model.eval()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    aco = ACO(
        n_ants=N_ANTS,
        heuristic=heu_mat,
        distances=distances,
        device=device,
        ucb_c=ucb_c,
        ucb_tau=ucb_tau,
        **schedule_kw,
    )
    results = torch.zeros(size=(len(t_aco_diff),), device=device)
    for i, t in enumerate(t_aco_diff):
        results[i] = aco.run(t)
    return results


@torch.no_grad()
def test_dataset(net, dataset, t_aco, ucb_c: float, ucb_tau: float, schedule_kw: dict):
    _t_aco = [0] + list(t_aco)
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    start = time.time()
    for pyg_data, distances in dataset:
        sum_results += infer_instance(
            net, pyg_data, distances, t_aco_diff, ucb_c, ucb_tau, schedule_kw
        )
    end = time.time()
    return sum_results / len(dataset), end - start


def run_tuning_grid(net=None, test_list=None, print_baseline: bool = False):
    """网格搜索 ucb_c × ucb_tau；可由 notebook 导入复用。

    返回 tune_results[(uc, tau)] = (avg_tensor, duration)，以及 test_list、net。
    CLI 默认 print_baseline=False；notebook 可设 True 以打印 ucb_c=0 参考。
    """
    n_node = 20
    k_sparse = 10

    if test_list is None:
        test_list = load_test_dataset(n_node, k_sparse, device)[:N_INSTANCES]

    if net is None:
        net = Net().to(device)
        net.load_state_dict(
            torch.load(f"../pretrained/tsp/tsp{n_node}.pt", map_location=device)
        )

    sk = {
        k: v
        for k, v in FIXED_SCHEDULE.items()
        if k not in ("ucb_c", "ucb_tau")
    }

    if print_baseline:
        avg_b, dur_b = test_dataset(net, test_list, T_ACO, 0.0, 1.0, sk)
        print("[baseline ucb_c=0]")
        print("total duration: ", dur_b)
        for i, t in enumerate(T_ACO):
            print(f"T={t}, average cost is {float(avg_b[i])}.")
        print()

    print(
        f"tsp20 DeepACO grid | maps={len(test_list)} ants={N_ANTS} "
        f"t_aco={T_ACO} device={device} | log={LOG_PATH}"
    )
    tune_results = {}
    for uc, tau in itertools.product(GRID_UCB_C, GRID_TAU):
        avg_aco_best, duration = test_dataset(net, test_list, T_ACO, uc, tau, sk)
        tune_results[(uc, tau)] = (avg_aco_best, duration)
        print(f"\nucb_c={uc:g} ucb_tau={tau:g}")
        print("total duration: ", duration)
        for i, t in enumerate(T_ACO):
            print(f"T={t}, average cost is {float(avg_aco_best[i])}.")
    return tune_results, test_list, net


def main():
    run_tuning_grid(print_baseline=False)


if __name__ == "__main__":
    with open(LOG_PATH, "w", encoding="utf-8") as log_f:
        old_out = sys.stdout
        sys.stdout = _Tee(old_out, log_f)
        try:
            main()
        finally:
            sys.stdout = old_out
