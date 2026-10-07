"""
TSP-20 测试集消融：去掉停滞触发的 boost（卡死跳跃），仅保留指数退火 ucb_schedule=exp。
网格与 tune_tsp20.py 相同：ucb_c × ucb_tau；分段 T 与原版一致；无 debug 快照。
控制台与同目录 tune_tsp20_ablation_decay_only.log 同步输出；Tee 每次 write 后 flush。
"""

import itertools
import sys
import time
from pathlib import Path

import torch

LOG_PATH =  "tune_tsp20_ablation_decay_only.log"


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

# 与 tune_tsp20 相同的退火侧；关闭 stagnation boost（aco 中 stagnation_k>0 且 boost_gamma>1 才生效）
FIXED_SCHEDULE = dict(
    ucb_schedule="exp",
    ucb_c_min=0.05,
    stagnation_k=0,
    boost_gamma=1.0,
    ucb_c_max=None,
    boost_cooldown=0,
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


def run_ablation_grid(net=None, test_list=None):
    """消融网格（decay_only，无 stagnation boost）；返回 ablation_results、test_list、net。"""
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

    print(
        f"ablation=decay_only no_stagnation_boost | tsp20 DeepACO grid | "
        f"maps={len(test_list)} ants={N_ANTS} t_aco={T_ACO} device={device} | log={LOG_PATH}"
    )
    ablation_results = {}
    for uc, tau in itertools.product(GRID_UCB_C, GRID_TAU):
        avg_aco_best, duration = test_dataset(net, test_list, T_ACO, uc, tau, sk)
        ablation_results[(uc, tau)] = (avg_aco_best, duration)
        print(f"\nucb_c={uc:g} ucb_tau={tau:g}")
        print("total duration: ", duration)
        for i, t in enumerate(T_ACO):
            print(f"T={t}, average cost is {float(avg_aco_best[i])}.")
    return ablation_results, test_list, net


def main():
    run_ablation_grid()


if __name__ == "__main__":
    with open(LOG_PATH, "w", encoding="utf-8") as log_f:
        old_out = sys.stdout
        sys.stdout = _Tee(old_out, log_f)
        try:
            main()
        finally:
            sys.stdout = old_out
