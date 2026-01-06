from torch.multiprocessing import Queue, Value, Barrier, Array
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from ctypes import c_bool, c_int
import torch

from RapidIn.calc_inner import grad_z
from RapidIn.data_loader import TrainDataset, TestDataset, get_tokenizer, get_model
from RapidIn.data_loader import read_data
from RapidIn.influence_function import calc_s_test_single
from RapidIn.utils import save_json, display_progress, load_json
from RapidIn.RapidGrad import RapidGrad

import numpy as np
import time
from pathlib import Path
from copy import copy
import os
import gc
import logging
import re


MAX_CAPACITY = 2048
MAX_DATASET_SIZE = int(1e8)


def _redact_sensitive_text(s: str) -> str:
    """
    Redact common user-identity-bearing absolute paths in logs / saved metadata.
    This is a best-effort sanitizer (does not change runtime behavior).
    """
    if s is None:
        return s
    # Linux home
    s = re.sub(r"/home/[^/\s]+", "/home/USER", s)
    # macOS home
    s = re.sub(r"/Users/[^/\s]+", "/Users/USER", s)
    return s


def MP_run_calc_infulence_function(rank, world_size, process_id, config, mp_engine, restart=False):
    model = None
    tokenizer = None

    print(f"rank: {rank}, world_size: {world_size}")
    tokenizer = get_tokenizer(config.model, device_map=f"cuda:{rank}")
    print(f"CUDA {rank}: Tokenizer loaded!")

    train_dataset = TrainDataset(
        config.data.train_data_path,
        tokenizer,
        shuffle=False,
        begin_id=config.data.begin_id,
        end_id=config.data.end_id,
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"CUDA {rank}: Training DataLoader loaded!")

    test_dataset = TestDataset(config.data.test_data_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"CUDA {rank}: Testing DataLoader loaded!")

    train_dataset_size = len(train_dataset)
    with mp_engine.train_dataset_size.get_lock():
        mp_engine.train_dataset_size.value = train_dataset_size
    with mp_engine.test_dataset_size.get_lock():
        mp_engine.test_dataset_size.value = len(test_dataset)

    grad_reshape = True
    oporp_eng = None
    if config.influence.RapidGrad.enable:
        oporp_eng = RapidGrad(config, f"cuda:{rank}")
        grad_reshape = False

    def get_s_test_vec_list(model):
        s_test_vec_list = []
        test_dataset_size_local = len(test_dataset)
        for i in range(test_dataset_size_local):
            z_test, t_test, input_len = test_dataset[i]

            if config.influence.infl_method == "IF":
                s_test_vec = calc_s_test_single(
                    model,
                    z_test,
                    t_test,
                    input_len,
                    train_dataset,
                    gpu=rank,
                    recursion_depth=config.influence.IF.recursion_depth,
                    scale=config.influence.IF.scale,
                    r=config.influence.IF.r_averaging,
                    need_reshape=grad_reshape,
                )
            else:
                s_test_vec = grad_z(
                    z_test,
                    t_test,
                    input_len,
                    model,
                    gpu=rank,
                    need_reshape=grad_reshape,
                    use_deepspeed=config.influence.deepspeed.enable,
                )

            if config.influence.RapidGrad.enable and isinstance(config.influence.RapidGrad.RapidGrad_K, int):
                s_test_vec = oporp_eng(s_test_vec, config.influence.RapidGrad.RapidGrad_K)

            if config.influence.offload_test_grad is True or config.influence.calculate_infl_in_gpu is False:
                s_test_vec = s_test_vec.cpu()

            s_test_vec_list.append(s_test_vec)
            display_progress("Calc. s-test vector: ", i, test_dataset_size_local, cur_time=time.time())

        return s_test_vec_list

    with mp_engine.gpu_locks[rank].get_lock():
        if config.influence.deepspeed.enable is False:
            model = get_model(config.model, tokenizer, device_map=f"cuda:{rank}")

            # Quantized models (e.g., 4-bit) usually have fixed dtype; do not blindly call .half()
            can_half = True
            qc = getattr(getattr(model, "config", None), "quantization_config", None)
            if qc is not None or getattr(model, "is_loaded_in_4bit", False):
                can_half = False
            if can_half:
                model.half()

            # Key point 1: enable requires_grad only for floating-point parameters (skip quantized weights)
            for _, p in model.named_parameters():
                if hasattr(p, "is_floating_point") and p.is_floating_point():
                    p.requires_grad_(True)
                else:
                    # Non-floating tensors are typically quantized weights / buffers; keep them frozen.
                    p.requires_grad_(False)

            # Key point 2: initialize calc_inner's parameter cache (so grad_z can use autograd.grad path)
            from RapidIn.calc_inner import get_params
            _ = get_params(model, create_if_not_exist=True)

            model.eval()
            print(f"CUDA {rank}: Model loaded!")
        else:
            deepspeed_config = load_json(config.influence.deepspeed.config_path)
            model = get_model(config["model"], tokenizer)

            import deepspeed
            model, _, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config=deepspeed_config,
            )
            model.optimizer.override_loss_scale(1)

        print(f"CUDA {rank}: Model loaded!")

        s_test_vec_list = []
        if config.influence.skip_test is not True:
            s_test_vec_list = get_s_test_vec_list(model)

        if config.influence.delete_model:
            del model, oporp_eng
            gc.collect()
            torch.cuda.empty_cache()

        with mp_engine.gpu_locks_num.get_lock():
            display_progress(
                "Model Loading: ",
                mp_engine.gpu_locks_num.value,
                config.influence.n_threads * world_size,
                cur_time=time.time(),
            )
            mp_engine.gpu_locks_num.value += 1

    if restart is False:
        mp_engine.start_barrier.wait()

    idx = 0
    while True:
        cal_word_infl = -1

        while True:
            with mp_engine.train_idx.get_lock(), mp_engine.finished_idx.get_lock(), mp_engine.cal_word_infl.get_lock():
                idx = mp_engine.train_idx.value
                mp_engine.train_idx.value = (mp_engine.train_idx.value + 1) % train_dataset_size
                if mp_engine.finished_idx[idx] is False:
                    mp_engine.finished_idx[idx] = True
                    cal_word_infl = mp_engine.cal_word_infl[idx]
                    break
            time.sleep(0.002)

        if idx >= train_dataset_size:
            break

        try:
            z, t, input_len, real_id = train_loader.dataset[idx]

            if cal_word_infl < 0:
                influence = None
                s_test_vec_t = None
                grad_z_vec = None
                grad_path_name = None

                if config.influence.grads_path is not None and len(config.influence.grads_path) != 0:
                    if config.influence.RapidGrad.enable and isinstance(config.influence.RapidGrad.RapidGrad_K, int):
                        grad_path_name = (
                            config.influence.grads_path
                            + f"/train_grad_K{config.influence.RapidGrad.RapidGrad_K}_{real_id:08d}.pt"
                        )
                    else:
                        grad_path_name = config.influence.grads_path + f"/train_grad_{real_id:08d}.pt"

                if config.influence.load_from_grads_path:
                    if config.influence.grads_path is None:
                        raise AssertionError("Load from grads path, but did not provide grads_path.")
                    grad_z_vec = torch.load(grad_path_name, map_location=f"cuda:{rank}")

                if grad_z_vec is None:
                    grad_z_vec = grad_z(
                        z,
                        t,
                        input_len,
                        model,
                        gpu=rank,
                        need_reshape=grad_reshape,
                        use_deepspeed=config.influence.deepspeed.enable,
                    )

                    if config.influence.RapidGrad.enable and isinstance(config.influence.RapidGrad.RapidGrad_K, int):
                        grad_z_vec = oporp_eng(grad_z_vec, config.influence.RapidGrad.RapidGrad_K)

                    if config.influence.RapidGrad.enable and isinstance(config.influence.RapidGrad.RapidGrad_K, list):
                        grad_z_vec_list = oporp_eng(grad_z_vec, config.influence.RapidGrad.RapidGrad_K)
                        for i_k, (path, k) in enumerate(
                            zip(config.influence.RapidGrad.multi_k_save_path_list, config.influence.RapidGrad.RapidGrad_K)
                        ):
                            torch.save(grad_z_vec_list[i_k], path + f"/train_grad_K{k}_{real_id:08d}.pt")

                if config.influence.save_to_grads_path:
                    if config.influence.grads_path is None:
                        raise AssertionError("Save to grads path, but did not provide grads_path.")
                    torch.save(grad_z_vec, grad_path_name)

                if config.influence.skip_influence or config.influence.skip_test:
                    for i_test in range(max(len(test_dataset), 1)):
                        mp_engine.result_q.put((i_test, idx, real_id, 0), block=True, timeout=None)
                    continue

                if config.influence.calculate_infl_in_gpu is False or config.influence.offload_train_grad is True:
                    grad_z_vec = grad_z_vec.cpu()

                for i_test in range(len(test_dataset)):
                    s_test_vec_t = s_test_vec_list[i_test]

                    if (not config.influence.deepspeed.enable) and config.influence.calculate_infl_in_gpu:
                        if config.influence.offload_test_grad is True:
                            s_test_vec_t = s_test_vec_t.to(rank)

                    if config.influence.deepspeed.enable and config.influence.calculate_infl_in_gpu:
                        influence = None
                        for k in range(grad_z_vec.shape[0]):
                            x = grad_z_vec[k]
                            y = s_test_vec_t[k]
                            if k == 0:
                                influence = torch.sum(x.to(rank) * y.to(rank))
                            else:
                                influence = influence + torch.sum(x.to(rank) * y.to(rank))
                            x = None
                            y = None
                    else:
                        influence = torch.sum(grad_z_vec * s_test_vec_t)

                    if config.influence.calculate_infl_in_gpu is True:
                        influence = influence.cpu()
                    influence = influence.numpy()

                    if influence != influence:  # NaN check
                        raise Exception("Got unexpected NaN influence!")
                    mp_engine.result_q.put((i_test, idx, real_id, influence), block=True, timeout=None)

            else:
                _, words_influence = grad_z(
                    z,
                    t,
                    input_len,
                    model,
                    gpu=rank,
                    return_words_loss=True,
                    s_test_vec=s_test_vec_list[cal_word_infl],
                )
                mp_engine.result_q.put((cal_word_infl, idx, real_id, words_influence), block=True, timeout=None)

        except Exception as e:
            with mp_engine.finished_idx.get_lock():
                mp_engine.finished_idx[idx] = False
                print(e)
            raise e


def MP_run_get_result(config, mp_engine):
    train_dataset_size = 0
    test_dataset_size = 0
    while True:
        with mp_engine.train_dataset_size.get_lock():
            train_dataset_size = mp_engine.train_dataset_size.value
        with mp_engine.test_dataset_size.get_lock():
            test_dataset_size = mp_engine.test_dataset_size.value
        if train_dataset_size != 0 and (test_dataset_size != 0 or config.influence.skip_test):
            break
        time.sleep(1)

    print(f"train_dataset_size: {train_dataset_size}, test_dataset_size: {test_dataset_size}")

    with mp_engine.train_dataset_size.get_lock(), mp_engine.finished_idx.get_lock():
        if mp_engine.train_dataset_size.value > len(mp_engine.finished_idx):
            raise Exception("Size of train dataset larger than MAX_DATASET_SIZE")

    outdir = Path(config.influence.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    influences_path = outdir.joinpath(f"influence_results_{train_dataset_size}.json")
    influences_path = save_json({}, influences_path, unique_fn_if_exists=True)

    mp_engine.start_barrier.wait()

    test_data_dicts = []
    if not config.influence.skip_test:
        test_data_dicts = read_data(config.data.test_data_path)

    influences = {}
    # IMPORTANT: config may contain absolute paths / usernames. Store a redacted version only.
    influences["config"] = _redact_sensitive_text(str(config))

    for k in range(test_dataset_size):
        influences[k] = {}
        influences[k]["test_data"] = test_data_dicts[k]

    infl_list = [[0 for _ in range(train_dataset_size)] for _ in range(max(test_dataset_size, 1))]
    real_id2shuffled_id = {}
    shuffled_id2real_id = {}

    total_size = max(test_dataset_size, 1) * train_dataset_size

    i = 0
    while True:
        try:
            result_item = mp_engine.result_q.get(block=True)
        except Exception:
            print("Cal Influence Function Finished!")
            break

        if result_item is None:
            save_json(influences, influences_path, overwrite_if_exists=True)
            raise Exception("Got unexpected result from queue.")

        test_id, shuffled_id, real_id, influence = result_item
        if influence != influence:  # NaN check
            raise Exception("Got unexpected NaN influence!")

        infl_list[test_id][shuffled_id] = influence
        real_id2shuffled_id[real_id] = shuffled_id
        shuffled_id2real_id[shuffled_id] = real_id

        with mp_engine.finished_idx.get_lock():
            # This is needed because we retrieve items by shuffled_id during computation.
            mp_engine.finished_idx[shuffled_id] = True

        display_progress("Calc. influence function: ", i, total_size, cur_time=time.time())

        topk_num = int(config.influence.top_k)

        if (not config.influence.skip_influence) and ((i + 1) % (total_size // 50) == 0 or i == total_size - 1):
            for j in range(test_dataset_size):
                harmful_shuffle_ids = np.argsort(infl_list[j]).tolist()
                harmful = [shuffled_id2real_id[x] for x in harmful_shuffle_ids if x in shuffled_id2real_id.keys()]
                helpful = harmful[::-1]

                infl = [x.tolist() if not isinstance(x, int) else x for x in infl_list[j]]
                helpful_topk = helpful[:topk_num]
                harmful_topk = harmful[:topk_num]

                influences[j]["helpful"] = copy(helpful_topk)
                influences[j]["helpful_infl"] = copy([infl[x] for x in harmful_shuffle_ids[-topk_num:][::-1]])
                influences[j]["harmful"] = copy(harmful_topk)
                influences[j]["harmful_infl"] = copy([infl[x] for x in harmful_shuffle_ids[:topk_num]])

            influences["finished_cnt"] = f"{i + 1}/{total_size}"
            influences_path = save_json(influences, influences_path, overwrite_if_exists=True)

        i += 1
        if i >= total_size:
            finished = True
            with mp_engine.finished_idx.get_lock():
                for idx in range(train_dataset_size):
                    if mp_engine.finished_idx[idx] is False:
                        print("Warning: i >= total_size, but it has not finished!")
                        finished = False
                        break
            if finished is True:
                break

    if config.influence.cal_words_infl is True:
        # Calculate word-level influence
        for j in range(test_dataset_size):
            word_infl_dict = {}

            infl_num = len(set(influences[j]["helpful"] + influences[j]["harmful"]))
            with mp_engine.train_idx.get_lock(), mp_engine.finished_idx.get_lock(), mp_engine.cal_word_infl.get_lock():
                for x in influences[j]["helpful"]:
                    mp_engine.cal_word_infl[real_id2shuffled_id[x]] = j
                    mp_engine.finished_idx[real_id2shuffled_id[x]] = False
                for x in influences[j]["harmful"]:
                    mp_engine.cal_word_infl[real_id2shuffled_id[x]] = j
                    mp_engine.finished_idx[real_id2shuffled_id[x]] = False

            i_word = 0
            while True:
                try:
                    result_item = mp_engine.result_q.get(block=True, timeout=300)
                except Exception as e:
                    print(e)
                    break

                if result_item is None:
                    save_json(influences, influences_path, overwrite_if_exists=True)
                    raise Exception("Got unexpected result from queue.")

                test_id, shuffled_id, real_id, word_influence = result_item

                with mp_engine.finished_idx.get_lock(), mp_engine.cal_word_infl.get_lock():
                    mp_engine.finished_idx[shuffled_id] = True
                    mp_engine.cal_word_infl[shuffled_id] = -1

                word_infl_dict[real_id] = (
                    word_influence.tolist() if not isinstance(word_influence, list) else word_influence
                )
                display_progress(
                    f"Calc. word influence for test {j + 1}/{test_dataset_size}",
                    i_word,
                    infl_num,
                    cur_time=time.time(),
                )

                i_word += 1
                if i_word >= infl_num:
                    finished = True
                    with mp_engine.finished_idx.get_lock():
                        for idx in range(train_dataset_size):
                            if mp_engine.finished_idx[idx] is False:
                                print("Warning: i >= total_size, but it has not finished!")
                                finished = False
                                break
                    if finished is True:
                        break

            influences[j]["word_influence"] = word_infl_dict
            try:
                influences_path = save_json(influences, influences_path, overwrite_if_exists=True)
            except Exception as e:
                print(influences)
                print(e)

    influences_path = save_json(influences, influences_path, overwrite_if_exists=True)
    print(influences_path)
    return influences


class MPEngine:
    def __init__(self, world_size):
        self.result_q = Queue(maxsize=MAX_CAPACITY)

        self.train_idx = Value(c_int, 0)

        self.start_barrier = Barrier(world_size + 1)
        self.finished_a_test = Value(c_int, 0)
        self.cur_processes_num = Value(c_int, 0)

        self.gpu_locks = [Value(c_int, 0) for _ in range(world_size)]
        self.gpu_locks_num = Value(c_int, 0)

        self.train_dataset_size = Value(c_int, 0)
        self.test_dataset_size = Value(c_int, 0)

        self.finished_idx = Array(c_bool, [False for _ in range(MAX_DATASET_SIZE)])

        # -1 means do not compute word-level influence.
        # >= 0 means compute word-level influence for the given test sample index.
        self.cal_word_infl = Array(c_int, [-1 for _ in range(MAX_DATASET_SIZE)])

    def action_finished_a_test(self):
        with self.train_idx.get_lock():
            self.train_idx.value = 0


def calc_infl_mp(config):
    gpu_num = torch.cuda.device_count()
    print(f"{gpu_num} GPUs available!")

    threads_per_gpu = int(config.influence.n_threads)

    if config.influence.grads_path is not None and len(config.influence.grads_path) != 0:
        os.makedirs(config.influence.grads_path, exist_ok=True)

        if config.influence.RapidGrad.enable and isinstance(config.influence.RapidGrad.RapidGrad_K, list):
            config.influence.RapidGrad.multi_k_save_path_list = []
            for k in config.influence.RapidGrad.RapidGrad_K:
                path = config.influence.grads_path + f"/K{k}"
                config.influence.RapidGrad.multi_k_save_path_list.append(path)
                os.makedirs(path, exist_ok=True)

    num_processing = gpu_num * threads_per_gpu
    mp_engine = MPEngine(num_processing)

    mp_handler = []
    mp_args = []
    print(f"GPU Num: {gpu_num}, Threads per GPU: {threads_per_gpu}")

    for i in range(gpu_num):
        for j in range(threads_per_gpu):
            mp_handler.append(
                mp.Process(
                    target=MP_run_calc_infulence_function,
                    args=(i, gpu_num, i * threads_per_gpu + j, config, mp_engine),
                )
            )
            mp_args.append(mp_handler[-1]._args)

    mp_handler.append(mp.Process(target=MP_run_get_result, args=(config, mp_engine)))

    for x in mp_handler:
        x.start()

    while mp_handler[-1].is_alive():
        cur_processes_num = len([1 for x in mp_handler if x.is_alive()])
        if cur_processes_num < num_processing + 1:
            print(f"ready to restart processing, {cur_processes_num}/{num_processing}")
            for i, x in enumerate(mp_handler):
                # Only restart worker processes that have corresponding mp_args
                if i < len(mp_args) and (x.is_alive() is not True):
                    print(f"start {mp_args[i]}")
                    mp_handler[i] = mp.Process(
                        target=MP_run_calc_infulence_function,
                        args=mp_args[i] + (True,),
                    )
                    mp_handler[i].start()
            continue

        with mp_engine.cur_processes_num.get_lock():
            mp_engine.cur_processes_num.value = cur_processes_num
        time.sleep(1)

    for x in mp_handler:
        x.terminate()