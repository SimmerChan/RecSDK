import random
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict

import pytest
import torch

BLOCK_HEIGHT = 256


@dataclass
class BackgroundInfo:
    seq_len_q: int
    seq_len_k: int
    num_target: int
    num_context: int
    num_history: int
    target_group_size: int
    block_h: int
    block_w: int

    def __post_init__(self):
        self.seq_blk_len_q: int = (self.seq_len_q + self.block_h - 1) // self.block_h
        self.seq_blk_len_k: int = (self.seq_len_k + self.block_h - 1) // self.block_h

        self.block_size: int = self.block_h * self.block_w

    def get_all_blocks(self) -> Iterable[Tuple[int, int]]:
        for blk_q in range(self.seq_blk_len_q):
            for blk_k in range(self.seq_blk_len_k):
                yield blk_q, blk_k

    def get_block_range(self, blk_q: int, blk_k: int, check_border=True) -> Tuple[int, int, int, int]:
        blk_q_start = blk_q * self.block_h
        blk_k_start = blk_k * self.block_w
        blk_q_end = blk_q_start + self.block_h
        blk_k_end = blk_k_start + self.block_w
        if check_border:
            blk_q_end = min(blk_q_end, self.seq_len_q)
            blk_k_end = min(blk_k_end, self.seq_len_k)
        return blk_q_start, blk_q_end, blk_k_start, blk_k_end

    def __str__(self):
        return (f"attnscore({self.seq_len_q}x{self.seq_len_k})"
                f"block({self.block_h}x{self.block_w})"
                f"ctx({self.num_context})"
                f"hist({self.num_history})"
                f"target({self.num_target})"
                f"tgsize({self.target_group_size})")


class BlockMaskGenerator:
    def __init__(self, info: BackgroundInfo):
        self.info = info

    def gen_context_mask(self, mask: torch.Tensor, blk_q: int, blk_k: int) -> torch.Tensor:
        cmask_width = self.info.seq_len_k - self.info.num_target
        local_mask_width = min(self.info.block_w, cmask_width - blk_k * self.info.block_w)
        local_mask_height = self.info.num_context  # st: num_context <= block_height
        for i in range(local_mask_height):
            duplicate(mask[i, :], 1, local_mask_width)
        return mask

    def gen_causal_mask(self, mask: torch.Tensor, blk_q: int, blk_k: int) -> torch.Tensor:
        for i in range(self.info.block_w):
            duplicate(mask[i, :], 1, i + 1)
        return mask

    def gen_target_mask(self, mask: torch.Tensor, blk_q: int, blk_k: int) -> torch.Tensor:
        block_top, block_bottom, block_left, block_right = self.info.get_block_range(blk_q, blk_k, check_border=False)
        tbase = self.info.seq_len_q - self.info.num_target
        # 当前block中，tmask的起始列
        if block_left > tbase:
            mask_start_in_score = block_left
        else:
            mask_start_in_score = tbase
        mask_start_in_block = mask_start_in_score - block_left
        valid_height = min(block_bottom, self.info.seq_len_q) - block_top
        for line in range(valid_height):
            line_in_score = line + block_top
            tri_num = (line_in_score - tbase) // self.info.target_group_size  # 本行上方有tri_num个完整的三角形
            tri_line_in_score = tri_num * self.info.target_group_size + tbase  # 本行上方最近的三角形的底边、列
            if tri_num < 1 or tri_line_in_score < block_left:  # 无挖空区域/挖空区域在block左边
                continue
            mask_len = min(block_right, tri_line_in_score) - mask_start_in_block
            duplicate(mask[line, mask_start_in_block:], 0, mask_len)
        return mask

    def gen_empty_mask(self) -> torch.Tensor:
        return torch.zeros((self.info.block_h, self.info.block_w))

    def get_mask_checklist(self, blk_q: int, blk_k: int) -> Dict[str, bool]:
        mask_checklist = dict()
        # enable context mask
        cmask_height, cmask_width = self.info.num_context, self.info.seq_len_k - self.info.num_target
        mask_checklist['context'] = bool(
            blk_q * self.info.block_h < cmask_height and blk_k * self.info.block_w < cmask_width)
        # enable causal mask
        mask_checklist['causal'] = bool(blk_q == blk_k)
        # enable target mask
        tmask_start_line = self.info.seq_len_q - self.info.num_target
        tmask_start_blk = tmask_start_line // self.info.block_h
        mask_checklist['target'] = bool(tmask_start_blk <= blk_k <= blk_q)
        return mask_checklist

    def __call__(self, blk_q: int, blk_k: int) -> torch.Tensor:
        mask = self.gen_empty_mask()
        mask_checklist = self.get_mask_checklist(blk_q, blk_k)
        if mask_checklist['context']:
            mask = self.gen_context_mask(mask, blk_q, blk_k)
        if mask_checklist['causal']:
            mask = self.gen_causal_mask(mask, blk_q, blk_k)
        if mask_checklist['target']:
            if not mask_checklist['causal']:
                duplicate(mask, 1, self.info.block_size)  # mask = torch.ones_like(mask)
            mask = self.gen_target_mask(mask, blk_q, blk_k)
        return mask.bool().int()


def duplicate(lt: torch.Tensor, val: int, cnt: int):
    lt[:cnt] = val


def gen_mask(info: BackgroundInfo):
    mask = torch.zeros(info.seq_len_q, info.seq_len_k, dtype=torch.uint8)
    blk_mask_gen = BlockMaskGenerator(info)
    for blk_q, blk_k in info.get_all_blocks():
        x1, x2, y1, y2 = info.get_block_range(blk_q, blk_k)
        dx, dy = x2 - x1, y2 - y1
        block_mask = blk_mask_gen(blk_q, blk_k)
        mask[x1: x2, y1: y2] = block_mask[:dx, :dy]
    return mask


def save_mask_to_file(mask: torch.Tensor, dst_path: str = "mask.txt"):
    with open(dst_path, 'w') as f:
        for line in mask.tolist():
            string = ",".join(map(str, line)) + "\n"
            f.write(string)


@pytest.mark.parametrize("num_context", [6])
@pytest.mark.parametrize("num_history_range", [[100, 100]])
@pytest.mark.parametrize("num_target", [2, 16, 30, 512])
@pytest.mark.parametrize("target_group_size", [1, 3])
def test_mask_gen(num_context, num_history_range, num_target, target_group_size):
    num_history = random.randint(*num_history_range)
    seq_len = num_context + num_history + num_target
    info = BackgroundInfo(seq_len_q=seq_len,
                          seq_len_k=seq_len,
                          num_context=num_context,
                          num_history=num_history,
                          num_target=num_target,
                          target_group_size=target_group_size,
                          block_h=BLOCK_HEIGHT,
                          block_w=BLOCK_HEIGHT)
    mask = gen_mask(info)
    save_mask_to_file(mask, str(info) + "_mask.txt")
