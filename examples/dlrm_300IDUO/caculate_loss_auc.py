import logging
from absl import app, flags

import torch_npu


import dlrm.scripts.utils as utils
import torch

FLAGS = flags.FLAGS

flags.DEFINE_string("path", None, "Path to the model")


loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def caculate(argv):
    path = FLAGS.path
    output_click_lst = torch.load(path)
    for i, output_click in enumerate(output_click_lst):
        y_true, y_score = output_click
        y_score = torch.sigmoid(y_score).float()
        auc = utils.roc_auc_score(y_true, y_score)
        loss = loss_fn(y_score, y_true).item()
        logging.info(f"step {i}, loss: {loss}, auc: {auc}")


if __name__ == "__main__":
    app.run(caculate)