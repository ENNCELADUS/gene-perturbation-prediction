"""The released stage-one regressor and SL classifier with resident-tensor data paths.

``fast_via_film`` and ``fast_pair_classifier`` subclass the released ``via_film`` and
``pair_classifier`` and override how rows reach the network and when the host reads
the loss; the released ``Framework`` drives them
(docs/specs/2026-10-08-cilantro-sl-reproduction-design.md §4.4):

- training batches come from ``fast_core.sliced_loader``: one gather per batch instead
  of 128 or 512 ``__getitem__`` calls and a collate, under the same ``RandomSampler``
  permutation drawn from the same RNG stream, so every batch is identical;
- ``train_nn`` is the released loop (same initialisation and RNG draws, loss,
  optimiser, epoch loop, model saving) with one change: the released loop read
  ``loss.item()`` after every batch, a GPU synchronisation that serialises the small
  kernels of this network; here the detached per-batch losses are kept and added to
  ``running_loss`` in the same order at the end of the epoch, which gives the same
  Python double (measured: 5.8 ms per batch with the sync, 2.6 ms without, on a
  shared GPU);
- the batch-size-1 test, calibration and conformal loops become loops over row slices
  of one resident GPU tensor (``fast_core.predict_rows``), with the forward pass still
  taken one row at a time in the released shape (2-D rows of one, or 1-D in
  ``extract_nn_second_last``), which is bitwise what the released loops compute;
  batched forward passes are not (max abs difference 6e-8 / 3.6e-7, measured), so
  they are not used; each released loop also drew one base seed from the global torch
  RNG when its DataLoader was iterated, and the next fold's network is initialised
  from that stream, so every replaced loop draws it too (``consume_loader_seed``);
- class probabilities are returned as one (rows, 2) float32 array instead of a list of
  one-row CUDA tensors, so ``compute_acc`` and the conformal p-values read the same
  values from an array; the conformal p-values are evaluated by one ``searchsorted``
  over all rows instead of per unique score through pandarallel, which gives the same
  numbers. The metrics pickle therefore holds arrays under ``probs`` and
  ``mondrian_probs_list`` where the released code pickled tensor lists;
- ``skip_fold`` replays one fold's draws from the global torch RNG without training,
  so each fold can run in its own process from the state a sequential run reaches;
- only the harness's call shapes are supported: no validation split, an empty
  Mondrian class map (plain split conformal), ``custom_func=None``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from net_layers import FiLMNet, SLNet
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from dataloader import dataset, dataset_classifier, dataset_unlabeled
from fast_core import consume_loader_seed, predict_rows, sliced_loader
from pair_classifier import ModelType, pair_classifier
from via_film import via_film


def as_matrix(probs) -> np.ndarray:
    """(rows, 2) float32 probabilities from either representation."""
    if isinstance(probs, np.ndarray):
        return probs
    return torch.cat([p.reshape(1, -1) for p in probs]).cpu().numpy()


def epoch_loss(losses: list[torch.Tensor]) -> float:
    """``running_loss`` as the released loops accumulate it, ``+= loss.item()`` per
    batch in order, from the kept per-batch losses."""
    running_loss = 0.0
    if losses:
        for value in torch.stack(losses).tolist():
            running_loss += value
    return running_loss


class fast_pair_classifier(pair_classifier):
    def train_nn(self, path):
        """The released train_nn (COMBINED model, no validation split) with the
        per-batch ``loss.item()`` deferred to the end of the epoch."""
        if self.model_type != ModelType.COMBINED:
            raise NotImplementedError("only the COMBINED model is supported")
        if self.validation:
            raise NotImplementedError("the harness trains without a validation split")
        print("Using normal net")
        net = SLNet()
        net.to(self.device)

        labels = torch.tensor(self.train_SL.tolist())
        print(labels)
        labels.to(self.device)
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        print(f"Class weights: {class_weights}")

        # Kept for the RNG draws the released code makes here.
        test_output = torch.randn(128, 2)
        test_labels = torch.randint(0, 2, (128,)).long()
        nn.CrossEntropyLoss()(test_output, test_labels)

        class_weights = class_weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        num_samples = len(self.dl_train)
        loss_arr = []
        for epoch in range(self.num_epochs):
            losses = []
            net.train()
            for inputs, labels in self.dl_train:
                inputs = inputs.float().to(self.device)
                labels = labels.squeeze().long().to(self.device)
                optimizer.zero_grad()
                outputs = net(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())
            mean_loss = epoch_loss(losses) / num_samples
            print(f"Epoch: {epoch} / {self.num_epochs}, Loss: {mean_loss}")
            loss_arr.append(mean_loss)
        print("Finished Training")
        self.loss_arr = loss_arr
        self.net = net
        if path is not None:
            print(f"Saving model at {path}")
            torch.save(net.state_dict(), path)
        else:
            print("No path supplied, model not saved")

    @staticmethod
    def skip_fold(num_epochs: int) -> None:
        """Draw from the global torch RNG exactly what one fold of ``run_cv`` draws
        (``train_nn`` then ``test_nn``), without training.

        The folds of ``run_cv`` share that stream: each fold's network is initialised
        from where the previous fold left it. A process that runs fold k alone replays
        folds 0..k-1 with this and then trains fold k from the state a sequential run
        reaches. Per fold the stream is drawn only by SLNet's initialisation (built on
        the CPU), the two dummy tensors, the two seeds each epoch's loader iterator
        takes (its base seed and the RandomSampler's seed, independent of the rows)
        and the test loop's base seed; SLNet has no dropout, and the data, loss and
        optimiser draw nothing.
        """
        SLNet()
        torch.randn(128, 2)
        torch.randint(0, 2, (128,))
        loader = sliced_loader(torch.zeros(1, 1), torch.zeros(1, 1), 1)
        for _ in range(num_epochs):
            next(iter(loader))
        consume_loader_seed()

    def _tensors(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        data = dataset_classifier(df, self.input_length)
        return data.features, data.labels

    def setup_dataloaders_from_sets(self, test, train, validation=False):
        if validation:
            raise NotImplementedError("the harness trains without a validation split")
        print(f"Setting up dataloaders with input {self.input_length}, no validation")
        self.validation = False
        features, labels = self._tensors(train)
        self.dl_train = sliced_loader(features, labels, self.batch_size)
        self.dl_test = self._tensors(test)
        self.test_names = test.iloc[:, :5]
        self.test_SL = test.iloc[:, -1]
        self.train_SL = train.iloc[:, -1]

    def _predict(self, features: torch.Tensor, labels: torch.Tensor):
        """The released per-row test loop: argmax, softmax and labels per row."""
        x = features.float().to(self.device)
        y = labels.long().to(self.device)
        net = self.net

        def row(i):
            outputs = net(x[i : i + 1])
            return outputs, torch.nn.functional.softmax(outputs, dim=1)

        logits, probs = predict_rows(row, len(x))
        _, predicted = torch.max(logits.data, 1)
        conf, _ = torch.max(probs, 1)
        return (
            predicted.tolist(),
            y.flatten().tolist(),
            conf.tolist(),
            probs.cpu().numpy(),
        )

    def test_nn(self):
        if self.net is None:
            print("Error: no net exists")
            return
        consume_loader_seed()  # the released loop iterates dl_test
        vals, corr, conf_scores, probs = self._predict(*self.dl_test)
        correct = sum(v == c for v, c in zip(vals, corr))
        print(f"Accuracy: {correct / len(vals)}")
        return vals, corr, conf_scores, probs

    def compute_acc(self, preds, labels, probs):
        labels = np.asarray(labels).astype(int)
        preds = np.asarray(preds).astype(int)
        probs = as_matrix(probs)
        indicators = np.eye(2)[labels]
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        two = len(np.unique(labels)) == 2
        auc = roc_auc_score(indicators, probs) if two else 0.0
        aupr = average_precision_score(indicators, probs) if two else 0.0
        acc = (preds == labels).mean()
        return precision, recall, f1, auc, acc, aupr

    def get_calib_test(
        self, test, calib_frac, mondrian_class_dict=None, extract_name=None
    ):
        if mondrian_class_dict:
            raise NotImplementedError("only the empty Mondrian class map is supported")
        calib_size = int(len(test) * calib_frac)
        calib = test.sample(n=calib_size, random_state=0)
        test = test.drop(calib.index)
        self.test_names = test.iloc[:, :5]
        self.calib_names = calib.iloc[:, :5]
        return self._tensors(calib), self._tensors(test)

    def conformal_pred(
        self, calibration_dl, test_dl, mondrian_class_dict={}, extract_name=lambda x: x
    ):
        if self.net is None:
            print("Error: no net exists")
            return
        if mondrian_class_dict:
            raise NotImplementedError("only the empty Mondrian class map is supported")
        mondrian_class = -1
        print("Calibrating with 0 mondrian classes")
        consume_loader_seed()  # the released calibrate() iterates calibration_dl
        _, corr_c, _, probs_c = self._predict(*calibration_dl)
        p_calib = probs_c.astype(np.float64)
        true_class_prob = p_calib[np.arange(len(corr_c)), np.asarray(corr_c)]
        calibration_scores = {mondrian_class: (1 - true_class_prob).tolist()}

        consume_loader_seed()  # the released test loop iterates test_dl
        vals, corr, _, probs = self._predict(*test_dl)
        p = probs.astype(np.float64)
        predicted = np.asarray(vals)
        rows = np.arange(len(vals))
        best_conf = 1 - p[rows, predicted]
        worst_conf = 1 - p[rows, 1 - predicted]
        calib_np = np.sort(np.array(calibration_scores[mondrian_class]))
        n_calib = len(calib_np)

        def p_value(scores):
            count_greater_equal = n_calib - np.searchsorted(
                calib_np, scores, side="left"
            )
            return (count_greater_equal + 1) / (n_calib + 1)

        conf_scores = list(
            zip(p_value(best_conf).tolist(), p_value(worst_conf).tolist())
        )
        print("Finished conformal prediction")
        return vals, corr, conf_scores, self.test_names, calibration_scores, probs


class fast_via_film(via_film):
    def train_nn(self, path=None):
        """The released train_nn (one gene embedding, ``two_genes`` False) with the
        per-batch ``loss.item()`` deferred to the end of the epoch."""
        if self.two_genes:
            raise NotImplementedError("only the single-gene FiLM net is supported")
        net = FiLMNet(protein_size=self.input_length - 512)
        print(f"Protein size is {self.input_length - 512}")
        net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        num_samples = len(self.dl)
        print(f"Starting to train, saving result at {path}")
        loss_arr = []
        for epoch in range(self.num_epochs):
            losses = []
            for inputs, labels in self.dl:
                gene_embs = inputs[:, 512:]
                inputs = inputs[:, :512]
                inputs = inputs.float().to(self.device)
                gene_embs = gene_embs.float().to(self.device)
                labels = labels.float().to(self.device)
                optimizer.zero_grad()
                outputs, _ = net(inputs, gene_embs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())
            mean_loss = epoch_loss(losses) / num_samples
            print(f"Epoch: {epoch} / {self.num_epochs}, Loss: {mean_loss}")
            loss_arr.append(mean_loss)
        print("Finished Training")
        self.loss_arr = loss_arr
        self.net = net
        if path is not None:
            print(f"Saving model at {path}")
            torch.save(net.state_dict(), path)
        else:
            print("No path supplied, model not saved")

    def setup_dataloaders(self, test_size, seed=None):
        print(f"Setting up dataloaders with input length {self.input_length}")
        if test_size == 0.0:
            train_data = dataset(self.df, self.input_length)
            self.dl = sliced_loader(
                train_data.features, train_data.labels, self.batch_size
            )
            return
        if seed is not None:
            train, test = train_test_split(
                self.df, test_size=test_size, random_state=seed
            )
        else:
            train, test = train_test_split(self.df, test_size=test_size)
        train_data = dataset(train, self.input_length)
        test_data = dataset(test, self.input_length)
        self.dl = sliced_loader(train_data.features, train_data.labels, self.batch_size)
        self.dl_test = (test_data.features, test_data.labels)

    def _split(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Resident contiguous copies of the 512 cell and the gene embedding columns,
        converted to float32 as the released per-row ``.float()`` does."""
        x = features.float().to(self.device)
        return x[:, :512].contiguous(), x[:, 512:].contiguous()

    def test_nn(self):
        if self.net is None:
            print("Error: no net exists")
            return
        features, labels = self.dl_test
        cells, genes = self._split(features)
        net = self.net
        consume_loader_seed()  # the released loop iterates dl_test
        (outputs,) = predict_rows(
            lambda i: (net(cells[i : i + 1], genes[i : i + 1])[0],), len(cells)
        )
        vals = outputs.flatten().tolist()
        y_true = labels.float().flatten().tolist()
        mse = float(np.mean((np.asarray(vals) - np.asarray(y_true)) ** 2))
        print(f"Done testing, MSE: {mse}")
        return vals, y_true

    def extract_nn_second_last(self, test, path=None):
        if self.net is None:
            print("Error: no net exists")
            return
        data = dataset_unlabeled(test)
        cells, genes = self._split(data.features)
        net = self.net
        (outputs, second_last) = predict_rows(
            lambda i: tuple(t.unsqueeze(0) for t in net(cells[i], genes[i])),
            len(cells),
        )
        res = pd.DataFrame(second_last.cpu().numpy(), index=data.names)
        if path is not None:
            res.to_hdf(path, key="table")
        return pd.Series(outputs.flatten().cpu().numpy(), index=data.names), res
