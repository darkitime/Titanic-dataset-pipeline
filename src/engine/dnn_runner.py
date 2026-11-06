import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Dict, Any, Optional, List, Tuple

from configs.config_DNN import config_DNN
from src.models.tabular_nn import TitanicModel
from src.data_prep.dataset import get_kfold_datasets

class EarlyStopping():
  def __init__(
        self, 
        mode = config_DNN.earlystopping.mode, 
        patience = config_DNN.earlystopping.patience, 
        threshold=config_DNN.earlystopping.threshold, 
        threshold_mode = config_DNN.earlystopping.threshold_mode
    ):
    if mode not in {'min','max'}:
      raise ValueError(f"Available values for mode: max, min")
    if threshold_mode not in {'rel','abs'}:
      raise ValueError(f"Available values for threshold_mode: rel, abs")
    if not isinstance(patience, int):
      raise ValueError(f"Available values for patience must be int")
    if not isinstance(threshold, float):
      raise ValueError(f"Available values for threshold must be int and less than 1,0.")
    if threshold >= 1.0:
      raise ValueError(f"Available values for threshold must be int and less than 1,0.")

    self.mode = mode
    self.patience = patience
    self.threshold = threshold
    self.threshold_mode = threshold_mode
    self.count = 0
    self.best = None

  def changed_better(self, current, best):
    if self.mode == 'min' and self.threshold_mode == 'rel':
      return current < best - best * self.threshold
    elif self.mode == 'min' and self.threshold_mode == 'abs':
      return current < best - self.threshold

    elif self.mode =='max' and self.threshold_mode == 'rel':
      return current > best + best * self.threshold
    else:
      return current > best + self.threshold

  def __call__(self, tracked_parameter):
    current = float(tracked_parameter)
    if self.best is None:
      self.best = current
      return False

    if self.changed_better(current, self.best):
      self.best = current
      self.count = 0
    else:
      self.count += 1

    if self.count >= self.patience:
      self.count = 0
      return True

def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = config_DNN.training.lr,
    optimizer_name: str = config_DNN.optimizer.name,
    scheduler_name: Optional[str] = config_DNN.scheduler.name,
    weight_decay: float = config_DNN.training.weight_decay,
    n_epochs: int = config_DNN.training.num_epochs,
    early_stopping_patience: int = config_DNN.training.early_stopping_patience,
    loss_name: str = config_DNN.loss.name,
    device: str = config_DNN.general.device,
    verbose: bool = False
) -> Dict[str, Any]:
    '''
    train one model on one fold
    '''
    model.to(device)

    loss_model = getattr(nn, loss_name)()
    optimizer_cls = getattr(optim, optimizer_name)
    opt = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    #Scheduler
    lr_scheduler = None
    if scheduler_name in ('Plateau', 'ReduceLROnPlateau'):
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=config_DNN.scheduler.params.mode,
            factor=config_DNN.scheduler.params.factor,
            patience=config_DNN.scheduler.params.patience
        )
    elif scheduler_name == ('Cosine', 'CosineAnnealingLR'):
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=n_epochs,
            eta_min=0.0
        )
    else:
        raise ValueError(f"Scheduler name: {scheduler_name} not in config_DNN")
    #EarlyStopping
    earlystopping = EarlyStopping()

    #Dict
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_weights = deepcopy(model.state_dict())
    
    for epoch in range(n_epochs):
        # TRAINING 
        model.train()
        running_train_loss = []
        true_answer = 0

        train_loop = tqdm(train_loader, leave=False, disable=not verbose)
        for x, targets in train_loop:
            x, targets = x.to(device), targets.to(device)

            # Forward
            pred = model(x)
            loss = loss_model(pred, targets)

            # Backward
            opt.zero_grad()
            loss.backward()
            if config_DNN.training.gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), config_DNN.training.clip_value)
            opt.step()

            #Compute metrics and loss
            running_train_loss.append(loss.item())
            mean_train_loss = np.mean(running_train_loss)

            preds_cls = (pred >= 0.0).float()
            true_answer += (preds_cls == targets).sum().item()

            if verbose:
                train_loop.set_description(f"Epoch [{epoch+1}/{n_epochs}], train_loss = {mean_train_loss:.4f}")

        running_train_acc = true_answer / len(train_loader.dataset)
        history['train_loss'].append(mean_train_loss)
        history['train_acc'].append(running_train_acc)

        # VALIDATION 
        model.eval()
        val_true_answer = 0
        running_val_loss = []

        with torch.no_grad():
            for x, targets in val_loader:
                x, targets = x.to(device), targets.to(device)

                pred = model(x)
                loss = loss_model(pred, targets)

                running_val_loss.append(loss.item())
                preds_cls = (pred >= 0.0).float()
                val_true_answer += (preds_cls == targets).sum().item()

        mean_val_loss = np.mean(running_val_loss)
        running_val_acc = val_true_answer / len(val_loader.dataset)

        history['val_loss'].append(mean_val_loss)
        history['val_acc'].append(running_val_acc)

        # Scheduler step 
        if lr_scheduler is not None:
            if scheduler_name in ('Plateau', 'ReduceLROnPlateau'):
                lr_scheduler.step(mean_val_loss)
            else:
                lr_scheduler.step()

        current_lr = opt.param_groups[0]['lr']
        history['lr'].append(current_lr)

        if verbose:
            print(
                f"Epoch [{epoch+1}/{n_epochs}] | "
                f"train_loss: {mean_train_loss:.4f}, train_acc: {running_train_acc:.4f} | "
                f"val_loss: {mean_val_loss:.4f}, val_acc: {running_val_acc:.4f} | lr: {current_lr:.6f}"
            )

        # Saving best
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_val_acc = running_val_acc
            best_model_weights = deepcopy(model.state_dict())

        # checking EarlyStopping
        if earlystopping(mean_val_loss):
            if verbose:
                print(f"Early stopping worked on {epoch+1}")
            break

    # loading best weights
    model.load_state_dict(best_model_weights)

    return {
        'model': model,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'epochs_trained': epoch + 1,
        'history': history
    }

@torch.no_grad()
def predict_proba_dataloader(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Returns class probability of 1_class (using Sigmoid) for any DataLoader."""
    model.eval()
    model.to(device)
    all_probs = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.append(probs)
    return np.concatenate(all_probs)

def train_kfold_dnn(
    folds_data: List[Dict[str, Any]],
    y_full_train: np.ndarray,
    num_continuous_features: int,
    emb_dims: Optional[List[Tuple[int, int]]] = None,
    verbose_folds: bool = True
) -> Dict[str, Any]:
    """
    Initialize new model on all folds, using train_one_model,
    computes their loss/accuracy respect to folds, OOF accuracy and averrage predict on test.
    """
    device = config_DNN.general.device
    fold_losses = []
    fold_accs = []
    fold_models = []

    oof_proba = np.zeros(len(y_full_train))
    n_test = len(folds_data[0]['test_dataset'])
    test_proba_mean = np.zeros(n_test)

    for fold_info in folds_data:
        fold_idx = fold_info['fold']
        train_loader = DataLoader(
            fold_info['train_dataset'],
            batch_size=config_DNN.data.train_batch_size,
            shuffle=True,
            num_workers=config_DNN.data.num_workers
        )
        val_loader = DataLoader(
            fold_info['val_dataset'],
            batch_size=config_DNN.data.val_batch_size,
            shuffle=False,
            num_workers=config_DNN.data.num_workers
        )
        test_loader = DataLoader(
            fold_info['test_dataset'],
            batch_size=config_DNN.data.test_batch_size,
            shuffle=False,
            num_workers=config_DNN.data.num_workers
        )

        # For every fold creating NEW model
        model = TitanicModel(
            num_continuous_features=num_continuous_features,
            emb_dims=emb_dims
        )

        # Trains on current fold
        res = train_one_model(model, train_loader, val_loader, verbose=False)
        best_model = res['model']

        fold_losses.append(res['best_val_loss'])
        fold_accs.append(res['best_val_acc'])
        fold_models.append(best_model)

        # Writes OOF predictions and accumulate predictions on test
        oof_proba[fold_info['val_idx']] = predict_proba_dataloader(best_model, val_loader, device)
        test_proba_mean += predict_proba_dataloader(best_model, test_loader, device) / len(folds_data)

        if verbose_folds:
            print(f"[Fold {fold_idx + 1}/{len(folds_data)}] "
                  f"Best Val Loss: {res['best_val_loss']:.4f} | "
                  f"Best Val Acc: {res['best_val_acc']:.4f} (stopped at epoch {res['epochs_trained']})")

    mean_val_loss = float(np.mean(fold_losses))
    mean_val_acc = float(np.mean(fold_accs))
    oof_acc = accuracy_score(y_full_train, (oof_proba >= config_DNN.metric.threshold).astype(int))

    print(f"\n---> DNN K-Fold Summary: Mean Val Loss = {mean_val_loss:.4f} | "
          f"Mean Val Acc = {mean_val_acc:.4f} | OOF Acc = {oof_acc:.4f}")

    return {
        'mean_val_loss': mean_val_loss,
        'mean_val_acc': mean_val_acc,
        'oof_accuracy': oof_acc,
        'oof_proba': oof_proba,
        'test_proba': test_proba_mean,
        'test_preds': (test_proba_mean >= config_DNN.metric.threshold).astype(int),
        'models': fold_models
    }

def run_dnn():
    '''Data preprocessing and K-Fold run for TabularMLP'''
    folds_data, y_train, y_test = get_kfold_datasets()

    if config_DNN.model.params.use_embeddings:
        emb_dims = [(2, 2), (3, 2), (5, 3)] # Sex: 2 класса (0..1), Embarked: 3 класса (0..2), Initial: 5 классов (0..4)
        num_cont = 5  # ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    else:
        emb_dims = None
        num_cont = folds_data[0]['train_dataset'].X.shape[1]

    dnn_res = train_kfold_dnn(
        folds_data = folds_data,
        y_full_train = y_train,
        num_continuous_features = num_cont,
        emb_dims = emb_dims,
        verbose_folds = True
    )
    return dnn_res