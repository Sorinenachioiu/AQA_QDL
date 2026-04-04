import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pennylane as qml
from pennylane import numpy as np


@dataclass
class VQCConfig:
    n_qubits: int = 3
    n_layers: int = 5
    device_name: str = "default.qubit"
    seed: int = 42
    

class VariationalQuantumClassifier:
    def __init__(self, config: VQCConfig, backend=None):
        self.config = config
        self.backend = backend
        np.random.seed(config.seed)
        
        if backend is None:
            self.dev = qml.device(
                config.device_name,
                wires=config.n_qubits,
            )
        else:
            self.dev = qml.device(
                "qiskit.remote",
                wires=config.n_qubits,
                backend=backend,
                shots=2048,
                optimization_level=1,
            )
        
        self.qnode = self._build_qnode()
    
    # Initialize params
    def init_params(self, scale: float = 0.01) -> np.ndarray:
        return scale * np.random.randn(
            self.config.n_layers,
            self.config.n_qubits,
            2,  
            requires_grad=True,
        )
    
    # Define variational layer (RY and RZ rotations + CNOT entanglement)
    def _variational_layer(self, params_layer: np.ndarray):
        for q in range(self.config.n_qubits):
            qml.RY(params_layer[q, 0], wires=q)
            qml.RZ(params_layer[q, 1], wires=q)
  
        for q in range(self.config.n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        qml.CNOT(wires=[self.config.n_qubits - 1, 0])
    
    # Define the full circuit with amplitude embedding and variational layers
    def _circuit(self, x: np.ndarray, params: np.ndarray):
        qml.AmplitudeEmbedding(x, wires=range(self.config.n_qubits), normalize=True)
        
        for layer in range(self.config.n_layers):
            self._variational_layer(params[layer])
    
    def _build_qnode(self):
        """Build quantum node for classification"""
        @qml.qnode(self.dev, interface="autograd")
        def qnode(x, params):
            self._circuit(x, params)
            # Measure expectation of Z on the first qubit to get a value in [-1, 1]
            return qml.expval(qml.PauliZ(0))
        
        return qnode
    
    def predict_proba_single(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        measurement = self.qnode(x, params)
        prob_class_0 = (1 + measurement) / 2
        prob_class_1 = 1 - prob_class_0
        
        return np.array([prob_class_0, prob_class_1])
    
    def predict_single(self, x: np.ndarray, params: np.ndarray) -> int:
        """Predict class label for a single sample"""
        probs = self.predict_proba_single(x, params)
        return int(np.argmax(probs))
    
    def predict_proba(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Predict class probabilities for multiple samples"""
        return np.array([self.predict_proba_single(x, params) for x in X])
    
    def predict(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Predict class labels for multiple samples"""
        return np.array([self.predict_single(x, params) for x in X])
    
    def loss(self, params: np.ndarray, X_batch: np.ndarray, y_batch: np.ndarray):
        losses = []
        eps = 1e-8  
        
        for x, y_true in zip(X_batch, y_batch):
            probs = self.predict_proba_single(x, params)
            loss_sample = -np.log(probs[int(y_true)] + eps)
            losses.append(loss_sample)
        
        return np.mean(np.array(losses))
    
    def accuracy(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X, params)
        return float(np.mean(y_pred == y))
    
    def train(
        self,
        X_train_raw: np.ndarray,
        y_train: np.ndarray,
        X_val_raw: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        steps: int = 100,
        lr: float = 0.0001,
        batch_size: Optional[int] = 32,
        init_scale: float = 0.01,
        val_every: int = 10,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, List[float]], Dict[str, np.ndarray]]:
        X_train = np.array(X_train_raw, dtype=float)
        X_val = None
        if X_val_raw is not None:
            X_val = np.array(X_val_raw, dtype=float)
        
        params = self.init_params(scale=init_scale)
        opt = qml.AdamOptimizer(stepsize=lr)
        
        n_samples = len(X_train)
        if batch_size is None or batch_size > n_samples or batch_size <= 0:
            batch_size = n_samples
        
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        best_params = np.array(params, requires_grad=False)
        best_metric = float("inf")
        
        for step in range(steps):
            idx = np.random.choice(n_samples, size=batch_size, replace=False)
            X_batch = X_train[idx]
            y_batch = y_train[idx]
            
            params, train_loss = opt.step_and_cost(
                lambda p: self.loss(p, X_batch, y_batch),
                params
            )
            train_loss = float(train_loss)
            
            train_acc = self.accuracy(params, X_batch, y_batch)
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            
            metric = train_loss
            msg = f"[Step {step:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
            
            do_val = (
                X_val is not None and y_val is not None and
                (step % val_every == 0 or step == steps - 1)
            )
            
            if do_val:
                val_loss = float(self.loss(params, X_val, y_val))
                val_acc = self.accuracy(params, X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                metric = val_loss
                msg += f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            
            if metric < best_metric:
                best_metric = metric
                best_params = np.array(params, requires_grad=False)
            
            if verbose and (step % 5 == 0 or step == steps - 1):
                print(msg)
        
        return best_params, history
    
    def evaluate(
        self,
        X_raw: np.ndarray,
        y: np.ndarray,
        params: np.ndarray,
    ) -> Dict[str, float]:
        X = np.array(X_raw, dtype=float)
        loss = self.loss(params, X, y)
        acc = self.accuracy(params, X, y)
        
        return {
            "loss": float(loss),
            "accuracy": float(acc)
        }
    
