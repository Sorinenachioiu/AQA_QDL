import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pennylane as qml
from pennylane import numpy as np


@dataclass
class QAEConfig:
    n_qubits: int = 4
    n_latent: int = 2
    n_layers: int = 1
    device_name: str = "default.qubit"
    seed: int = 42
    use_swap_test_loss: bool = True
    use_log_cost: bool = True
    eps: float = 1e-8

    @property
    def n_trash(self) -> int:
        return self.n_qubits - self.n_latent

    @property
    def latent_wires(self) -> List[int]:
        return list(range(self.n_latent))

    @property
    def trash_wires(self) -> List[int]:
        return list(range(self.n_latent, self.n_qubits))

    @property
    def ref_wires(self) -> List[int]:
        return list(range(self.n_qubits, self.n_qubits + self.n_trash))

    @property
    def ancilla_wire(self) -> int:
        return self.n_qubits + self.n_trash

    @property
    def total_swap_wires(self) -> int:
        return self.n_qubits + self.n_trash + 1


class QuantumAutoencoder:
    # def __init__(self, config: QAEConfig):
    #     self.config = config
    #     np.random.seed(config.seed)

    #     # Device for normal encoding / latent extraction
    #     self.dev = qml.device(config.device_name, wires=config.n_qubits)

    #     # Device for SWAP-test loss
    #     self.dev_swap = qml.device(
    #         config.device_name,
    #         wires=config.total_swap_wires
    #     )

    #     (
    #         self.trash_probs_qnode,
    #         self.latent_qnode,
    #         self.swap_test_prob_qnode,
    #     ) = self._build_qnodes()
    def __init__(self, config: QAEConfig, backend=None):
        self.config = config
        self.backend = backend
        np.random.seed(config.seed)

        if backend is None:
            self.dev = qml.device(
                config.device_name,
                wires=config.n_qubits,
            )
            self.dev_swap = qml.device(
                config.device_name,
                wires=config.total_swap_wires,
            )
        else:
            self.dev = qml.device(
                "qiskit.remote",
                wires=config.n_qubits,
                backend=backend,
                shots=2048,
                optimization_level=1,
            )
            self.dev_swap = qml.device(
                "qiskit.remote",
                wires=config.total_swap_wires,
                backend=backend,
                shots=2048,
                optimization_level=1,
            )

        (
            self.trash_probs_qnode,
            self.latent_qnode,
            self.swap_test_prob_qnode,
        ) = self._build_qnodes()

    @staticmethod
    def fit_angle_scaler(X: np.ndarray) -> Dict[str, np.ndarray]:
        X = np.array(X, dtype=float)
        x_min = np.min(X, axis=0)
        x_max = np.max(X, axis=0)
        return {"x_min": x_min, "x_max": x_max}

    @staticmethod
    def transform_to_angles(X: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
        X = np.array(X, dtype=float)
        x_min = scaler["x_min"]
        x_max = scaler["x_max"]

        denom = np.where((x_max - x_min) == 0.0, 1.0, (x_max - x_min))
        X01 = (X - x_min) / denom
        X01 = np.clip(X01, 0.0, 1.0)

        return 2.0 * np.pi * X01 - np.pi

    def init_params(self, scale: float = 0.01) -> np.ndarray:
        return scale * np.random.randn(
            self.config.n_layers,
            self.config.n_qubits,
            2,
            requires_grad=True,
        )

    def _encoder_block(self, params: np.ndarray):
        for layer in range(self.config.n_layers):
            for q in range(self.config.n_qubits):
                qml.RY(params[layer, q, 0], wires=q)
                qml.RZ(params[layer, q, 1], wires=q)

            for q in range(self.config.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[self.config.n_qubits - 1, 0])

    def _build_qnodes(self):
        cfg = self.config

        @qml.qnode(self.dev, interface="autograd")
        def trash_probs_qnode(x, params):
            qml.AngleEmbedding(x, wires=range(cfg.n_qubits), rotation="Y")
            self._encoder_block(params)
            return qml.probs(wires=cfg.trash_wires)

        @qml.qnode(self.dev, interface="autograd")
        def latent_qnode(x, params):
            qml.AngleEmbedding(x, wires=range(cfg.n_qubits), rotation="Y")
            self._encoder_block(params)
            return [qml.expval(qml.PauliZ(w)) for w in cfg.latent_wires]

        @qml.qnode(self.dev_swap, interface="autograd")
        def swap_test_prob_qnode(x, params):
            # Data register on 0 .. n_qubits-1
            qml.AngleEmbedding(x, wires=range(cfg.n_qubits), rotation="Y")
            self._encoder_block(params)

            # Reference register B' is left at |00...0> by default

            # SWAP test between trash B and reference B'
            anc = cfg.ancilla_wire
            qml.Hadamard(wires=anc)

            for trash_w, ref_w in zip(cfg.trash_wires, cfg.ref_wires):
                qml.CSWAP(wires=[anc, trash_w, ref_w])

            qml.Hadamard(wires=anc)

            return qml.probs(wires=[anc])

        return trash_probs_qnode, latent_qnode, swap_test_prob_qnode

    def _sample_fidelity_proxy(self, x: np.ndarray, params: np.ndarray) -> float:
        """
        Return a fidelity-like quantity C2.

        If using SWAP test:
            P(ancilla=0) = (1 + overlap)/2
            overlap = 2*P0 - 1

        Since the reference is |00...0>, overlap here is the
        trash/reference fidelity estimate.
        """
        if self.config.use_swap_test_loss:
            probs = self.swap_test_prob_qnode(x, params)
            p0 = probs[0]
            fidelity = 2.0 * p0 - 1.0
            fidelity = np.clip(fidelity, 0.0, 1.0)
            return fidelity
        else:
            # old simpler proxy
            probs = self.trash_probs_qnode(x, params)
            return probs[0]

    def loss(self, params: np.ndarray, X_batch: np.ndarray) -> float:
        """
        Paper-inspired reduced cost:
            maximize trash/reference fidelity C2

        We minimize either:
            1 - C2
        or
            log10(1 - C2 + eps)

        Romero et al. minimize log10(1 - C2) for stability.
        """
        fidelities = []
        for x in X_batch:
            fidelities.append(self._sample_fidelity_proxy(x, params))

        c2 = np.mean(np.array(fidelities))

        if self.config.use_log_cost:
            return np.log10(1.0 - c2 + self.config.eps)
        else:
            return 1.0 - c2

    def train(
        self,
        X_train_raw: np.ndarray,
        X_val_raw: Optional[np.ndarray] = None,
        steps: int = 30,
        lr: float = 0.03,
        batch_size: Optional[int] = 16,
        init_scale: float = 0.01,
        val_every: int = 10,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, List[float]], Dict[str, np.ndarray]]:
        if X_train_raw.shape[1] != self.config.n_qubits:
            raise ValueError(
                f"Expected {self.config.n_qubits} features, got {X_train_raw.shape[1]}"
            )

        scaler = self.fit_angle_scaler(X_train_raw)
        X_train = self.transform_to_angles(X_train_raw, scaler)

        X_val = None
        if X_val_raw is not None:
            if X_val_raw.shape[1] != self.config.n_qubits:
                raise ValueError(
                    f"Expected {self.config.n_qubits} features, got {X_val_raw.shape[1]}"
                )
            X_val = self.transform_to_angles(X_val_raw, scaler)

        params = self.init_params(scale=init_scale)
        opt = qml.AdamOptimizer(stepsize=lr)

        n_samples = len(X_train)
        if batch_size is None or batch_size > n_samples or batch_size <= 0:
            batch_size = n_samples

        history = {"train_loss": [], "val_loss": []}
        best_params = np.array(params, requires_grad=False)
        best_metric = float("inf")

        for step in range(steps):
            idx = np.random.choice(n_samples, size=batch_size, replace=False)
            X_batch = X_train[idx]

            params, train_loss = opt.step_and_cost(
                lambda p: self.loss(p, X_batch),
                params
            )
            train_loss = float(train_loss)
            history["train_loss"].append(train_loss)

            metric = train_loss
            msg = f"[Step {step:03d}] train_loss={train_loss:.6f}"

            do_val = (
                X_val is not None and
                (step % val_every == 0 or step == steps - 1)
            )

            if do_val:
                val_loss = float(self.loss(params, X_val))
                history["val_loss"].append(val_loss)
                metric = val_loss
                msg += f" | val_loss={val_loss:.6f}"

            if metric < best_metric:
                best_metric = metric
                best_params = np.array(params, requires_grad=False)

            if verbose and (step % 5 == 0 or step == steps - 1):
                print(msg)

        return best_params, history, scaler

    def latent_features_from_raw(
        self,
        X_raw: np.ndarray,
        params: np.ndarray,
        scaler: Dict[str, np.ndarray],
    ) -> np.ndarray:
        X = self.transform_to_angles(X_raw, scaler)
        feats = [self.latent_qnode(x, params) for x in X]
        return np.array(feats, dtype=float)

    def batch_transform_latent_features_from_raw(
        self,
        X_raw: np.ndarray,
        params: np.ndarray,
        scaler: Dict[str, np.ndarray],
        batch_size: int = 1000,
        verbose: bool = True,
    ) -> np.ndarray:
        X_raw = np.array(X_raw, dtype=float)
        out = []

        n = len(X_raw)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = X_raw[start:end]
            chunk_latent = self.latent_features_from_raw(chunk, params, scaler)
            out.append(chunk_latent)

            if verbose:
                print(f"Latent transform: {end}/{n}")

        return np.vstack(out)

    def trash_probabilities_from_raw(
        self,
        x_raw: np.ndarray,
        params: np.ndarray,
        scaler: Dict[str, np.ndarray],
    ) -> np.ndarray:
        x = self.transform_to_angles(np.array([x_raw]), scaler)[0]
        return self.trash_probs_qnode(x, params)

    def swap_test_probability_from_raw(
        self,
        x_raw: np.ndarray,
        params: np.ndarray,
        scaler: Dict[str, np.ndarray],
    ) -> np.ndarray:
        x = self.transform_to_angles(np.array([x_raw]), scaler)[0]
        return self.swap_test_prob_qnode(x, params)

    def save_checkpoint(
        self,
        path: str,
        params: np.ndarray,
        scaler: Dict[str, np.ndarray],
        history: Optional[Dict[str, List[float]]] = None,
        extra_metadata: Optional[Dict] = None,
    ) -> None:
        metadata = {
            "n_qubits": self.config.n_qubits,
            "n_latent": self.config.n_latent,
            "n_layers": self.config.n_layers,
            "device_name": self.config.device_name,
            "seed": self.config.seed,
            "use_swap_test_loss": self.config.use_swap_test_loss,
            "use_log_cost": self.config.use_log_cost,
        }
        if extra_metadata is not None:
            metadata.update(extra_metadata)

        if history is None:
            history = {}

        np.savez(
            path,
            params=np.array(params, dtype=float),
            x_min=np.array(scaler["x_min"], dtype=float),
            x_max=np.array(scaler["x_max"], dtype=float),
            train_loss=np.array(history.get("train_loss", []), dtype=float),
            val_loss=np.array(history.get("val_loss", []), dtype=float),
            metadata_json=json.dumps(metadata),
        )

    @staticmethod
    def load_checkpoint(path: str):
        data = np.load(path, allow_pickle=True)

        params = data["params"]
        scaler = {
            "x_min": data["x_min"],
            "x_max": data["x_max"],
        }
        history = {
            "train_loss": data["train_loss"].tolist(),
            "val_loss": data["val_loss"].tolist(),
        }
        metadata = json.loads(str(data["metadata_json"]))

        return params, scaler, history, metadata