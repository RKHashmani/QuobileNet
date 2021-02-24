import torch.nn as nn
import torch
import pennylane as qml
from pennylane import numpy as np
from networks.backbones.custom_layers import remote_cirq


class Quanv(nn.Module):
    def __init__(self, kernal_size, output_depth, circuit_layers=1):
        super().__init__()


        self.kernal_size = kernal_size  # kernel_size
        self.f = output_depth  # depth
        self.kernal_area = self.kernal_size ** 2

        # set number of qubits to fit all sizes
        if self.kernal_area >= self.f:
            # if kernel size is larger, we will just have fewer measurements than n_qubits
            self.n_qubits = self.kernal_area
        else:
            # if kernel size is smaller, we will leave some states as zero
            # more circuit layers will be beneficial for this mode
            self.n_qubits = self.f

        self.circuit_layers = circuit_layers # circuit layers

        if 26 <= self.n_qubits <= 32:  # If it meets the requirement for Floq, use it.
            API_KEY = floq_key
            sim = remote_cirq.RemoteSimulator(API_KEY)

            dev = qml.device("cirq.simulator",
                             wires=self.n_qubits,
                             simulator=sim,
                             analytic=False)
        else:
            dev = qml.device("default.qubit", wires=self.n_qubits)


        @qml.qnode(dev)
        def circuit(inputs, weights):
            for j in range(inputs.shape[0]):
                qml.RY(np.pi * inputs[j], wires=j)
            # Apply Hadamard to rest of the qubits, if there are less inputs then n_qubits
            # Hadamard is applied to kill the preference of starting from 0.
            for j in range(self.n_qubits - inputs.shape[0]):
                qml.Hadamard(wires=j+inputs.shape[0])
            # Random Layers is generally bad, they don't train good.
            # We can use Entangling Layers, or our own layers, or layers that we can run on real hardware :)
            #qml.templates.RandomLayers(weights, wires=list(range(self.n_qubits)))
            qml.templates.layers.BasicEntanglerLayers(weights, wires=list(range(self.n_qubits)))
            return [qml.expval(qml.PauliZ(j)) for j in range(self.f)]

        params = {"weights": (self.circuit_layers, self.n_qubits)} # 4 for area of kernel, 2x2
        self.qlayer = qml.qnn.TorchLayer(circuit, params)


    def forward(self, x):
        q_out = torch.zeros((x.shape[3] - self.kernal_size + 1), (x.shape[3] - self.kernal_size + 1), self.f)

        for idx in range(x.shape[3] - self.kernal_size + 1):
            for idy in range(x.shape[2] - self.kernal_size + 1):
                for idz in range(x.shape[1]):
                    q_out[idx, idy] += self.qlayer(self.flatten(x[0, idz, idx:idx + self.kernal_size, idy:idy + self.kernal_size]))

        return torch.reshape(q_out, (1, self.f, x.shape[3] - self.kernal_size + 1, x.shape[3] - self.kernal_size + 1))

    def flatten(self, t):
        t = t.reshape(1, -1)
        t = t.squeeze()
        return t
