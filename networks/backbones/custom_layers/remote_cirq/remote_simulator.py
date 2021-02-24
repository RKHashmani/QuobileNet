""" A Floq Client based on Cirq.

Simulate quantum circuits on the cloud. This client uses the cirq interface and
implements the SimulatesSamples and SimulatesExpectationValues.

  Typical usage:
  sim = remote_cirq.RemoteSimulator(my_api_key)

  circuit = cirq.Circuit()
  observables = cirq.PauliSum()

  result = sim.run(circuit)
  expectation = sim.simulate_expectation_values(circuit, observables)
"""

API_URL = 'floq.endpoints.quantum-x99.cloud.goog'

from .base_simulator import BaseSimulator

class RemoteSimulator(BaseSimulator):
  def __init__(self, api_key):
    super().__init__(api_key, API_URL)