"""Floq Client based on Cirq base class.

Simulate quantum circuits on the cloud. This client uses the cirq interface and
implements the SimulatesSamples and SimulatesExpectationValues.

  Typical usage:
  sim = remote_cirq.BaseSimulator(my_api_key, api_url)

  circuit = cirq.Circuit()
  observables = cirq.PauliSum()

  result = sim.run(circuit)
  expectation = sim.simulate_expectation_values(circuit, observables)
"""

import json
import time
import cirq
import requests
import numpy as np
import marshmallow_dataclass
from typing import List, Union, Any, Dict
from . import schemas


class SimulatorError(Exception):
  pass


class ServiceException(Exception):
  pass


class TimeoutException(Exception):
  pass


SERIAL_ERROR_MSG = '''
Cirq encountered a serialization error. This may be due to passing gates
parameterized on more than one symbol, which Cirq currently does not support.
Because Cirq rx, ry, and rz gates depend on an implicit internal symbol they
can fail. This is actively being resolved! In the meantime try using XPow,
YPow, ZPow gates instead:

cirq.rx(s) -> cirq.XPowGate(exponent=s / np.pi, global_shift=-0.5)
'''

API_VERSION = '1'
TIMEOUT = 60 * 10
BASE_POLL_FREQUENCY = 0.1
MAX_POLL_FREQUENCY = 15.0


class BaseSimulator(cirq.SimulatesSamples):
  MAX_SIMS = 5
  count = 0

  def __init__(self, api_key, api_url, use_http=False):
    BaseSimulator.count += 1

    if BaseSimulator.count > BaseSimulator.MAX_SIMS:
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      print("WARNING: You are creating a large number of simulators!"
            "In the vast majority of cases you only need one."
            "Creating too many simulators may lead to slower results.")
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    prefix = 'http' if use_http else 'https'
    self.api_key = api_key
    self.api_url = f'{prefix}://{api_url}/api/v{API_VERSION}'

  def __del__(self):
    BaseSimulator.count -= 1

  def _get_headers(self):
    return {'X-API-Key': self.api_key}

  def _execute_req(self, r_func):
    try:
      r = r_func()
      r.raise_for_status()
    except requests.HTTPError as e:
      invalid = schemas.APIErrorSchema.validate(r.text)
      msg = 'Unknown service error'
      if not invalid:
        api_error = schemas.APIErrorSchema.loads(r.text)
        msg = api_error.error
      elif r.headers.get('content-type') == 'application/json':
        msg = r.json()
      raise ServiceException(msg) from e
    return r

  def _get_job_results(self, job_type: schemas.JobType,
                       job_id: schemas.UUID) -> schemas.JobResult:
    if not isinstance(job_type, schemas.JobType):
      TypeError('Invalid job_type.')

    definitions = {
        schemas.JobType.SAMPLE: ('sample', schemas.SampleJobResultSchema),
        schemas.JobType.EXPECTATION: ('exp', schemas.ExpectationJobResultSchema)
    }

    url_str, schema = definitions[job_type]

    r_fun = lambda: requests.get(
        url=f'{self.api_url}/jobs/{url_str}/{job_id}/results',
        headers=self._get_headers())

    poll_frequency = BASE_POLL_FREQUENCY
    t0 = time.time()
    while time.time() < t0 + TIMEOUT:
      r = self._execute_req(r_fun)
      r_data = self._decode(schema, r.text)
      if r_data.status == schemas.JobStatus.COMPLETE:
        return r_data
      elif r_data.status == schemas.JobStatus.ERROR:
        raise SimulatorError(f'ERROR status returned for JobID: {job_id}'
                             f' Error message: {r_data.error_message}')
      time.sleep(poll_frequency)
      poll_frequency = self._update_poll_frequency(poll_frequency,
                                                   time.time() - t0)
    else:
      raise TimeoutError(
          f'Timeout limit {TIMEOUT}s reached for JobID: {job_id}'
          'Call the [jobtype]_get_results method to resume polling')

  @staticmethod
  def _update_poll_frequency(pf, t):
    schedule = ((1.5, 0.1), (10, 1.0), (60, 5.0))

    for t_v, pf_v in schedule:
      if t < t_v:
        return pf_v

    return MAX_POLL_FREQUENCY

  def _submit_job(self, job_type: schemas.JobType,
                  data: schemas.SubmitJobContext):

    definitions = {
        schemas.JobType.SAMPLE: ('sample', schemas.SampleJobContextSchema),
        schemas.JobType.EXPECTATION:
            ('exp', schemas.ExpectationJobContextSchema)
    }

    url_str, schema = definitions[job_type]

    json = self._encode(schema, data)

    r_fun = lambda: requests.post(f'{self.api_url}/jobs/{url_str}/submit',
                                  data=json,
                                  headers=self._get_headers())

    r = self._execute_req(r_fun)

    r_data = self._decode(schemas.JobSubmittedSchema, r.text)
    job_id = r_data.id
    return job_id

  def _run(
      self,
      circuit: 'cirq.Circuit',
      param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
      repetitions: int = 1,
  ) -> Dict[str, np.ndarray]:
    """
    This method implements the cirq SimulatesSamples interface.

    For more information see documentation in the base class
    cirq.SimulatesSamples.
    """
    try:
      data = schemas.SampleJobContext(circuit=cirq.to_json(circuit),
                                      param_resolver=cirq.to_json(
                                          cirq.ParamResolver(param_resolver)),
                                      repetitions=repetitions)
    except TypeError as e:
      raise TypeError(SERIAL_ERROR_MSG) from e

    job_id = self._submit_job(schemas.JobType.SAMPLE, data)
    return self.sample_job_results(job_id)._measurements

  def sample_job_results(self, job_id: schemas.UUID) -> 'cirq.TrialResult':
    """Poll for results jon the given job_id."""
    data = self._get_job_results(schemas.JobType.SAMPLE, job_id)
    return cirq.read_json(json_text=data.result)

  def simulate_expectation_values(
      self,
      program: 'cirq.Circuit',
      observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']],
      param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
      qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
      initial_state: Any = None,
      permit_terminal_measurements: bool = True,
  ) -> List[float]:
    """Simulates the supplied circuit and calculates exact expectation
      values for the given observables on its final state.
      This method has no perfect analogy in hardware. Instead compare with
      Sampler.sample_expectation_values, which calculates estimated
      expectation values by sampling multiple times.
      Args:
          program: The circuit to simulate.
          observables: An observable or list of observables.
          param_resolver: Parameters to run with the program.
          qubit_order: Determines the canonical ordering of the qubits. This
              is often used in specifying the initial state, i.e. the
              ordering of the computational basis states.
          initial_state: The initial state for the simulation. The form of
              this state depends on the simulation implementation. See
              documentation of the implementing class for details.
          permit_terminal_measurements: If the provided circuit ends with
              measurement(s), this method will generate an error unless this
              is set to True. This is meant to prevent measurements from
              ruining expectation value calculations.
      Returns:
          A list of expectation values, with the value at index `n`
          corresponding to `observables[n]` from the input.
      """

    return self.simulate_expectation_values_sweep(
        program,
        observables,
        cirq.ParamResolver(param_resolver),
        qubit_order,
        initial_state,
        permit_terminal_measurements,
    )[0]

  def simulate_expectation_values_sweep(
      self,
      program: 'cirq.Circuit',
      observables: Union['cirq.PauliSum', List['cirq.PauliSum']],
      params: 'cirq.Sweepable',
      qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
      initial_state: Any = None,
      permit_terminal_measurements: bool = True,
  ) -> List[List[float]]:
    """Simulates the supplied circuit and calculates exact expectation
        values for the given observables on its final state, sweeping over the
        given params.
        This method has no perfect analogy in hardware. Instead compare with
        Sampler.sample_expectation_values, which calculates estimated
        expectation values by sampling multiple times.
        Args:
            program: The circuit to simulate.
            observables: An observable or list of observables.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.
            permit_terminal_measurements: If the provided circuit ends in a
                measurement, this method will generate an error unless this
                is set to True. This is meant to prevent measurements from
                ruining expectation value calculations.
        Returns:
            A list of expectation-value lists. The outer index determines the
            sweep, and the inner index determines the observable. For instance,
            results[1][3] would select the fourth observable measured in the
            second sweep.
        Raises:
            ValueError if 'program' has terminal measurement(s) and
            'permit_terminal_measurements' is False.
        """

    if len(program.all_qubits()) < 26:
      raise ValueError('Expectations are currently only supported on num_qubits'
                       ' >= 26')
    if qubit_order is not cirq.QubitOrder.DEFAULT:
      raise ValueError('A non-default qubit order is currently not supported')
    if initial_state is not None:
      raise ValueError('An initial state is currently not supported')
    if permit_terminal_measurements is not True:
      raise ValueError('Terminal measurements are always allowed and'
                       ' will always be removed automatically')
    if not isinstance(observables, cirq.PauliSum):
      if not all([isinstance(op, cirq.PauliSum) for op in observables]):
        raise TypeError(
            'Observables must be Union[cirq.PauliSum, Iterable[cirq.PauliSum]]')

    expectation_values = []
    for param_resolver in cirq.to_resolvers(params):
      expectation_values.append(
          self._expectation(
              circuit=program,
              observables=observables,
              param_resolver=param_resolver,
          ))
    return expectation_values

  def _expectation(
      self,
      circuit: 'cirq.Circuit',
      observables: Union['cirq.PauliSum', List['cirq.PauliSum']],
      param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
  ) -> List[float]:
    try:
      data = schemas.ExpectationJobContext(
          circuit=cirq.to_json(circuit),
          operators=self._serialize_operators(observables),
          param_resolver=cirq.to_json(cirq.ParamResolver(param_resolver)),
      )
    except TypeError as e:
      raise TypeError(SERIAL_ERROR_MSG) from e

    job_id = self._submit_job(schemas.JobType.EXPECTATION, data)
    result = self.expectation_job_results(job_id)
    return result

  def expectation_job_results(self, job_id: schemas.UUID) -> List[float]:
    """Poll for results on the given job_id."""
    data = self._get_job_results(schemas.JobType.EXPECTATION, job_id)
    return data.result

  @staticmethod
  def _serialize_operators(
      operators: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']]
  ) -> List[str]:

    def _dumps_paulisum(ps):
      return json.dumps([cirq.to_json(term) for term in ps])

    if isinstance(operators, cirq.PauliSum):
      return [_dumps_paulisum(operators)]
    return [_dumps_paulisum(op) for op in operators]

  @staticmethod
  def _encode(schema, data):
    return schema.dumps(data)

  @staticmethod
  def _decode(schema, data):
    return schema.loads(data)
