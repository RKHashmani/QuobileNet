"""Shared marshmallow schemas"""
import dataclasses
import enum
import typing
import marshmallow
import marshmallow_dataclass
import marshmallow_enum


#####################################
# API relevant types                #
#####################################


UUID = marshmallow_dataclass.NewType("UUID", str, field=marshmallow.fields.UUID)


@dataclasses.dataclass
class APIError:
    """API error response data.

    Properties:
        error: Error details.
    """

    error: str


class JobStatus(enum.IntEnum):
    """Current job status."""

    NOT_STARTED = 0
    IN_PROGRESS = 1
    COMPLETE = 2
    ERROR = 3


class JobType(enum.IntEnum):
    """Simulation job type"""

    SAMPLE = 0
    EXPECTATION = 1


@dataclasses.dataclass
class JobResult:
    """Get job results response data.

    Properties:
        id: Unique job id.
        result: Job result.
        status: Current job status.
    """

    id: UUID  # pylint: disable=invalid-name
    status: JobStatus = dataclasses.field(
        metadata={
            "marshmallow_field": marshmallow_enum.EnumField(JobStatus, by_value=True)
        }
    )
    error_message: typing.Optional[str] = dataclasses.field(default=None)
    result: typing.Optional[typing.Any] = dataclasses.field(default=None)

    def __post_init__(self):
        """See base class documentation."""
        if self.status == JobStatus.ERROR:
            if not self.error_message:
                raise ValueError("Missing error messsage")
            if self.result:
                raise ValueError("Failed job cannot have result field")

        if self.status == JobStatus.COMPLETE:
            if not self.result:
                raise ValueError("Missing job result")
            if self.error_message:
                raise ValueError("Completed job cannot have error_message field")


@dataclasses.dataclass
class JobSubmitted:
    """Submit job response data.

    Properties:
        id: Unique job id.
    """

    id: UUID  # pylint: disable=invalid-name


@dataclasses.dataclass
class SubmitJobContext:
    """Submit job request data.

    Properties:
        circuit: JSON-encoded `cirq.circuits.Circuit` to be run.
        param_resolver: JSON-encoded `cirq.study.ParamResolver` to be used with
          the circuit.
    """

    circuit: str
    param_resolver: str


@dataclasses.dataclass
class ExpectationJobContext(SubmitJobContext):
    """Submit expectation job request data.

    Properties:
        operators: List of JSON-encoded `cirq.ops.PauliSum` operators.
    """

    operators: typing.List[str]


@dataclasses.dataclass
class ExpectationJobResult(JobResult):
    """Get expectation job results response data.

    Properties:
        id: Unique job id.
        result: List of floats, same size as input operators size.
        status: Current job status.
    """

    result: typing.Optional[typing.List[float]] = dataclasses.field(default=None)


@dataclasses.dataclass
class SampleJobContext(SubmitJobContext):
    """Submit sample job request data.

    Properties:
        repetitions: Number of times the circuit will run.
    """

    repetitions: int = dataclasses.field(default=1)


@dataclasses.dataclass
class SampleJobResult(JobResult):
    """Get sample job results response data.

    Properties:
        id: Unique job id.
        result: A JSON-encoded `cirq.study.TrialResult` object containing the
          output from running the circuit.
        status: Current job status.
    """

    result: typing.Optional[str] = dataclasses.field(default=None)


#####################################
# Redis relevant types              #
#####################################


@dataclasses.dataclass
class Job:
    """Start a new job request data.

    Properties:
        context: Job specific context.
        id: Unique job id.
    """

    id: UUID  # pylint: disable=invalid-name
    type: JobType = dataclasses.field(
        metadata={
            "marshmallow_field": marshmallow_enum.EnumField(JobType, by_value=True)
        }
    )
    context: typing.Any


@dataclasses.dataclass
class JobIds:
    """Utility type for mapping API key to associated job ids.

    Properties:
        ids: List of unique jobs ids.
    """

    ids: typing.List[UUID]


#####################################
# Schema Objects                    #
#####################################


class JobResultBaseSchema(marshmallow.Schema):
    """Base marshmallow schema for JobResult dataclass."""

    @marshmallow.post_dump
    def remove_empty_fields(self, data: typing.Dict, **_kwargs) -> typing.Dict:
        """Removes all None fields from the input data.

        Args:
            data: Input data object.

        Returns:
            Filtered data.
        """
        return {k: v for k, v in data.items() if v is not None}


(
    APIErrorSchema,
    JobSchema,
    JobIdsSchema,
    JobSubmittedSchema,
    SubmitJobContextSchema,
    ExpectationJobContextSchema,
    ExpectationJobResultSchema,
    SampleJobContextSchema,
    SampleJobResultSchema,
) = tuple(
    marshmallow_dataclass.class_schema(x)()
    for x in (
        APIError,
        Job,
        JobIds,
        JobSubmitted,
        SubmitJobContext,
        ExpectationJobContext,
        ExpectationJobResult,
        SampleJobContext,
        SampleJobResult,
    )
)

JobResultSchema = marshmallow_dataclass.class_schema(
    JobResult, base_schema=JobResultBaseSchema
)()
