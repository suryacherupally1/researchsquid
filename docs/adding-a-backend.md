# Adding a New Backend

One file. Implement `BaseBackend`. Register in `registry.py`.

## Steps

1. Create `squid/backends/my_new_backend/executor.py`
2. Implement `BaseBackend.validate_inputs()` and `BaseBackend.run_experiment()`
3. Add one line to `squid/backends/registry.py`
4. Done — Coordinator routes to it automatically

## Example

```python
from squid.backends.base import BaseBackend
from squid.schema.experiment import ExperimentResult, ExperimentSpec

class MyNewBackend(BaseBackend):
    def validate_inputs(self, inputs: dict) -> None:
        if "my_required_field" not in inputs:
            raise ValueError("Missing required field: my_required_field")

    async def run_experiment(self, spec: ExperimentSpec) -> ExperimentResult:
        try:
            # do the actual work
            return ExperimentResult(
                id=f"run_{uuid4().hex[:8]}",
                spec_id=spec.id,
                session_id=spec.session_id,
                hypothesis_finding_id=spec.hypothesis_finding_id,
                backend_type="my_new_backend",
                status="completed",
                summary="Result summary",
                metrics={},
                judgment=BackendJudgment(outcome="supports", confidence="moderate", reason="..."),
                artifacts=[],
                environment={"version": "1.0"},
                cost={},
            )
        except Exception as e:
            return self.make_failed_result(spec, str(e))
```
