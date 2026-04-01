"""HiveResearch coordinator — session lifecycle, routing, budget, telemetry."""

# Lazy imports to avoid dependency chain issues at import time.
# These are available via direct import:
#   from hive.coordinator.session import create_session
#   from hive.coordinator.app import create_app
#   from hive.coordinator.audit import AuditLogger, EventType
#   from hive.coordinator.telemetry import SessionTelemetry
#   from hive.coordinator.report_agent import ReportAgent
