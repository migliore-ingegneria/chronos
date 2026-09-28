"""Tests for distributed lock heartbeat renewal worker."""

import time
import pytest
import torch

from chronos.core.problem import InnerProblem
from chronos.distributed.coordinator import Coordinator
from chronos.distributed.protocols import HeartbeatRequest, MessageType
from chronos.distributed.worker import Worker, WorkerConfig


class DummyInnerProblem(InnerProblem):
    def __init__(self):
        model = torch.nn.Linear(1, 1)
        super().__init__(model=model)

    def objective(self, params, outer_params, data=None):
        return (self.model(torch.tensor([[1.0]])) - outer_params.get("target", torch.tensor(0.0))) ** 2

    def solve(self, outer_params, init_params=None, num_steps=100, data_loader=None):
        from chronos.core.state import Trajectory
        trajectory = Trajectory(version=0, worker_id="test", outer_params=outer_params)
        return self.get_params(), trajectory


class TestHeartbeatProtocol:
    def test_heartbeat_request_serialization(self):
        req = HeartbeatRequest(worker_id="worker-test-123", current_version=2, status="computing")
        serialized = req.serialize()

        from chronos.distributed.protocols import parse_message
        msg_type, payload = parse_message(serialized)

        assert msg_type == MessageType.HEARTBEAT
        assert payload["worker_id"] == "worker-test-123"
        assert payload["current_version"] == 2
        assert payload["status"] == "computing"


class TestWorkerHeartbeatRenewal:
    def test_worker_heartbeat_renewal_prevents_timeout(self):
        port = 5577
        coordinator = Coordinator(
            outer_params={"target": torch.tensor(5.0)},
            port=port,
            heartbeat_timeout=0.5,
        )
        coordinator.start(blocking=False)
        time.sleep(0.1)

        try:
            problem = DummyInnerProblem()
            config = WorkerConfig(
                coordinator_addr=f"tcp://localhost:{port}",
                worker_id="heartbeat-worker-1",
                heartbeat_interval=0.1,
            )
            worker = Worker(problem, config)
            worker.connect()

            assert coordinator.num_workers == 1

            # Start heartbeat renewal worker
            worker.start_heartbeat_renew_worker()

            # Sleep 0.8s (longer than 0.5s heartbeat_timeout)
            time.sleep(0.8)

            # Worker should still be registered because of active heartbeat renewal
            assert coordinator.num_workers == 1

            # Stop heartbeat renewal worker
            worker.stop_heartbeat_renew_worker()

            # Sleep 1.2s to allow coordinator socket timeout to trigger _cleanup_dead_workers
            time.sleep(1.2)

            assert coordinator.num_workers == 0

            worker.disconnect()
        finally:
            coordinator.stop()
