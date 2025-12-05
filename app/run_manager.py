import asyncio
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class RunEvent:
    type: str
    message: str
    timestamp: str = field(default_factory=now_iso)


@dataclass
class RunRecord:
    id: str
    status: str = "pending"  # pending | running | failed | succeeded
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    command: List[str] = field(default_factory=list)
    events: List[RunEvent] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["events"] = [asdict(ev) for ev in self.events]
        return data


class RunManager:
    def __init__(self) -> None:
        self._runs: Dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def list_runs(self) -> List[Dict]:
        async with self._lock:
            return [r.to_dict() for r in self._runs.values()]

    async def get_run(self, run_id: str) -> Optional[RunRecord]:
        async with self._lock:
            return self._runs.get(run_id)

    async def start_run(self, command: List[str]) -> RunRecord:
        run_id = str(uuid.uuid4())
        record = RunRecord(id=run_id, status="pending", command=command)
        async with self._lock:
            self._runs[run_id] = record

        asyncio.create_task(self._execute(run_id, command))
        return record

    async def _append_event(self, run_id: str, event_type: str, message: str) -> None:
        async with self._lock:
            record = self._runs.get(run_id)
            if not record:
                return
            record.events.append(RunEvent(type=event_type, message=message))

    async def _update_status(
        self, run_id: str, status: str, error: Optional[str] = None
    ) -> None:
        async with self._lock:
            record = self._runs.get(run_id)
            if not record:
                return
            record.status = status
            if status == "running":
                record.started_at = now_iso()
            if status in {"succeeded", "failed"}:
                record.finished_at = now_iso()
            record.error = error

    async def _execute(self, run_id: str, command: List[str]) -> None:
        await self._update_status(run_id, "running")
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            await self._append_event(run_id, "error", f"Command not found: {exc}")
            await self._update_status(run_id, "failed", error=str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            await self._append_event(run_id, "error", f"Failed to start: {exc}")
            await self._update_status(run_id, "failed", error=str(exc))
            return

        async def _drain(stream, label: str) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                await self._append_event(run_id, label, line.decode(errors="ignore").rstrip())

        stdout_task = asyncio.create_task(_drain(proc.stdout, "stdout"))
        stderr_task = asyncio.create_task(_drain(proc.stderr, "stderr"))

        await asyncio.wait([stdout_task, stderr_task])
        return_code = await proc.wait()
        await self._append_event(run_id, "status", f"Process exited with code {return_code}")

        if return_code == 0:
            await self._update_status(run_id, "succeeded")
        else:
            await self._update_status(run_id, "failed", error=f"Exit code {return_code}")


run_manager = RunManager()
