"""Background watcher for automated scenario completion and simulation teardown."""

import argparse
import logging
from pathlib import Path
import threading
import time
from typing import Optional

from control import (
    get_teardown_commands,
    list_compose_containers,
    resolve_runtime_repo_root,
    run_processes,
)

SCENARIO_RUNNER_SERVICE = "carla-scenario-runner-ros-automated-testing"

logger = logging.getLogger(__name__)

# Also ensure watcher events are written to a file for transparent debugging
_log_file_handler = logging.FileHandler("/tmp/openadsim_watcher.log", mode="a", encoding="utf-8")
_log_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(_log_file_handler)
    logger.setLevel(logging.INFO)

_watcher_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_watcher_lock = threading.RLock()


def is_scenario_runner(container: dict[str, str]) -> bool:
    """Check if the given compose container dictionary matches the automated scenario runner."""
    service = container.get("service", "")
    name = container.get("name", "")
    return SCENARIO_RUNNER_SERVICE in service or SCENARIO_RUNNER_SERVICE in name


def watch_scenario_and_teardown(
    repo_root: Path,
    record_bag: bool,
    stop_event: Optional[threading.Event] = None,
    startup_timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 1.5,
    post_scenario_grace_seconds: float = 4.0,
) -> bool:
    """Monitor scenario runner container and trigger teardown when completed.

    Phase 1: Wait up to startup_timeout_seconds for the runner container to enter 'Up' state.
    Phase 2: Poll every poll_interval_seconds until the runner container exits (with consecutive confirmations).
    Phase 3: Wait a short grace period (to let vehicle settle and user see the outcome), then trigger teardown.
    """
    if stop_event is None:
        stop_event = threading.Event()

    logger.info("Auto-stop watcher started (record_bag=%s, timeout=%.0fs)", record_bag, startup_timeout_seconds)

    # Phase 1: Wait for scenario runner to be active ("Up")
    deadline = time.time() + startup_timeout_seconds
    runner_started = False

    while not stop_event.is_set():
        containers = list_compose_containers(repo_root, all_containers=True, include_gui=False)
        runner = next((c for c in containers if is_scenario_runner(c)), None)

        if runner:
            status = runner.get("status", "").lower()
            if status.startswith("up"):
                runner_started = True
                logger.info("Auto-stop watcher: Scenario runner is now running (status: '%s')", runner.get("status", ""))
                break

        if time.time() > deadline:
            logger.warning(
                "Auto-stop watcher: Scenario runner '%s' did not start within %ss timeout",
                SCENARIO_RUNNER_SERVICE,
                startup_timeout_seconds,
            )
            return False

        stop_event.wait(poll_interval_seconds)

    if stop_event.is_set() or not runner_started:
        logger.info("Auto-stop watcher: Aborted during startup phase.")
        return False

    # Phase 2: Poll until scenario runner exits, requiring consecutive confirmations to avoid transient glitches
    consecutive_exited = 0
    REQUIRED_EXITED_CONFIRMATIONS = 2

    while not stop_event.is_set():
        containers = list_compose_containers(repo_root, all_containers=True, include_gui=False)
        runner = next((c for c in containers if is_scenario_runner(c)), None)

        if not runner:
            consecutive_exited += 1
            logger.info("Auto-stop watcher: Runner container not found (confirmation %d/%d)",
                        consecutive_exited, REQUIRED_EXITED_CONFIRMATIONS)
        else:
            status = runner.get("status", "").lower()
            if not status.startswith("up"):
                consecutive_exited += 1
                logger.info("Auto-stop watcher: Runner exited/stopped (status: '%s', confirmation %d/%d)",
                            runner.get("status", ""), consecutive_exited, REQUIRED_EXITED_CONFIRMATIONS)
            else:
                consecutive_exited = 0

        if consecutive_exited >= REQUIRED_EXITED_CONFIRMATIONS:
            logger.info("Auto-stop watcher: Scenario runner completion confirmed.")
            break

        stop_event.wait(poll_interval_seconds)

    if stop_event.is_set():
        logger.info("Auto-stop watcher: Stopped manually before teardown.")
        return False

    # Phase 3: Grace period so user sees the scenario completion before windows close
    if post_scenario_grace_seconds > 0:
        logger.info("Auto-stop watcher: Waiting %.1fs grace period before teardown...", post_scenario_grace_seconds)
        stop_event.wait(post_scenario_grace_seconds)
        if stop_event.is_set():
            logger.info("Auto-stop watcher: Stopped manually during grace period.")
            return False

    # Phase 4: Trigger teardown
    logger.info("Auto-stop watcher: Executing teardown (record_bag=%s)...", record_bag)
    teardown_cmds = get_teardown_commands(record_bag)
    runtime_root = resolve_runtime_repo_root(repo_root)
    ok, output = run_processes(teardown_cmds, cwd=runtime_root)
    if not ok:
        logger.error("Auto-stop watcher: Teardown failed:\n%s", output)
    else:
        logger.info("Auto-stop watcher: Teardown completed successfully.")
    return ok


def start_auto_stop_watcher(repo_root: Path, record_bag: bool) -> bool:
    """Start background watcher thread for scenario runner completion."""
    global _watcher_thread, _stop_event
    with _watcher_lock:
        stop_auto_stop_watcher()
        _stop_event = threading.Event()
        _watcher_thread = threading.Thread(
            target=watch_scenario_and_teardown,
            args=(repo_root, record_bag, _stop_event),
            daemon=True,
            name="auto-stop-watcher",
        )
        _watcher_thread.start()
        return True


def stop_auto_stop_watcher() -> bool:
    """Signal background watcher thread to terminate."""
    global _watcher_thread, _stop_event
    with _watcher_lock:
        if _stop_event:
            _stop_event.set()
        if _watcher_thread and _watcher_thread.is_alive():
            _watcher_thread.join(timeout=3.0)
        _watcher_thread = None
        return True


def is_auto_stop_watcher_running() -> bool:
    """Return whether the auto-stop watcher thread is currently running."""
    global _watcher_thread
    with _watcher_lock:
        return _watcher_thread is not None and _watcher_thread.is_alive()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Watch automated scenario execution and tear down simulation upon completion."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Path to repository root (defaults to current directory)",
    )
    parser.add_argument(
        "--record-bag",
        action="store_true",
        help="Perform graceful teardown (docker compose down) to flush ROS bags",
    )
    args = parser.parse_args()

    print(f"Starting auto-stop watcher (repo_root={args.repo_root}, record_bag={args.record_bag})...")
    success = watch_scenario_and_teardown(args.repo_root, record_bag=args.record_bag)
    if success:
        print("Scenario completed and simulation successfully torn down.")
        exit(0)
    else:
        print("Auto-stop watcher finished without teardown (cancelled or timed out).")
        exit(1)
