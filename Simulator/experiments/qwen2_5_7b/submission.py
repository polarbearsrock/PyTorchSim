"""Read complete launch metadata without adding a simulated DEVICE_SYNC."""
import threading


def phase_kernel_ids(session, command_offset, enqueue_host_task, timeout=30):
    """Fence the host submission stream, then snapshot this phase's kernel IDs.

    PyTorchSim's launch_model returns before its stream worker necessarily
    appends the final launch command. Enqueueing a host marker after that work
    waits only for command submission, not for simulated kernel completion.
    """
    submitted = threading.Event()
    enqueue_host_task(submitted.set)
    if not submitted.wait(timeout):
        raise TimeoutError("Host kernel submissions did not reach the phase boundary")
    return [int(line.split(",")[1])
            for line in session.trace_log[command_offset:].splitlines()
            if line.startswith("LAUNCH_KERNEL,")]
