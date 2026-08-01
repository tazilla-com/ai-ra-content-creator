"""`harness.audit` — read-only instrumentation used to establish which code actually runs.

Added for the code-archaeology audit. Nothing here is imported by the application; every
module is a standalone entry point run with ``python -m harness.audit.<name>``.

Three tools:

- ``exec_probe``    function-level execution tracer (stdlib ``sys.setprofile``, no new deps)
- ``inventory``     every function/method in the tree, from the AST — the denominator
- ``surface_drive`` drives the real ASGI app in-process over every route, under the probe
- ``langfuse_recon`` read-only dump of the hosted prompt registry and past traces

The point of the pair (inventory, exec_probe) is the difference between them: functions that
exist but were never called under real traffic. That is evidence static analysis cannot give,
because a module can be imported and still never execute.
"""
