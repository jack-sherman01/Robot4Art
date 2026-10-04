"""Locate a usable Isaac Sim standalone installation on this machine.

Robot4Art does not bundle or depend on Isaac Sim itself, and does not assume
any other project's install path -- it only needs *a* working standalone
install, found independently at run time.

Resolution order:
  1. $ROBOT4ART_ISAAC_SIM_PATH, if set (must point at a valid install root).
  2. Common per-user locations under the current user's own home.

Deliberately does NOT search other accounts' home directories, even ones
that happen to be readable -- this machine is shared with other users, and
their installs are theirs, not a resource for this project to depend on.
If no install is found under the current user's own home, install one
there (see sim/README.md) rather than reaching into someone else's.

A directory counts as a valid install if it contains an executable
``python.sh`` (Isaac Sim's bundled Python launcher) and an
``isaac-sim.sh`` entry point.
"""

import os


def _is_valid_install(path: str) -> bool:
    python_sh = os.path.join(path, "python.sh")
    launch_sh = os.path.join(path, "isaac-sim.sh")
    return os.path.isfile(python_sh) and os.access(python_sh, os.X_OK) and os.path.isfile(launch_sh)


def find_isaac_sim_path() -> str:
    env_path = os.environ.get("ROBOT4ART_ISAAC_SIM_PATH")
    if env_path:
        if _is_valid_install(env_path):
            return env_path
        raise RuntimeError(
            f"ROBOT4ART_ISAAC_SIM_PATH={env_path!r} does not look like a valid Isaac "
            "Sim install (expected python.sh and isaac-sim.sh in that directory)."
        )

    home = os.path.expanduser("~")
    own_candidates = [
        os.path.join(home, "isaacsim"),
        os.path.join(home, "isaac-sim"),
    ]
    for candidate in own_candidates:
        if _is_valid_install(candidate):
            return candidate

    raise RuntimeError(
        "Could not find an Isaac Sim installation under this account "
        f"({home}/isaacsim or {home}/isaac-sim). This project does not search "
        "other users' home directories. Install Isaac Sim under this account "
        "(see sim/README.md) or set ROBOT4ART_ISAAC_SIM_PATH."
    )


if __name__ == "__main__":
    print(find_isaac_sim_path())
