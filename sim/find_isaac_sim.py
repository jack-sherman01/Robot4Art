"""Locate a usable Isaac Sim standalone installation on this machine.

Robot4Art does not bundle or depend on Isaac Sim itself, and does not assume
any other project's install path -- it only needs *a* working standalone
install, found independently at run time.

Resolution order:
  1. $ROBOT4ART_ISAAC_SIM_PATH, if set (must point at a valid install root).
  2. Common per-user locations under the current user's own home.
  3. World-readable installs under other accounts' homes, since shared
     GPU workstations often have Isaac Sim installed once per user
     rather than system-wide.

A directory counts as a valid install if it contains an executable
``python.sh`` (Isaac Sim's bundled Python launcher) and an
``isaac-sim.sh`` entry point.
"""

import glob
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
        os.path.join(home, "humanoid", "isaacsim"),
    ]
    for candidate in own_candidates:
        if _is_valid_install(candidate):
            return candidate

    # Shared workstation fallback: look for other accounts' installs that
    # happen to be world-readable/executable.
    shared_patterns = ("/home/*/isaacsim", "/home/*/isaac-sim", "/home/*/*/isaacsim")
    for pattern in shared_patterns:
        for candidate in sorted(glob.glob(pattern)):
            if _is_valid_install(candidate):
                return candidate

    raise RuntimeError(
        "Could not find an Isaac Sim installation. Set ROBOT4ART_ISAAC_SIM_PATH to "
        "the install root (the directory that contains python.sh)."
    )


if __name__ == "__main__":
    print(find_isaac_sim_path())
