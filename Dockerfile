# Standard Linux/x86_64 image -- deliberately uses the SAME pinned package
# versions (torch==2.2.2, stable-baselines3==2.4.1, etc.) that produced the
# results in docs/results.md, rather than newer versions, so results are
# exactly reproducible rather than just "the method still runs."
#
# Note: pybullet==3.2.7 has no published wheel for Python 3.12 on any
# platform (checked against PyPI directly), so it compiles from source here
# -- expect the build step to take a few minutes. Every other pinned
# package has a prebuilt wheel for linux/cp312.
FROM python:3.12-slim

# build-essential: C/C++ toolchain for pybullet's source build.
# git: needed to clone gym-pybullet-drones (not published on PyPI).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the dependency manifest first so Docker can cache this layer
# (the slow part -- pybullet's source build) across rebuilds that only
# change application code.
COPY requirements.txt .

# Same two-step install as README.md's "Installation and Deployment":
# gym-pybullet-drones is installed with --no-deps to skip its own
# overly strict pins (torch>=2.13 etc., which don't match what actually
# produced docs/results.md), then the real, tested pins come from
# requirements.txt.
RUN git clone --depth 1 https://github.com/utiasDSL/gym-pybullet-drones.git external/gym-pybullet-drones \
    && pip install --no-cache-dir -e external/gym-pybullet-drones --no-deps \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: a fast, no-training smoke test proving the whole stack works
# (PID baseline, one episode, a few seconds). Override to run anything else,
# e.g.:
#   docker run --rm quadrotor-hybrid-rl-pid python scripts/train_ppo.py
#   docker run --rm quadrotor-hybrid-rl-pid python scripts/evaluate_robustness.py
CMD ["python", "scripts/run_pid_baseline.py"]
