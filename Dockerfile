FROM nautronics_core_base2:r36.4.tegra-aarch64-cu126-22.04

WORKDIR /workspace

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
  /usr/local/lib/python3.10/dist-packages/tensorrt-10.3.0-cp310-none-linux_aarch64.whl \
  /usr/local/lib/python3.10/dist-packages/tensorrt_dispatch-10.3.0-cp310-none-linux_aarch64.whl \
  /usr/local/lib/python3.10/dist-packages/tensorrt_lean-10.3.0-cp310-none-linux_aarch64.whl
COPY requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-deps ultralytics

RUN echo "source /opt/ros/humble/install/setup.bash" >> ~/.bashrc && \
    echo "if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi" >> ~/.bashrc