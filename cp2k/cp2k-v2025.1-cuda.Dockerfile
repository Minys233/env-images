# This file is modified from /https://github.com/cp2k/cp2k-containers/raw/refs/heads/master/docker/2025.1_openmpi_native_cuda_A100_psmp.Dockerfile
# By Yaosen Min @ 2025/04/05
# Difference from original:
#   1. Bypass GFW with a mirror site.
#   2. Use more cpus for compiling.
#   3. Install conda and other tools for development


# Stage 1: build step
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS build

# Setup CUDA environment
ENV CUDA_PATH=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# Disable JIT cache as there seems to be an issue with file locking on overlayfs
# See also https://github.com/cp2k/cp2k/pull/2337
ENV CUDA_CACHE_DISABLE=1

# Install packages required for the CP2K toolchain build
RUN apt-get update -qq && apt-get install -qq --no-install-recommends \
    g++ gcc gfortran openssh-client python3 libtool libtool-bin \
    bzip2 ca-certificates git make patch pkg-config unzip wget zlib1g-dev xz-utils

# Download CP2K
# NOTE: this is a workaround for the GFW rather than clone from github
# RUN tee ~/.gitconfig <<-'EOF'
# [url "https://ghfast.top/https://github.com/"]
#     insteadOf = https://github.com/
# EOF
COPY cp2k-v2025.1.zip /tmp/cp2k.zip
RUN unzip /tmp/cp2k.zip -d /opt/ && mv /opt/cp2k-v2025.1 /opt/cp2k
# RUN git clone --recursive -b support/v2025.1 https://github.com/cp2k/cp2k.git /opt/cp2k


# Patch CP2K build script with dftd4-3.6.0 from github
RUN tee /opt/cp2k/tools/toolchain/patch.sh <<-'EOF'
wget -O /tmp/dftd4-3.6.0.tar.xz -nc https://github.com/dftd4/dftd4/releases/download/v3.6.0/dftd4-3.6.0-source.tar.xz
tar -xf /tmp/dftd4-3.6.0.tar.xz -C /tmp/ && \
[ -d /tmp/dftd4-3.6.0 ] && echo "Patch: /tmp/dftd4-3.6.0 patch exists -- OK" && \
rm -rf /opt/cp2k/tools/toolchain/build/dftd4-3.6.0 && \
mv /tmp/dftd4-3.6.0 /opt/cp2k/tools/toolchain/build/
EOF

RUN sed -i "/tar -xzf dftd4-\${dftd4_ver}\.tar\.gz/a bash /opt/cp2k/tools/toolchain/patch.sh" /opt/cp2k/tools/toolchain/scripts/stage8/install_dftd4.sh

# # Build CP2K toolchain for target CPU native
WORKDIR /opt/cp2k/tools/toolchain
RUN /bin/bash -c -o pipefail \
    "./install_cp2k_toolchain.sh -j 64 \
     --install-all \
     --enable-cuda=yes --gpu-ver=A100 --with-deepmd=no --with-libtorch=no \
     --target-cpu=native \
     --with-cusolvermp=yes \
     --with-gcc=system \
     --with-openmpi=install \
     --with-cusolvermp=install"
# install cusolvermp and dependencies ref to https://github.com/cp2k/cp2k/issues/2986
RUN wget -O /tmp/ucx-1.18.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 \
    https://ghfast.top/https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 && \
    tar -xjf /tmp/ucx-1.18.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 -C /tmp/ && \
    dpkg -i /tmp/ucx-1.18.0.deb /tmp/ucx-cuda-1.18.0.deb

RUN wget -O /tmp/ucc-1.3.0.tar.gz \
    https://ghfast.top/https://github.com/openucx/ucc/archive/refs/tags/v1.3.0.tar.gz && \
    tar -xzf /tmp/ucc-1.3.0.tar.gz -C /tmp/ && \
    apt-get update && apt-get install -y autoconf
WORKDIR /tmp/ucc-1.3.0
RUN ./autogen.sh && \
    ./configure --prefix=/usr --with-cuda="$CUDA_PATH" --with-ucx=/usr --with-nvcc-gencode='-arch sm_80' && \
    make -j 64 && make install 
    
# with Dianxin's machine, sometime we cannot apt update due to:
# W: Failed to fetch http://archive.ubuntu.com/ubuntu/dists/jammy/InRelease  Couldn't create temporary file /tmp/apt.conf.wMk092 for passing config to apt-key 
# So apt-get update should not be used with &&
RUN wget -O /tmp/cuda-keyring_1.1-1_all.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring_1.1-1_all.deb && \
    apt-get update; \
    apt-get -y install cusolvermp-cuda-12

RUN wget -O /tmp/libcal.tar.xz https://developer.download.nvidia.cn/compute/cublasmp/redist/libcal/linux-x86_64/libcal-linux-x86_64-0.4.4.50_cuda12-archive.tar.xz && \
    tar xvf /tmp/libcal.tar.xz -C /tmp/ && \
    cp -r /tmp/libcal-linux-x86_64-0.4.4.50_cuda12-archive/include/* /usr/include/ && \
    cp -r /tmp/libcal-linux-x86_64-0.4.4.50_cuda12-archive/lib/* /usr/lib/
# patch /opt/cp2k/src/fm/cp_fm_cusolver.c:375:13 unused ldB variable will cause error due to -Werror=unused-parameter
RUN sed -i 's/const int ldB = b_matrix_desc\[8\];//' /opt/cp2k/src/fm/cp_fm_cusolver.c


# Build CP2K for target CPU/GPU native
# See https://github.com/cp2k/cp2k/blob/master/INSTALL.md for the meanings of VERSION
# See https://dashboard.cp2k.org/index.html for comparisons
WORKDIR /opt/cp2k
RUN /bin/bash -c -o pipefail \
    "cp /opt/cp2k/tools/toolchain/install/arch/* /opt/cp2k/arch/ && \
    source /opt/cp2k/tools/toolchain/install/setup && \
    make -j 64 ARCH=local_cuda VERSION=psmp"

    
# Collect components for installation and remove symbolic links
RUN /bin/bash -c -o pipefail \
    "mkdir -p /toolchain/install /toolchain/scripts; \
    for libdir in \$(ldd ./exe/local_cuda/cp2k.psmp | \
                    grep /opt/cp2k/tools/toolchain/install | \
                    awk '{print \$3}' | cut -d/ -f7 | \
                    sort | uniq) setup; do \
        cp -ar /opt/cp2k/tools/toolchain/install/\${libdir} /toolchain/install; \
    done; \
    cp /opt/cp2k/tools/toolchain/scripts/tool_kit.sh /toolchain/scripts; \
    unlink ./exe/local_cuda/cp2k.popt; \
    unlink ./exe/local_cuda/cp2k_shell.psmp"


# Stage 2: install step
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS install
ENV TZ=Asia/Shanghai
ENV CUDA_PATH=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
# Install required packages fix tmp permission issue
RUN chmod 1777 /tmp && apt-get update -qq && apt-get install -qq --no-install-recommends \
    g++ gcc gfortran openssh-client python3 libtool libtool-bin \
    bzip2 ca-certificates git make patch pkg-config unzip wget zlib1g-dev && rm -rf /var/lib/apt/lists/*


# install cusolvermp and dependencies ref to https://github.com/cp2k/cp2k/issues/2986
RUN wget -O /tmp/ucx-1.18.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 \
    https://ghfast.top/https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 && \
    tar -xjf /tmp/ucx-1.18.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 -C /tmp/ && \
    dpkg -i /tmp/ucx-1.18.0.deb /tmp/ucx-cuda-1.18.0.deb

RUN wget -O /tmp/ucc-1.3.0.tar.gz \
    https://ghfast.top/https://github.com/openucx/ucc/archive/refs/tags/v1.3.0.tar.gz && \
    tar -xzf /tmp/ucc-1.3.0.tar.gz -C /tmp/ && \
    apt-get update && apt-get install -y autoconf
WORKDIR /tmp/ucc-1.3.0
RUN ./autogen.sh && \
    ./configure --prefix=/usr --with-cuda="$CUDA_PATH" --with-ucx=/usr --with-nvcc-gencode='-arch sm_80' && \
    make -j 64 && make install
    
# with Dianxin's machine, sometime we cannot apt update due to:
# W: Failed to fetch http://archive.ubuntu.com/ubuntu/dists/jammy/InRelease  Couldn't create temporary file /tmp/apt.conf.wMk092 for passing config to apt-key 
# So apt-get update should not be used with &&
RUN wget -O /tmp/cuda-keyring_1.1-1_all.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring_1.1-1_all.deb && \
    apt-get update; \
    apt-get -y install cusolvermp-cuda-12

RUN wget -O /tmp/libcal.tar.xz https://developer.download.nvidia.cn/compute/cublasmp/redist/libcal/linux-x86_64/libcal-linux-x86_64-0.4.4.50_cuda12-archive.tar.xz && \
    tar xvf /tmp/libcal.tar.xz -C /tmp/ && \
    cp -r /tmp/libcal-linux-x86_64-0.4.4.50_cuda12-archive/include/* /usr/include/ && \
    cp -r /tmp/libcal-linux-x86_64-0.4.4.50_cuda12-archive/lib/* /usr/lib/

# Cleanup
WORKDIR /
RUN rm -rf /tmp/ucx-1.18.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 /tmp/*.deb \
    /tmp/ucc-1.3.0.tar.gz /tmp/ucc-1.3.0 \
    /tmp/cuda-keyring_1.1-1_all.deb \
    /tmp/libcal.tar.xz /tmp/libcal-linux-x86_64-0.4.4.50_cuda12-archive


# Install CP2K binaries
COPY --from=build /opt/cp2k/exe/local_cuda/ /opt/cp2k/exe/local_cuda/

# Install CP2K regression tests
COPY --from=build /opt/cp2k/tests/ /opt/cp2k/tests/
COPY --from=build /opt/cp2k/tools/regtesting/ /opt/cp2k/tools/regtesting/
COPY --from=build /opt/cp2k/src/grid/sample_tasks/ /opt/cp2k/src/grid/sample_tasks/

# Install CP2K database files
COPY --from=build /opt/cp2k/data/ /opt/cp2k/data/

# Install shared libraries required by the CP2K binaries
COPY --from=build /toolchain/ /opt/cp2k/tools/toolchain/

# Create links to CP2K binaries
RUN /bin/bash -c -o pipefail \
    "for binary in cp2k dumpdcd graph xyz2dcd; do \
        ln -sf /opt/cp2k/exe/local_cuda/\${binary}.psmp \
               /usr/local/bin/\${binary}; \
     done; \
     ln -sf /opt/cp2k/exe/local_cuda/cp2k.psmp \
            /usr/local/bin/cp2k_shell; \
     ln -sf /opt/cp2k/exe/local_cuda/cp2k.psmp \
            /usr/local/bin/cp2k.popt"

# Create entrypoint script file
RUN printf "#!/bin/bash\n\
ulimit -c 0 -s unlimited\n\
export CUDA_CACHE_DISABLE=1\n\
export CUDA_PATH=/usr/local/cuda\n\
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:\${CUDA_PATH}/lib64\n\
export OMPI_ALLOW_RUN_AS_ROOT=1\n\
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1\n\
export OMPI_MCA_btl_vader_single_copy_mechanism=none\n\
export OMP_STACKSIZE=16M\n\
export PATH=/opt/cp2k/exe/local_cuda:\${PATH}\n\
source /opt/cp2k/tools/toolchain/install/setup\n\
\"\$@\"" \
>/usr/local/bin/entrypoint.sh && chmod 755 /usr/local/bin/entrypoint.sh

# Create shortcut for regression test
RUN printf "/opt/cp2k/tests/do_regtest.py --mpiexec \"mpiexec --bind-to none\" --maxtasks 8 --workbasedir /mnt \$* /opt/cp2k/exe/local_cuda psmp" \
>/usr/local/bin/run_tests && chmod 755 /usr/local/bin/run_tests

# Define entrypoint
WORKDIR /mnt
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["cp2k", "--help"]

# Label docker image
LABEL author="CP2K Developers" \
      cp2k_version="2025.1" \
      dockerfile_generator_version="0.2"

# EOF
