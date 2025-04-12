# This file is modified from /https://github.com/cp2k/cp2k-containers/raw/refs/heads/master/docker/2025.1_openmpi_native_psmp.Dockerfile
# By Yaosen Min @ 2025/04/05
# Difference from original:
#   1. Bypass GFW with a mirror site.
#   2. Use more cpus for compiling.
#   3. Install conda and other tools for development


# Stage 1: build step
FROM ubuntu:22.04 AS build

# Install packages required for the CP2K toolchain build
RUN apt-get update -qq && apt-get install -qq --no-install-recommends \
    g++ gcc gfortran openssh-client python3 libtool libtool-bin \
    bzip2 ca-certificates git make patch pkg-config unzip wget zlib1g-dev xz-utils

# Download CP2K
# NOTE: this is a workaround for the GFW rather than clone from github
RUN tee ~/.gitconfig <<-'EOF'
[url "https://ghfast.top/https://github.com/"]
    insteadOf = https://github.com/
EOF
COPY cp2k-v2025.1.zip /tmp/cp2k.zip
RUN unzip /tmp/cp2k.zip -d /opt/ && mv /opt/cp2k-v2025.1 /opt/cp2k
# RUN git clone --recursive -b support/v2025.1 https://github.com/cp2k/cp2k.git /opt/cp2k


# Patch CP2K build script with dftd4-3.6.0 from github
RUN tee /opt/cp2k/tools/toolchain/patch.sh <<-'EOF'
wget -O /tmp/dftd4-3.6.0.tar.xz -nc https://ghfast.top/https://github.com/dftd4/dftd4/releases/download/v3.6.0/dftd4-3.6.0-source.tar.xz
tar -xf /tmp/dftd4-3.6.0.tar.xz -C /tmp/ && \
[ -d /tmp/dftd4-3.6.0 ] && echo "Patch: /tmp/dftd4-3.6.0 patch exists -- OK" && \
rm -rf /opt/cp2k/tools/toolchain/build/dftd4-3.6.0 && \
mv /tmp/dftd4-3.6.0 /opt/cp2k/tools/toolchain/build/
EOF

RUN sed -i "/tar -xzf dftd4-\${dftd4_ver}\.tar\.gz/a bash /opt/cp2k/tools/toolchain/patch.sh" /opt/cp2k/tools/toolchain/scripts/stage8/install_dftd4.sh

# Build CP2K toolchain for target CPU native
WORKDIR /opt/cp2k/tools/toolchain
RUN /bin/bash -c -o pipefail \
    "./install_cp2k_toolchain.sh -j 64 \
     --install-all \
     --enable-cuda=no --with-deepmd=no --with-libtorch=no \
     --target-cpu=native \
     --with-cusolvermp=yes \
     --with-gcc=system \
     --with-openmpi=install"

# Build CP2K for target CPU native
# See https://github.com/cp2k/cp2k/blob/master/INSTALL.md for the meanings of VERSION
# See https://dashboard.cp2k.org/index.html for comparisons
WORKDIR /opt/cp2k
RUN /bin/bash -c -o pipefail \
    "cp /opt/cp2k/tools/toolchain/install/arch/* /opt/cp2k/arch/ && \
    source /opt/cp2k/tools/toolchain/install/setup && \
    make -j 64 ARCH=local VERSION=psmp"

# Collect components for installation and remove symbolic links
RUN /bin/bash -c -o pipefail \
    "mkdir -p /toolchain/install /toolchain/scripts; \
     for libdir in \$(ldd ./exe/local/cp2k.psmp | \
                      grep /opt/cp2k/tools/toolchain/install | \
                      awk '{print \$3}' | cut -d/ -f7 | \
                      sort | uniq) setup; do \
        cp -ar /opt/cp2k/tools/toolchain/install/\${libdir} /toolchain/install; \
     done; \
     cp /opt/cp2k/tools/toolchain/scripts/tool_kit.sh /toolchain/scripts; \
     unlink ./exe/local/cp2k.popt; \
     unlink ./exe/local/cp2k_shell.psmp"

# Stage 2: install step
FROM ubuntu:22.04 AS install
ENV TZ=Asia/Shanghai
# Install required packages
RUN apt-get update -qq && apt-get install -qq --no-install-recommends \
    g++ gcc gfortran openssh-client python3 && rm -rf /var/lib/apt/lists/*

# Install CP2K binaries
COPY --from=build /opt/cp2k/exe/local/ /opt/cp2k/exe/local/

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
        ln -sf /opt/cp2k/exe/local/\${binary}.psmp \
               /usr/local/bin/\${binary}; \
     done; \
     ln -sf /opt/cp2k/exe/local/cp2k.psmp \
            /usr/local/bin/cp2k_shell; \
     ln -sf /opt/cp2k/exe/local/cp2k.psmp \
            /usr/local/bin/cp2k.popt"



# Create entrypoint script file
RUN printf "#!/bin/bash\n\
ulimit -c 0 -s unlimited\n\
\
export OMPI_ALLOW_RUN_AS_ROOT=1\n\
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1\n\
export OMPI_MCA_btl_vader_single_copy_mechanism=none\n\
export OMP_STACKSIZE=16M\n\
export PATH=/opt/cp2k/exe/local:\${PATH}\n\
source /opt/cp2k/tools/toolchain/install/setup\n\
\"\$@\"" \
>/usr/local/bin/entrypoint.sh && chmod 755 /usr/local/bin/entrypoint.sh

# Create shortcut for regression test
RUN printf "/opt/cp2k/tests/do_regtest.py --mpiexec \"mpiexec --bind-to none\" --maxtasks 8 --workbasedir /mnt \$* /opt/cp2k/exe/local psmp" \
>/usr/local/bin/run_tests && chmod 755 /usr/local/bin/run_tests

# Define entrypoint
WORKDIR /mnt
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["cp2k", "--help"]

# Label docker image
LABEL author="CP2K Developers & Yaosen Min" \
      cp2k_version="2025.1" \
      dockerfile_generator_version="0.1"

# EOF

