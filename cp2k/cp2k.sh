# ghfast.top as github mirror site
# sudo apt update
# sudo apt install -y wget
# wget -O /tmp/apptainer.deb https://ghfast.top/https://github.com/apptainer/apptainer/releases/download/v1.4.0/apptainer_1.4.0_amd64.deb
# sudo apt install -y /tmp/apptainer.deb


# set up docker mirror sites
sudo tee /etc/docker/daemon.json <<-'EOF'
{
    "registry-mirrors": [
     "https://docker.m.daocloud.io",
     "https://docker.imgdb.de",
     "https://docker-0.unsee.tech",
     "https://docker.hlmirror.com",
     "https://docker.1ms.run",
     "https://func.ink",
     "https://lispy.org",
     "https://docker.xiaogenban1993.com"
    ]
}
EOF


