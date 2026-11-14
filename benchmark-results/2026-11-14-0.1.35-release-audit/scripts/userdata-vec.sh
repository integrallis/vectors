#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq curl git unzip python3 python3-pip >/dev/null
install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor -o /etc/apt/keyrings/adoptium.gpg
echo "deb [signed-by=/etc/apt/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print $2}' /etc/os-release) main" > /etc/apt/sources.list.d/adoptium.list
apt-get update -qq && apt-get install -y -qq temurin-25-jdk >/dev/null
sudo -u ubuntu bash -c 'cd ~ && git clone -q https://github.com/integrallis/vectors.git vectors && cd vectors && git fetch -q --tags && mkdir -p ~/data && curl -sL -o ~/data/fashion-mnist-784-euclidean.hdf5 https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5 && curl -sL -o ~/data/sift-128-euclidean.hdf5 https://ann-benchmarks.com/sift-128-euclidean.hdf5; ls -la ~/data > ~/data.ls'
echo SETUP-DONE > /home/ubuntu/SETUP-DONE; chown ubuntu:ubuntu /home/ubuntu/SETUP-DONE
