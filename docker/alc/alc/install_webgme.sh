#!/bin/bash
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.0/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
source $HOME/.nvm/nvm.sh
nvm install 8
nvm install 12
nvm use 8
#mkdir -p /alc/webgme
#mkdir -p /alc/webgme/automate/gradle 
#mkdir -p /alc/webgme/automate/wrapper
#mkdir -p /alc/workflows
#mkdir -p /opt/gradle
pushd /alc/webgme
npm install
popd

apt purge -y openjdk-10-jdk openjdk-9-jdk openjdk-8-jdk java-common

apt-get update && \
    apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        openssh-server \
        openjdk-11-jdk \
        unzip \
        jq \
        wget \
        socat

mv /java.security /etc/java-11-openjdk/security/java.security
dpkg --purge --force-depends ca-certificates-java && \
    apt install -y ca-certificates-java

pushd /opt/gradle 
wget -q https://services.gradle.org/distributions/gradle-5.6.3-bin.zip 
unzip -q gradle-5.6.3-bin.zip 
rm gradle-5.6.3-bin.zip
ln -s gradle-5.6.3 latest
echo 'PATH="$PATH:/opt/gradle/latest/bin"' >> /root/.bashrc
popd

pushd /alc/webgme/automate/wrapper 
/opt/gradle/latest/bin/gradle wrapper --gradle-version=5.6.3 
./gradlew
popd
rm -rf /alc/webgme/automate/wrapper
