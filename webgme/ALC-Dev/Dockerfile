# Dockerfile for running the server itself
FROM node:8.16.2-buster

RUN echo '{"allow_root": true}' > /root/.bowerrc && mkdir -p /root/.config/configstore/ && \
    echo '{}' > /root/.config/configstore/bower-github.json

# Copy build directory contents to webgme directory and install npm packages
RUN mkdir -p /alc/webgme
ADD . /alc/webgme
WORKDIR /alc/webgme
RUN npm install

# Install and update pip, then install any python requirements for python-based plugins
RUN apt-get update
# openjdk and unzip are for gradle installation and execution
RUN apt-get install python-pip openjdk-11-jre unzip jq -y

# Install gradle
RUN mkdir -p /opt/gradle ; cd /opt/gradle ; wget -q https://services.gradle.org/distributions/gradle-5.6.3-bin.zip ; unzip -q gradle-5.6.3-bin.zip ; rm gradle-5.6.3-bin.zip
RUN cd /opt/gradle; ln -s gradle-5.6.3 latest
RUN echo 'PATH="$PATH:/opt/gradle/latest/bin"' >> /root/.bashrc

# create directories for gradle execution environment and build-status output
RUN mkdir -p /alc/webgme/automate/gradle
RUN mkdir -p /alc/workflows

# setup gradle wrapper so that it doesn't happen in production
RUN mkdir -p /alc/webgme/automate/wrapper
RUN cd /alc/webgme/automate/wrapper ; /opt/gradle/latest/bin/gradle wrapper --gradle-version=5.6.3 ; ./gradlew
RUN rm -rf /alc/webgme/automate/wrapper

RUN python -m pip install pip --upgrade --force-reinstall
RUN python -m pip install -r ./python_requirements.txt


EXPOSE 8888

CMD ["/bin/bash", "/alc/webgme/automate/gradle/src/main/shell/startup.sh"]
