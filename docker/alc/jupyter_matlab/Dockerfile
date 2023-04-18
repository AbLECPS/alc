# adapted from ninjaben/matlab-support
# 
# Create an image with enough dependencies to support a mounted-in matlab.
# change username 'matlabuser' to the user that can run matlab. 
# If all users can run matlab, then you do not need to change this
#
FROM alc_jupyter_matlab_base

RUN apt-get -y install sudo
#RUN adduser --disabled-password --gecos '' matlabuser 
#RUN adduser matlabuser sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
   
COPY setup_verivital.m /.
COPY setup.sh /.
 
#USER matlabuser

