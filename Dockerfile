FROM ubuntu:latest
RUN apt-get -y update
RUN apt-get -y upgrade
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata apt-utils
RUN apt-get -y install --no-install-recommends gcc-11 g++-11 make graphviz python3 python3-dev python3-pip python3-venv openjdk-8-jre-headless git nodejs npm wget textlive-full
ADD --chown=root:root doc/* /etc/skel/doc/
RUN apt-get -y autoclean
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 10
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 10
RUN update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-11 10
RUN useradd --skel /etc/skel --create-home --password ec2saXpWqj10U --user-group ecco
RUN npm install -g configurable-http-proxy
WORKDIR /usr/local
RUN git clone https://github.com/fpom/ecofolder.git ecofolder.git
WORKDIR /usr/local/ecofolder.git
RUN make clean && make
RUN cp -a ecofolder mci2dot mci2dot_ev pr_encoding mci2csv rs_complement /usr/local/bin/
WORKDIR /usr/local
RUN rm -rf ecofolder.git
SHELL ["/usr/bin/bash", "--login", "-c"]
RUN python3 -m venv /usr/local/ecco.venv
RUN echo '. /usr/local/ecco.venv/bin/activate' >> /etc/profile.d/ecco.sh
RUN echo 'export CC="gcc"' >> /etc/profile.d/ecco.sh
RUN echo 'export CXX="g++"' >> /etc/profile.d/ecco.sh
RUN echo 'export LD_LIBRARY_PATH=/usr/local/ecco.venv/lib' >> /etc/profile.d/ecco.sh
WORKDIR /usr/local/ecco.venv
RUN wget -nv https://github.com/fpom/pyddd/raw/master/libDDD.tar.gz && tar xzf libDDD.tar.gz && rm libDDD.tar.gz
RUN wget -nv https://github.com/fpom/pyits/raw/master/libITS.tar.gz && tar xzf libITS.tar.gz && rm libITS.tar.gz
RUN pip3 install --no-cache-dir git+https://github.com/fpom/ecco.git
