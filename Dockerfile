FROM ubuntu:latest
RUN apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt-get -y install gcc g++ make graphviz python3 python3-dev python3-pip openjdk-8-jre-headless git nodejs npm wget
ADD --chown=root:root doc/* /etc/skel/doc/
ADD --chown=root:root . /root/ecco.git/
RUN useradd --skel /etc/skel --create-home --password ec2saXpWqj10U --user-group ecco
RUN apt-get -y upgrade
RUN apt-get -y autoclean
RUN npm install -g configurable-http-proxy
RUN pip3 install --no-cache-dir --upgrade setuptools wheel jupyterhub jupyter networkx pandas numpy bqplot colour bitarray sympy cython tatsu==5.5.0 psutil prince pydot python-igraph ipycytoscape unidecode scikit-learn
RUN pip3 install --no-cache-dir git+https://github.com/fpom/pytl.git
RUN pip3 install --no-cache-dir git+https://github.com/fpom/cunf-ptnet-py3.git
RUN cd /root && wget -nv https://github.com/fpom/pyddd/raw/master/libDDD.tar.gz
RUN cd /usr/local && tar xzf /root/libDDD.tar.gz
RUN rm -f /root/libDDD.tar.gz
RUN ldconfig
RUN pip3 install --no-cache-dir git+https://github.com/fpom/pyddd.git
RUN pip3 install --no-cache-dir git+https://github.com/fpom/pyits.git
RUN pip3 install --no-cache-dir git+https://github.com/fpom/pymc.git
RUN pip3 install --no-cache-dir /root/ecco.git
RUN git clone https://github.com/fpom/ecofolder.git /root/ecofolder.git
RUN cd /root/ecofolder.git && make clean && make
RUN cp -a /root/ecofolder.git/ecofolder /root/ecofolder.git/mci2dot root/ecofolder.git/mci2dot_ev /root/ecofolder.git/pr_encoding /root/ecofolder.git/mci2csv /root/ecofolder.git/rs_complement /usr/local/bin/
RUN rm -rf /root/*.git
