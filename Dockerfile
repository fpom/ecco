FROM ubuntu:latest
RUN apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt-get -y install gcc g++ make graphviz python3 python3-dev python3-pip openjdk-8-jre-headless git nodejs npm
ADD --chown=root:root doc/* /etc/skel/doc/
ADD --chown=root:root . /root/ecco.git/
RUN useradd --skel /etc/skel --create-home --password ec2saXpWqj10U --user-group ecco
RUN apt-get -y upgrade
RUN apt-get -y autoclean
RUN npm install -g configurable-http-proxy
RUN pip3 install --no-cache-dir --upgrade setuptools wheel jupyterhub networkx pandas numpy bqplot colour bitarray sympy cython tatsu psutil prince pydot python-igraph ipycytoscape
RUN pip3 install --no-cache-dir git+https://github.com/fpom/pytl.git
RUN pip3 install --no-cache-dir git+https://github.com/fpom/cunf-ptnet-py3.git
RUN pip3 install --no-cache-dir git+https://github.com/fpom/pyddd.git
RUN ln -s /usr/lib/python3.8/site-packages/* /usr/local/lib/python3.8/dist-packages/
RUN pip3 install --no-cache-dir git+https://github.com/fpom/pyits.git
RUN pip3 install --no-cache-dir git+https://forge.ibisc.univ-evry.fr/cthomas/pyits_model_checker.git
RUN pip3 install --no-cache-dir /root/ecco.git
RUN git clone https://github.com/giannkas/ecofolder.git /root/ecofolder.git
RUN cd /root/ecofolder.git && make clean && make
RUN cp -a /root/ecofolder.git/ecofolder /usr/local/bin/
RUN cp -a /root/ecofolder.git/mci2dot /usr/local/bin/
RUN cp -a /root/ecofolder.git/mci2dot_ev /usr/local/bin/
RUN cp -a /root/ecofolder.git/pr_encoding /usr/local/bin/
RUN cp -a /root/ecofolder.git/rs_complement /usr/local/bin/
RUN rm -rf /root/*.git
