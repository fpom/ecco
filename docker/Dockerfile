FROM ubuntu:latest
RUN apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt-get -y install gcc g++ make graphviz python3 python3-dev python3-pip openjdk-8-jre-headless git nodejs npm
ADD doc/* /etc/skel/doc/
RUN useradd --skel /etc/skel --create-home --password ec2saXpWqj10U --user-group ecco
RUN apt-get -y upgrade
RUN apt-get -y autoclean
RUN npm install -g configurable-http-proxy
RUN pip3 install --no-cache-dir --upgrade setuptools wheel jupyterhub networkx pandas numpy bqplot colour bitarray sympy cython tatsu psutil prince pydot
RUN pip3 install --no-cache-dir git+https://github.com/fpom/pytl.git
RUN pip3 install --no-cache-dir git+https://github.com/fpom/cunf-ptnet-py3
RUN pip3 install --no-cache-dir git+https://github.com/fpom/pyddd.git
RUN pip3 install --no-cache-dir git+https://github.com/fpom/pyits.git
RUN pip3 install --no-cache-dir git+https://forge.ibisc.univ-evry.fr/cthomas/pyits_model_checker.git
RUN pip3 install --no-cache-dir git+https://github.com/fpom/ecco.git
