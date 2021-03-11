FROM continuumio/anaconda3
RUN apt-get update
RUN apt-get -y install gcc g++ make graphviz
RUN apt-get -y autoclean
RUN conda install -c conda-forge jupyterhub networkx pandas numpy bqplot colour bitarray sympy cython tatsu psutil openjdk prince-factor-analysis pydot
RUN conda clean --all --yes
RUN pip install --no-cache-dir git+https://github.com/fpom/pytl.git
RUN pip install --no-cache-dir git+https://github.com/fpom/cunf-ptnet-py3
RUN pip install --no-cache-dir git+https://github.com/fpom/pyddd.git
RUN pip install --no-cache-dir git+https://github.com/fpom/pyits.git
RUN pip install --no-cache-dir git+https://forge.ibisc.univ-evry.fr/cthomas/pyits_model_checker.git
RUN pip install --no-cache-dir git+https://github.com/fpom/ecco.git
RUN rm -rf /tmp/pip-*
ADD doc/* /etc/skel/doc/
RUN useradd --skel /etc/skel --create-home --password ec2saXpWqj10U --user-group ecco
