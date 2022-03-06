# coding: utf-8
form goatools.base import get_godag
from goatools.base import get_godag
from goatools.base import download_ncbi_associations
from goatools.anno.genetogo_reader import Gene2GoReader
gene2go = download_ncbi_associations()
g2go_baker = Gene2GoReader(gene2go, taxids = [4932])
g2go_baker = Gene2GoReader(gene2go, taxids = [545124])
g2go_baker = Gene2GoReader(gene2go, taxids = [559292])
# bakers-yeast 559292, fission = 4892
%save -r yeast_go.py 1-10
