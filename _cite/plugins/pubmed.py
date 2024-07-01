import json
from urllib.request import Request, urlopen
from urllib.parse import quote
from util import *


def main(entry):
    """
    receives single list entry from pubmed data file
    returns list of sources to cite
    """

    # ncbi api
    endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=$TERM&retmode=json&retmax=1000&usehistory=y"

    # get id from entry
    _id = get_safe(entry, "term", "")
    if not _id:
        raise Exception('No "term" key')

    # query api
    @log_cache
    @cache.memoize(name=__file__, expire=1 * (60 * 60 * 24))
    def query(_id):
        url = endpoint.replace("$TERM", quote(_id))
        request = Request(url=url)
        response = json.loads(urlopen(request).read())
        return get_safe(response, "esearchresult.idlist", [])

    response = query(_id)

    # list of sources to return
    sources = []

    # go through response and format sources
    for _id in response:
        # create source
        source = {"id": f"pubmed:{_id}"}

        # copy fields from entry to source
        source.update(entry)

        # add source to list
        sources.append(source)

    return sources
