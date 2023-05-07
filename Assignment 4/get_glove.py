import os
import io
import requests
import zipfile

import tqdm


FILENAME = "glove.6B.zip"

def download(url):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with tqdm.tqdm(
        desc=url,
        total=total,
        unit='b',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            bar.update(len(chunk))
            yield chunk

if not os.path.exists(FILENAME):
	bio = io.BytesIO()

	for chunk in download("http://nlp.stanford.edu/data/glove.6B.zip"):
	    bio.write(chunk)

	f = open(FILENAME, 'wb')
	f.write(bio.getbuffer())
	f.close()


with zipfile.ZipFile(FILENAME, 'r') as zip_ref:
    zip_ref.extractall(".")
