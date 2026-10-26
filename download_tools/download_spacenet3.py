import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE = 'https://spacenet-dataset.s3.amazonaws.com/'
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def list_objects(prefix, suffix='.tif'):
    token = None
    while True:
        query = {'list-type': '2', 'prefix': prefix}
        if token:
            query['continuation-token'] = token
        for attempt in range(6):
            try:
                with urllib.request.urlopen(BASE + '?' + urllib.parse.urlencode(query), timeout=90) as response:
                    root = ET.fromstring(response.read())
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        for obj in root.findall('s:Contents', NS):
            key = obj.findtext('s:Key', namespaces=NS)
            if key.endswith(suffix):
                yield {'key': key, 'size': int(obj.findtext('s:Size', namespaces=NS)),
                       'etag': obj.findtext('s:ETag', namespaces=NS).strip('"')}
        token = root.findtext('s:NextContinuationToken', namespaces=NS)
        if not token:
            break


def download(item, output):
    target = output / item['relative_path']
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + '.part')
    for attempt in range(6):
        try:
            digest = hashlib.md5()
            if target.exists() and target.stat().st_size == item['size']:
                with target.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                if '-' in item['etag'] or digest.hexdigest() == item['etag']:
                    return item['size']
            digest = hashlib.md5()
            with urllib.request.urlopen(BASE + urllib.parse.quote(item['key'], safe='/'), timeout=120) as response, part.open('wb') as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    stream.write(chunk)
                    digest.update(chunk)
            if part.stat().st_size != item['size']:
                raise ValueError('Size mismatch: ' + item['key'])
            if '-' not in item['etag'] and digest.hexdigest() != item['etag']:
                raise ValueError('MD5 mismatch: ' + item['key'])
            os.replace(part, target)
            return item['size']
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    items = []
    for split in ('train', 'test_public'):
        prefix = f'spacenet/SN3_roads/{split}/AOI_2_Vegas/PS-RGB/'
        objects = list(list_objects(prefix))
        print(f'{split}: {len(objects)} TIFFs, {sum(x["size"] for x in objects) / 1024**3:.3f} GiB', flush=True)
        for item in objects:
            filename = Path(item['key']).name
            if filename in ('', '.', '..'):
                raise ValueError('Invalid filename')
            item['relative_path'] = f'{split}/AOI_2_Vegas/PS-RGB/{filename}'
            items.append(item)
    if not items:
        raise RuntimeError('No imagery found')
    (output / 'manifest.json').write_text(json.dumps({'source': BASE, 'objects': items}, indent=2) + '\n')
    total = sum(item['size'] for item in items)
    print(f'TOTAL: {len(items)} TIFFs, {total / 1024**3:.3f} GiB', flush=True)
    started = time.monotonic()
    downloaded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(download, item, output): item for item in items}
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            downloaded += future.result()
            if count % 100 == 0 or count == len(items):
                elapsed = time.monotonic() - started
                print(f'{count}/{len(items)} verified; {downloaded / 1024**3:.3f}/{total / 1024**3:.3f} GiB; {downloaded / 1024**2 / elapsed:.2f} MiB/s', flush=True)
    print('COMPLETE: all TIFFs downloaded and verified.', flush=True)


if __name__ == '__main__':
    main()
