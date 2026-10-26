import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
import re
import runpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('downloader', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    helpers = runpy.run_path(str(args.downloader.resolve()))
    output = args.output.resolve()
    prefix = 'spacenet/SN3_roads/train/AOI_2_Vegas/'
    items = []
    for folder in ('geojson_roads', 'geojson_roads_speed'):
        objects = list(helpers['list_objects'](prefix + folder + '/', '.geojson'))
        print(f'{folder}: {len(objects)} files', flush=True)
        for item in objects:
            item['relative_path'] = 'train/AOI_2_Vegas/' + folder + '/' + Path(item['key']).name
            items.append(item)
    csv_objects = list(helpers['list_objects'](prefix, '.csv'))
    for item in csv_objects:
        if '/' not in item['key'][len(prefix):]:
            item['relative_path'] = 'train/AOI_2_Vegas/' + Path(item['key']).name
            items.append(item)
    (output / 'labels_manifest.json').write_text(json.dumps({'source': helpers['BASE'], 'objects': items}, indent=2) + '\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(helpers['download'], item, output) for item in items]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if count % 250 == 0 or count == len(items):
                print(f'{count}/{len(items)} label files downloaded and verified', flush=True)
    imagery = json.loads((output / 'manifest.json').read_text())['objects']
    train = [item for item in imagery if item['relative_path'].startswith('train/')]
    image_ids = {re.search(r'_img(\d+)\.tif$', item['key']).group(1) for item in train}
    report = {'training_images_expected': len(train), 'labels': {}}
    for folder in ('geojson_roads', 'geojson_roads_speed'):
        files = list((output / 'train' / 'AOI_2_Vegas' / folder).glob('*.geojson'))
        ids = set()
        features = 0
        for path in files:
            data = json.loads(path.read_text())
            if data.get('type') != 'FeatureCollection':
                raise ValueError('Unexpected GeoJSON structure: ' + str(path))
            features += len(data['features'])
            ids.add(re.search(r'_img(\d+)\.geojson$', path.name).group(1))
        missing = sorted(image_ids - ids)
        extra = sorted(ids - image_ids)
        report['labels'][folder] = {'files': len(files), 'features': features, 'missing_image_ids': missing, 'extra_image_ids': extra}
        if extra:
            raise ValueError(f'{folder}: unexpected label image IDs')
    csv_path = output / 'train/AOI_2_Vegas/SN3_roads_train_AOI_2_Vegas_road_labels.csv'
    with csv_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        report['road_csv_columns'] = reader.fieldnames
        rows = list(reader)
        report['road_csv_rows'] = len(rows)
        csv_ids = {re.search(r'_img(\d+)$', row['ImageId']).group(1) for row in rows}
        if image_ids != csv_ids:
            raise ValueError('Road CSV image/label mismatch')
    empty_ids = {image_id for image_id in image_ids if all(row['WKT_Pix'].strip().upper() == 'LINESTRING EMPTY' for row in rows if re.search(r'_img(\d+)$', row['ImageId']).group(1) == image_id)}
    report['empty_road_image_ids'] = sorted(empty_ids, key=int)
    for folder, result in report['labels'].items():
        unexpected_missing = set(result['missing_image_ids']) - empty_ids
        if unexpected_missing:
            raise ValueError(f'{folder}: nonempty-road images lack GeoJSON: {unexpected_missing}')
        result['absent_geojson_for_empty_road_images'] = result.pop('missing_image_ids')
    report['status'] = 'all training images covered by official CSV; absent GeoJSON only for CSV-confirmed empty-road images'
    (output / 'labels_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    print('COMPLETE: labels downloaded, checksummed, parsed and matched to training imagery manifest.', flush=True)


if __name__ == '__main__':
    main()
