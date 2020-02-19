import os
from tqdm import tqdm
from zipfile import ZipFile


def get_zipfiles(directory):
    list_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".zip"):
            list_files.append(os.path.join(directory, filename))
    return list_files


zipfiles = sorted(get_zipfiles('/data/projects/DeepFakeDetection/data'))

for zipfile in zipfiles:
    print(f'Extracting {zipfile}...')
    with ZipFile(file=zipfile) as z:
        for file in tqdm(iterable=z.namelist(), total=len(z.namelist())):
            z.extract(member=file)

    os.remove(zipfile)
