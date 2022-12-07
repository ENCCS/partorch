import zipfile
from pathlib import Path

def main():
    rootdir = Path('python')

    files_to_archive = ['dataset', 'partorch', 'scripts', 'environment.yml', 'README.md', 'setup.py']
    exclude_filters = ['__pycache__', '.pyc']

    output_file = Path('content') / '_static' / 'code_archive.zip'

    print('Archiving code')
    
    with zipfile.ZipFile(output_file, 'w') as zf:
        for file_to_archive in files_to_archive:
            path = rootdir / file_to_archive
            assert path.exists()
            if path.is_dir():
                ## Recurse throufgh the whole directory
                for p in path.rglob('*'):
                    valid = True 
                    for exclude_filter in exclude_filters:
                        if exclude_filter in str(p):
                            valid = False
                            break
                    if valid:
                        zf.write(p, p.relative_to(rootdir))
            else:
                zf.write(path, file_to_archive)

if __name__ == '__main__':
    main()