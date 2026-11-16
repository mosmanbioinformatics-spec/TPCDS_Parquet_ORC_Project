from common import ROOT, experiment_args, get_data, TABLES


def main():
    args = experiment_args()
    data, commit = get_data(args.scale, ROOT / 'work/data')
    print(f'SF{args.scale}: {commit}')
    for name in TABLES:
        path = data / f'{name}.dat'
        print(f'{name}: {path.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
