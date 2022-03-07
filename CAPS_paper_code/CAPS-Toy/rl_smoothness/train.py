
def training_dir_name(alg_name, commit_id, rand_seed):
    import datetime
    now = datetime.datetime.now()
    timestamp = now.strftime('%y%m%d-%H%M%S')
    return '{alg_name}_t{timestamp}_{commit_id}_rs{rand_seed}'.format(**locals())

def parse_args(training_algs):
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="algs", dest='alg_name', required=True)
    for alg in training_algs:
        subparser = subparsers.add_parser(alg.name())
        alg.add_args(subparser)

    return vars(parser.parse_args())

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    sys.exit(0)

if __name__ == '__main__':
    from signal import signal, SIGINT
    import os
    from os.path import join
    import pprint as pp
    import sys
    from rl_smoothness.train_alg import subclasses

    signal(SIGINT, handler)

    training_algs = subclasses()
    args = parse_args(training_algs)
    chosen_alg_name = args['alg_name']
    chosen_alg = [n for n in training_algs if n.name() == chosen_alg_name][0]

    commit_id = os.popen('git rev-parse local').read().rstrip('\n')
    training_dir = join("Training", training_dir_name(chosen_alg_name, commit_id, args['seed']))
    args['training_dir'] = training_dir
    args['summary_dir'] = join(training_dir, 'summary')
    args['ckpt_dir'] = join(training_dir, 'checkpoints')

    pp.pprint(args)

    chosen_alg.train(args)




