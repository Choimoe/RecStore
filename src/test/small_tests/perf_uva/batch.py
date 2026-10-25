
import argparse
import subprocess
import itertools
import datetime
import os
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[4]

CONFIG = {
    "method": ["CPU", "UVA"],
    # "emb_dim": [32, 64, 128],
    "emb_dim": [32, ],
    "key_space_M": [100],
    # "query_count": [100, 500, 1000, 2000, 5000, 10000,],
	"query_count": [128, 512, 1024, 1536, 2048,],
    "run_time": [30],
    "binding": [
        {"dummy": [0], },
    ],
}


parser = argparse.ArgumentParser(description='DCN inference')
parser.add_argument('--retest', action='store_true', default=False)
parser.add_argument('--gendone', action='store_true', default=True)


FLAGS = parser.parse_args()


def ExecuteCommand(execute_str):
    p = subprocess.Popen(execute_str, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    l = []
    for line in p.stdout.readlines():
        l.append(line.decode("utf-8"))
    return l


if FLAGS.gendone:
    lines = ExecuteCommand(
        "cd log && grep -RI success | sed 's|/| |g' | awk '{print $1}'")
    DONE = [each.strip() for each in lines]
    with open("DoneFile", "w") as f:
        f.write(str(DONE))
    # sys.exit(0)

try:
    with open("DoneFile", "r") as f:
        DONE = eval(f.read())
except:
    DONE = []

if FLAGS.retest:
    DONE = []


def gen_product(config):
    keys, values = zip(*config.items())
    permutations_config = [dict(zip(keys, v))
                           for v in itertools.product(*values)]
    return permutations_config


def gen_binding(config_list):
    r = []
    for each in config_list:
        r += gen_product(each)
    return r


def disjoint_dicts_to_one_dict(dicts):
    a = dicts[0].copy()
    for i in range(1, len(dicts)):
        a.update(dicts[i])
    return a


def get_next_config():
    CONFIG_BINDING = CONFIG['binding']
    del CONFIG['binding']
    for each_k in CONFIG.keys():
        CONFIG[each_k] = [str(each) for each in CONFIG[each_k]]

    permutations_config = gen_product(CONFIG)
    permutations_binding_config = gen_binding(CONFIG_BINDING)

    permutations_config = itertools.product(
        permutations_config, permutations_binding_config)

    # [(dictA, dictB), (dictA, dictB), (dictA, dictB),]
    permutations_config = [disjoint_dicts_to_one_dict(each)
                           for each in permutations_config]
    for each in permutations_config:
        yield each


config_generator = get_next_config()

config_list = list(config_generator)

all_configs = list(zip(range(len(config_list)), config_list))

for each_i, config in all_configs:
    print(datetime.datetime.now(), f'{each_i} / {len(all_configs)}')

    if f'run_{each_i}' in DONE:
        print(f"Pass run_{each_i}")
        continue

    run_path = f'log/run_{each_i}/'
    log_file = f'{run_path}/log'

    if not os.path.isdir(run_path):
        os.makedirs(run_path)

    bin_path = PROJECT_PATH / 'build/bin/perf_uva'

    execute_str = f'''{bin_path} --logtostderr \
        --method={config['method']} \
        --emb_dim={config['emb_dim']} --key_space_M={config['key_space_M']} \
        --query_count={config['query_count']} --run_time={config['run_time']} \
        >"{log_file}" 2>&1'''

    print(execute_str, flush=True)

    dump_config = config.copy()
    # dump_config['execute_str'] = execute_str
    with open(f'{run_path}/config', 'w') as f:
        import json
        s = json.dumps(dump_config, indent=2)
        f.write(s)
        f.write('\n')
        f.write(execute_str)

    p = subprocess.Popen(execute_str, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print(line.decode("utf-8"))
    retval = p.wait()
