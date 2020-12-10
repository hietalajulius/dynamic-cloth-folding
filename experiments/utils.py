import tracemalloc
import linecache
import argparse

def argsparser():
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('--run',  default=1, type=int)
    parser.add_argument('--title', default="notitle", type=str)
    parser.add_argument('--train_steps', default=1000, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_cycles', default=100, type=int)
    parser.add_argument('--her_percent', default=0.0, type=float)
    parser.add_argument('--buffer_size', default=1E5, type=int)
    parser.add_argument('--max_path_length', default=50, type=int)
    parser.add_argument('--env_name', type=str, default="Cloth-v1")
    parser.add_argument('--strict', default=1, type=int)
    parser.add_argument('--image_training', default=1, type=int)
    parser.add_argument('--task', type=str, default="sideways")
    parser.add_argument('--distance_threshold', type=float, default=0.05)
    parser.add_argument('--eval_steps', type=int, default=0)
    parser.add_argument('--min_expl_steps', type=int, default=0)
    parser.add_argument('--randomize_params', type=int, default=0)
    parser.add_argument('--randomize_geoms', type=int, default=0)
    parser.add_argument('--uniform_jnt_tend', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=84)
    parser.add_argument('--rgb', type=int, default=0)
    parser.add_argument('--max_advance', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cprofile', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_processes', type=int, default=3)
    return parser.parse_args()

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def get_variant(args):
    variant = dict(
        algorithm="SAC",
        layer_size=256,
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        path_collector_kwargs = dict(),
        policy_kwargs = dict(),
        replay_buffer_kwargs=dict(),
        algorithm_kwargs=dict(),
    )
    
    variant['env_name'] = args.env_name
    variant['version'] = args.title
    variant['image_training'] = bool(args.image_training)
    variant['num_processes'] = int(args.num_processes)

    variant['algorithm_kwargs'] = dict(
        num_epochs=args.num_epochs,
        num_trains_per_train_loop = args.train_steps,
        num_expl_steps_per_train_loop = args.train_steps,
        num_train_loops_per_epoch = int(args.num_cycles),
        max_path_length = int(args.max_path_length),
        num_eval_steps_per_epoch=args.eval_steps,
        min_num_steps_before_training=args.min_expl_steps,
        batch_size=args.batch_size,
    )

    variant['replay_buffer_kwargs'] =dict(
            max_size=int(args.buffer_size),
            fraction_goals_env_goals = 0,
            fraction_goals_rollout_goals = 1 - args.her_percent
        )

    variant['env_kwargs'] = dict(
        task=args.task,
        pixels=bool(args.image_training),
        strict=bool(args.strict),
        distance_threshold=args.distance_threshold,
        randomize_params=bool(args.randomize_params),
        randomize_geoms=bool(args.randomize_geoms),
        uniform_jnt_tend=bool(args.uniform_jnt_tend),
        image_size=args.image_size,
        rgb=bool(args.rgb),
        max_advance=args.max_advance,
        random_seed=args.seed
    )

    if args.image_training:
        if args.rgb:
            channels = 3
        else:
            channels = 1
        variant['policy_kwargs'] = dict(
            input_width=args.image_size,
            input_height=args.image_size,
            input_channels=channels,
            kernel_sizes=[3,3,3,3],
            n_channels=[32,32,32,32],
            strides=[2,2,2,2],
            paddings=[0,0,0,0],
            hidden_sizes=[256,256,256,256],
            init_w=1e-4
        )
        variant['path_collector_kwargs']['additional_keys'] = ['robot_observation']
        variant['replay_buffer_kwargs']['internal_keys'] = ['image','model_params','robot_observation']

    else:
        variant['path_collector_kwargs']['additional_keys'] = ['model_params']
        variant['replay_buffer_kwargs']['internal_keys'] = ['model_params']

    return variant