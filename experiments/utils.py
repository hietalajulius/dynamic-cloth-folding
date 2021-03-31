import tracemalloc
import linecache
import argparse
from gym.envs.robotics import task_definitions
import os
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from rlkit.envs.wrappers import NormalizedBoxEnv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time




def plot_trajectory(ee_initial, current_ee_positions, desired_starts, desired_ends):
    fig = plt.figure()
    ax1 = Axes3D(fig)
    ax1.set_xlim3d(-0.02, 0.18)
    ax1.set_ylim3d(-0.1, 0.1)
    ax1.set_zlim3d(-0.02, 0.18)
    ax1.set_box_aspect((1, 1, 1))

    print("EE INIT", ee_initial)

    traj = [[-0.06420172509633885, -0.024398635106693198, 0.15264362939491052] ,
            [-0.05745638412035337, -0.018977838449063995, 0.1593416338822173] ,
            [-0.05248220694052869, -0.01498041220673684, 0.16428090399320786] ,
            [-0.0488141286766758, -0.012032613650161877, 0.16792324094230326] ,
            [-0.046109199265571825, -0.009858835873317455, 0.1706091880868471] ,
            [-0.04411451905627606, -0.00825583969416436, 0.17258987031479825] ,
            [-0.04264359374450154, -0.007073751636336483, 0.1740504731774014] ,
            [-0.04155889793073856, -0.006202051382379839, 0.17512755696809681] ,
            [-0.04075901706732782, -0.005559238583691774, 0.17592182454204305] ,
            [-0.04016916562000719, -0.0050852129168623425, 0.17650753661428129] ,
            [-0.029363050067017848, 0.004670425213100397, 0.18729152070813007] ,
            [-0.022615941435557667, 0.010882761511415847, 0.19403088170887617] ,
            [-0.017640460743937905, 0.015463888182178784, 0.1990006491106811] ,
            [-0.013971421239037638, 0.01884212151756123, 0.20266547549782793] ,
            [-0.011265782985593572, 0.021333312285352964, 0.20536800689659118] ,
            [-0.009270580058854061, 0.02317037646222049, 0.2073609187537303] ,
            [-0.007799269282624972, 0.02452507190889698, 0.20883054003995966] ,
            [-0.0067142892180731386, 0.025524057015870925, 0.20991427423382358] ,
            [-0.0059141987412726004, 0.026260732736162216, 0.21071344597551983] ,
            [-0.005324192719981016, 0.02680397518600272, 0.21130277449937931] ,
            [0.005464784289390115, 0.0340197461671779, 0.22193306656591186] ,
            [0.012198934422763385, 0.03821577526977184, 0.22855160169029376] ,
            [0.01716485920524553, 0.04131002836717788, 0.23343226917571452] ,
            [0.020826851952193297, 0.043591805255924446, 0.2370313911348154] ,
            [0.023527293756306814, 0.04527444258587742, 0.23968547050309372] ,
            [0.025518664695935788, 0.0465152599497988, 0.2416426526639562] ,
            [0.026987149672415615, 0.04743026861102459, 0.24308592601010454] ,
            [0.028070045924054574, 0.04810501807413032, 0.24415023059725874] ,
            [0.028868599746937226, 0.04860259460783388, 0.24493507447819116] ,
            [0.02945747260255507, 0.048969519548763635, 0.24551383729183654] ,
            [0.0402319939741556, 0.05, 0.25347337650649193] ,
            [0.04695783095632473, 0.05, 0.2581443156150992] ,
            [0.05191762542053841, 0.05, 0.2615887788090432] ,
            [0.05557509752288948, 0.05, 0.264128809072459] ,
            [0.05827220569460829, 0.05, 0.26600188867174523] ,
            [0.0602611183332792, 0.05, 0.2673831427598534] ,
            [0.061727790499321394, 0.05, 0.2684017128497968] ,
            [0.06280934994080616, 0.05, 0.26915283100905457] ,
            [0.06360691796762803, 0.05, 0.2697067236618402] ,
            [0.06419506387344648, 0.05, 0.27011517746771396] ,
            [0.07501371350684916, 0.042176441925642835, 0.26209096168362245] ,
            [0.08177359724618014, 0.03640716144344825, 0.25532780398459703] ,
            [0.08675849861056752, 0.03215275482105395, 0.25034048832334066] ,
            [0.0904344851468582, 0.02901545273207887, 0.24666272142628912] ,
            [0.09314524630885988, 0.026701930478310792, 0.2439506473833078] ,
            [0.09514422699196966, 0.02499588335027388, 0.2419506985500879] ,
            [0.0966183235768917, 0.023737803036439662, 0.24047588802791622] ,
            [0.09770535796398663, 0.022810064259446764, 0.23938832716626832] ,
            [0.09850696334779487, 0.022125927301836205, 0.23858633354742248] ,
            [0.09909808649806394, 0.02162142820319369, 0.23799492410326978] ,
            [0.10581064007613065, 0.023835795811453522, 0.22716587712554992] ,
            [0.10953642364694278, 0.026513540987171097, 0.2204050849179458] ,
            [0.11228390634364839, 0.02848817488542172, 0.21541951362713535] ,
            [0.11430996637242544, 0.029944317563776315, 0.21174303307093706] ,
            [0.11580403192018308, 0.031018112306996394, 0.20903190760673634] ,
            [0.11690579190226205, 0.03180995445797983, 0.20703265827839548] ,
            [0.11771825629509439, 0.03239387796480739, 0.2055583635879985] ,
            [0.11831738714061202, 0.03282447724700311, 0.20447118311314666] ,
            [0.11875920068126367, 0.03314201154671787, 0.20366947000070185] ,
            [0.11908500464572906, 0.03337616900044424, 0.203078267408737] ,
            [0.11762658428024886, 0.038226125767313386, 0.19226200033003701] ,
            [0.11587637026535269, 0.04131766107379038, 0.18551020393765724] ,
            [0.11458572022353863, 0.04359743379793403, 0.18053126637803424] ,
            [0.11363396374332699, 0.045278593208554534, 0.1768596776952649] ,
            [0.11293211555189032, 0.04651832071910255, 0.17415215961655645] ,
            [0.11241455576210733, 0.047432525696156025, 0.1721555704613126] ,
            [0.1120328946860107, 0.0481066825031627, 0.170683237446739] ,
            [0.11175144860020035, 0.04860382199795986, 0.16959750355896852] ,
            [0.11154390346610368, 0.04897042465585566, 0.16879685719473916] ,
            [0.11139085467680114, 0.049240766300771815, 0.1682064412486457] ,
            [0.10178867609346641, 0.04721194526921502, 0.15747663463324085] ,
            [0.09502476180881393, 0.04515596644363063, 0.1507869671297874] ,
            [0.09003688822306194, 0.0436398380532596, 0.1458538449118359] ,
            [0.08635870989902653, 0.042521808398729634, 0.14221604156739698] ,
            [0.08364633245986088, 0.04169734635689524, 0.1395334376123087] ,
            [0.08164615989497462, 0.0410893681555375, 0.1375552207329228] ,
            [0.08017118438767419, 0.040641030361230124, 0.13609643587783288] ,
            [0.07908350186201178, 0.040310415254709, 0.13502069272974063] ,
            [0.0782814185251863, 0.040066611698444395, 0.13422741377799552] ,
            [0.0776899429208308, 0.039886825073832094, 0.13364243073962073] ,
            [0.06694826609854576, 0.0302911909369939, 0.12884875835110537] ,
            [0.060252039769237015, 0.02358747671245335, 0.12652528790116804] ,
            [0.055314080914855686, 0.018643996103468428, 0.12481190473325685] ,
            [0.05167271091802045, 0.014998554231565708, 0.12354841464662353] ,
            [0.048987476827402836, 0.01231031744203162, 0.12261668657264754] ,
            [0.04700732042262066, 0.010327946774622624, 0.12192960781119233] ,
            [0.04554710531470875, 0.00886609881602643, 0.12142293935541817] ,
            [0.04447030746367803, 0.007788096861505497, 0.12104930981179371] ,
            [0.04367625074853936, 0.006993152211401877, 0.12077378637180428] ,
            [0.043090694168673185, 0.006406940846869422, 0.12057060873963099] ,]

    traj = np.array(traj) - ee_initial

    ax1.plot(traj[:, 0], traj[:,1], traj[:, 2], linewidth=3, label="interpolated", color="green")

    ax1.plot(current_ee_positions[:, 0], current_ee_positions[:,
                                                              1], current_ee_positions[:, 2], linewidth=3, label="achieved", color="blue")
    for i, (ds, de) in enumerate(zip(desired_starts, desired_ends)):
        ax1.plot([ds[0], de[0]], [ds[1], de[1]], [ds[2], de[2]],
                 linewidth=2, label="desired " + str(i), color="orange")

    ax1.text(0, 0,
             0, "start", size=10, zorder=1, color='k')

    plt.legend()

    plt.show()


def render_env(env):
    cameras_to_render = ["birdview", "frontview"]

    for camera in cameras_to_render:
        camera_id = env.sim.model.camera_name2id(camera)
        env.sim._render_context_offscreen.render(
            1000, 1000, camera_id)
        image_obs = env.sim._render_context_offscreen.read_pixels(
            1000, 1000, depth=False)
        image_obs = image_obs[::-1, :, :]
        image = image_obs.reshape((1000, 1000, 3)).copy()
        cv2.imshow(camera, image)

    cv2.waitKey(10)
    time.sleep(0.1)


def get_obs_processor(observation_key, additional_keys, desired_goal_key):
    def obs_processor(o):
        obs = o[observation_key]
        for additional_key in additional_keys:
            obs = np.hstack((obs, o[additional_key]))

        return np.hstack((obs, o[desired_goal_key]))
    return obs_processor


def get_tracking_score(ee_positions, goals):
    norm = np.linalg.norm(ee_positions-goals, axis=1)
    return np.sum(norm)


def ATE(ee_positions, goals):
    squared_norms = np.linalg.norm(ee_positions-goals, axis=1)**2
    return np.sqrt(squared_norms.mean())


def get_robosuite_env(variant, evaluation=False):
    options = {}
    options["env_name"] = variant["env_name"]
    options["robots"] = "PandaReal"
    controller_name = variant['ctrl_kwargs']["ctrl_name"]
    options["controller_configs"] = load_controller_config(
        default_controller=controller_name)

    outp_override = variant["ctrl_kwargs"]["output_max"]
    options["controller_configs"]['output_max'][:3] = [
        outp_override for _ in range(3)]
    options["controller_configs"]['output_min'][:3] = [
        -outp_override for _ in range(3)]

    options["controller_configs"]['input_min'] = - \
        variant["ctrl_kwargs"]["input_max"]
    options["controller_configs"]['input_max'] = variant["ctrl_kwargs"]["input_max"]

    options["controller_configs"]["interpolation"] = variant["ctrl_kwargs"]["interpolator"]
    options["controller_configs"]["ramp_ratio"] = variant["ctrl_kwargs"]["ramp_ratio"]
    options["controller_configs"]["damping_ratio"] = variant["ctrl_kwargs"]["damping_ratio"]
    options["controller_configs"]["kp"] = variant["ctrl_kwargs"]["kp"]

    if variant["ctrl_kwargs"]["position_limits"] == "None":
        pos_limits = None
    else:
        pos_limits = variant["ctrl_kwargs"]["position_limits"]
    options["controller_configs"]["position_limits"] = pos_limits

    if variant['image_training'] or evaluation:
        has_offscreen_renderer = True
    else:
        has_offscreen_renderer = False

    env = suite.make(
        **options,
        **variant['env_kwargs'],
        has_renderer=False,
        has_offscreen_renderer=has_offscreen_renderer,
        ignore_done=False,
        use_camera_obs=False,
    )
    return NormalizedBoxEnv(env)


def argsparser():
    parser = argparse.ArgumentParser("Parser")
    # Generic
    parser.add_argument('--run',  default=1, type=int)
    parser.add_argument('--title', default="notitle", type=str)
    parser.add_argument('--num_processes', type=int, default=1)
    # TODO: no traditional logging at all
    parser.add_argument('--log_tabular_only', type=int, default=0)

    # Train
    parser.add_argument('--train_steps', default=1000, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--save_policy_every_epoch', default=1, type=int)
    parser.add_argument('--num_cycles', default=20, type=int)
    parser.add_argument('--min_expl_steps', type=int, default=0)
    parser.add_argument('--num_eval_rollouts', type=int, default=20)
    parser.add_argument('--num_eval_param_buckets', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--debug_same_batch', type=int, default=0)

    # Replay buffer
    # HER 0.8 from paper
    parser.add_argument('--her_percent', default=0.8, type=float)
    parser.add_argument('--buffer_size', default=1E5, type=int)

    # Collection
    parser.add_argument('--max_path_length', default=50, type=int)

    # Controller optimization
    parser.add_argument('--ctrl_eval_file', type=int, default=0)
    parser.add_argument('--ctrl_eval', type=int, default=0)

    # Controller
    parser.add_argument('--output_max', type=float, default=0.035)
    parser.add_argument('--input_max', type=float, default=1.)
    parser.add_argument('--position_limits',
                        default=[[-0.12, -0.25, 0.12], [0.12, 0.05, 0.4]])
    parser.add_argument('--interpolator', type=str,
                        default="filter")  # TODO: fix this shit
    parser.add_argument('--ctrl_name', type=str, default="OSC_POS_VEL")
    parser.add_argument('--ramp_ratio', type=float,
                        default=0.03)  # TODO: fix this shit
    parser.add_argument('--damping_ratio', type=float, default=1)
    parser.add_argument('--kp', type=float, default=1000.0)

    # Env
    parser.add_argument('--env_name', type=str, default="Cloth")
    parser.add_argument('--domain_randomization', type=int, default=0)
    parser.add_argument('--constant_goal', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--task', type=str, default="sideways_franka_1")
    parser.add_argument('--velocity_in_obs', type=int, default=1)
    parser.add_argument('--image_training', default=0, type=int)
    parser.add_argument('--image_size', type=int, default=100)
    parser.add_argument('--randomize_params', type=int, default=0)
    parser.add_argument('--randomize_geoms', type=int, default=0)
    parser.add_argument('--uniform_jnt_tend', type=int, default=1)
    parser.add_argument('--sparse_dense', type=int, default=1)
    parser.add_argument('--goal_noise_range', type=tuple, default=(0, 0.01))
    parser.add_argument('--max_advance', type=float, default=0.05)

    args = parser.parse_args()

    file = open("params.txt", "w")
    file.write(str(args.__dict__))
    return args


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
        path_collector_kwargs=dict(),
        policy_kwargs=dict(),
        replay_buffer_kwargs=dict(),
        algorithm_kwargs=dict()
    )

    variant['env_name'] = args.env_name
    variant['domain_randomization'] = bool(args.domain_randomization)
    variant['random_seed'] = args.seed
    variant['version'] = args.title
    variant['image_training'] = bool(args.image_training)
    variant['num_processes'] = int(args.num_processes)
    variant['log_tabular_only'] = bool(args.log_tabular_only)

    variant['algorithm_kwargs'] = dict(
        num_epochs=args.num_epochs,
        num_trains_per_train_loop=args.train_steps,
        num_expl_steps_per_train_loop=args.train_steps,
        num_train_loops_per_epoch=int(args.num_cycles),
        max_path_length=int(args.max_path_length),
        num_eval_rollouts_per_epoch=args.num_eval_rollouts,
        num_eval_param_buckets=args.num_eval_param_buckets,
        save_policy_every_epoch=args.save_policy_every_epoch,
        min_num_steps_before_training=args.min_expl_steps,
        batch_size=args.batch_size,
        debug_same_batch=bool(args.debug_same_batch)
    )

    variant['replay_buffer_kwargs'] = dict(
        max_size=int(args.buffer_size),
        fraction_goals_rollout_goals=1 - args.her_percent
    )

    variant['env_kwargs'] = dict(
        constant_goal=bool(args.constant_goal),
        sparse_dense=bool(args.sparse_dense),
        constraints=task_definitions.constraints[args.task],
        pixels=bool(args.image_training),
        goal_noise_range=tuple(args.goal_noise_range),
        randomize_params=bool(args.randomize_params),
        randomize_geoms=bool(args.randomize_geoms),
        uniform_jnt_tend=bool(args.uniform_jnt_tend),
        image_size=args.image_size,
        random_seed=args.seed,
        velocity_in_obs=bool(args.velocity_in_obs)
    )

    variant['ctrl_kwargs'] = dict(
        ctrl_name=str(args.ctrl_name),
        output_max=args.output_max,
        input_max=args.input_max,
        position_limits=args.position_limits,
        ctrl_eval=bool(
            args.ctrl_eval),
        ctrl_eval_file=args.ctrl_eval_file,
        interpolator=args.interpolator,
        ramp_ratio=args.ramp_ratio,
        damping_ratio=args.damping_ratio,
        kp=args.kp)

    if args.image_training:
        channels = 1
        variant['policy_kwargs'] = dict(
            input_width=args.image_size,
            input_height=args.image_size,
            input_channels=channels,
            kernel_sizes=[3, 3, 3, 3],
            n_channels=[32, 32, 32, 32],
            strides=[2, 2, 2, 2],
            paddings=[0, 0, 0, 0],
            hidden_sizes=[256, 256, 256, 256],
            init_w=1e-4
        )
        variant['path_collector_kwargs']['additional_keys'] = [
            'robot_observation']
        variant['replay_buffer_kwargs']['internal_keys'] = [
            'image', 'model_params', 'robot_observation']

    else:
        variant['path_collector_kwargs']['additional_keys'] = [
            'robot_observation']
        variant['replay_buffer_kwargs']['internal_keys'] = [
            'model_params', 'robot_observation']

    return variant
