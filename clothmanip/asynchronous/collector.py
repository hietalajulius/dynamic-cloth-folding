import psutil
import os
from rlkit.launchers.launcher_util import setup_logger
from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from clothmanip.utils.utils import get_variant, argsparser, get_randomized_env, dump_commit_hashes, get_keys_and_dims
from rlkit.envs.wrappers import SubprocVecEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhScriptPolicy, CustomScriptPolicy, CustomTanhScriptPolicy, ScriptPolicy
from rlkit.samplers.data_collector import KeyPathCollector, VectorizedKeyPathCollector
import copy

def collector(variant, path_queue, policy_weights_queue, paths_available_event, new_policy_event, keys, dims, num_collected_steps, collector_memory_usage, env_memory_usages):
    process = psutil.Process(os.getpid())

    def make_env():
        env = ClothEnv(**variant['env_kwargs'], save_folder=variant['save_folder'], has_viewer=variant['image_training'])
        env = NormalizedBoxEnv(env)
        env = get_randomized_env(env, variant)
        return env

    env_fns = [make_env for _ in range(variant['num_processes'])]
    vec_env = SubprocVecEnv(env_fns)

    print("Collector waiting for policy")
    new_policy_event.wait()

    policy = TanhScriptPolicy(
        output_size=dims['action_dim'],
        added_fc_input_size=dims['added_fc_input_size'],
        aux_output_size=8,
        **variant['policy_kwargs'],
    )

    expl_path_collector = VectorizedKeyPathCollector(
        vec_env,
        policy,
        output_max=variant["env_kwargs"]["output_max"],
        demo_coef=variant['demo_coef'],
        processes=variant['num_processes'],
        observation_key=keys['path_collector_observation_key'],
        desired_goal_key=keys['desired_goal_key'],
        use_demos=variant['use_demos'],
        demo_path=variant['demo_path'],
        num_demoers=variant['num_demoers'],
        **variant['path_collector_kwargs'],
    )

    steps_per_rollout = variant['algorithm_kwargs']['max_path_length'] * \
        variant['num_processes']

    while True:
        if new_policy_event.wait():
            state_dict = policy_weights_queue.get()
            local_state_dict = copy.deepcopy(state_dict)
            del state_dict

            policy.load_state_dict(local_state_dict)

            collector_memory_usage.value = process.memory_info().rss/10E9

        # Keep collecting paths even without new policy
        print("Collecting paths")
        paths = expl_path_collector.collect_new_paths(
            variant['algorithm_kwargs']['max_path_length'],
            steps_per_rollout,
            discard_incomplete_paths=False,
        )


        path_queue.put(paths)
        paths_available_event.set()
        print("Gave new paths")

        new_policy_event.clear()

        num_collected_steps.value = expl_path_collector._num_steps_total

