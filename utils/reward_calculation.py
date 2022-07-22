import numpy as np


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def get_task_reward_function(constraints, single_goal_dim, sparse_dense, success_reward, fail_reward, extra_reward):
    constraint_distances = [c['distance'] for c in constraints]

    def task_reward_function(achieved_goal, desired_goal, info):
        achieved_oks = np.zeros(
            (achieved_goal.shape[0], len(constraint_distances)))
        achieved_distances = np.zeros(
            (achieved_goal.shape[0], len(constraint_distances)))

        for i, constraint_distance in enumerate(constraint_distances):
            achieved = achieved_goal[:, i *
                                     single_goal_dim:(i+1)*single_goal_dim]
            desired = desired_goal[:, i *
                                   single_goal_dim:(i+1)*single_goal_dim]

            achieved_distances_per_constraint = goal_distance(
                achieved, desired)

            constraint_ok = achieved_distances_per_constraint < constraint_distance

            achieved_distances[:, i] = achieved_distances_per_constraint
            achieved_oks[:, i] = constraint_ok

        successes = np.all(achieved_oks, axis=1)
        fails = np.invert(successes)

        task_rewards = successes.astype(np.float32).flatten()*success_reward

        if sparse_dense:
            dist_rewards = np.sum((1 - achieved_distances/np.array(constraint_distances)),
                                  axis=1) / len(constraint_distances)

            task_rewards += dist_rewards*extra_reward  # Extra for being closer to the goal

            if "num_future_goals" in info.keys():
                num_future_goals = info['num_future_goals']
                task_rewards[-num_future_goals:] = success_reward

        task_rewards[fails] = fail_reward

        return task_rewards

    return task_reward_function
