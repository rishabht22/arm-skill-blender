"""Small actor and critic networks for the first PPO baseline."""

from stable_baselines3.common.policies import ActorCriticPolicy


class BasicPolicy(ActorCriticPolicy):
    """Separate MLPs predict both arms' actions and the state's value."""

    def __init__(self, *args, hidden_sizes=(128, 128), **kwargs):
        kwargs.setdefault(
            "net_arch", {"pi": list(hidden_sizes), "vf": list(hidden_sizes)}
        )
        super().__init__(*args, **kwargs)
