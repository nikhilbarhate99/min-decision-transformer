import argparse
import os
import gym
import torch
import numpy as np
from decision_transformer.utils import evaluate_on_env, get_d4rl_normalized_score, get_d4rl_dataset_stats
from decision_transformer.model import DecisionTransformer

def test(args):

    eval_dataset = args.dataset         # medium / medium-replay / medium-expert
    eval_rtg_scale = args.rtg_scale     # normalize returns to go

    if args.env == 'walker2d':
        eval_env_name = 'Walker2d-v3'
        eval_rtg_target = 5000
        eval_env_d4rl_name = f'walker2d-{eval_dataset}-v2'

    elif args.env == 'halfcheetah':
        eval_env_name = 'HalfCheetah-v3'
        eval_rtg_target = 6000
        eval_env_d4rl_name = f'halfcheetah-{eval_dataset}-v2'

    elif args.env == 'hopper':
        eval_env_name = 'Hopper-v3'
        eval_rtg_target = 3600
        eval_env_d4rl_name = f'hopper-{eval_dataset}-v2'

    else:
        raise NotImplementedError

    render = args.render                # render the env frames

    num_test_eval_ep = args.num_eval_ep         # num of evaluation episodes
    eval_max_eval_ep_len = args.max_eval_ep_len # max len of one episode

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability


    eval_chk_pt_dir = args.chk_pt_dir

    eval_chk_pt_name = args.chk_pt_name
    eval_chk_pt_list = [eval_chk_pt_name]


    ## manually override check point list
    ## passing a list will evaluate on all checkpoints
    ## and output mean and std score

    # eval_chk_pt_list = [
    #     "dt_halfcheetah-medium-v2_model_22-02-09-10-38-54_best.pt",
    #     "dt_halfcheetah-medium-v2_model_22-02-10-11-56-32_best.pt",
    #     "dt_halfcheetah-medium-v2_model_22-02-11-10-13-57_best.pt"
    # ]


    device = torch.device(args.device)
    print("device set to: ", device)

    env_data_stats = get_d4rl_dataset_stats(eval_env_d4rl_name)
    eval_state_mean = np.array(env_data_stats['state_mean'])
    eval_state_std = np.array(env_data_stats['state_std'])

    eval_env = gym.make(eval_env_name)

    state_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]

    all_scores = []

    for eval_chk_pt_name in eval_chk_pt_list:

        eval_model = DecisionTransformer(
        			state_dim=state_dim,
        			act_dim=act_dim,
        			n_blocks=n_blocks,
        			h_dim=embed_dim,
        			context_len=context_len,
        			n_heads=n_heads,
        			drop_p=dropout_p,
        		).to(device)

        eval_chk_pt_path = os.path.join(eval_chk_pt_dir, eval_chk_pt_name)

        # load checkpoint
        eval_model.load_state_dict(torch.load(eval_chk_pt_path, map_location=device))

        print("model loaded from: " + eval_chk_pt_path)

        # evaluate on env
        results = evaluate_on_env(eval_model, device, context_len,
                                eval_env, eval_rtg_target, eval_rtg_scale,
                                num_test_eval_ep, eval_max_eval_ep_len,
                                eval_state_mean, eval_state_std, render=render)
        print(results)

        norm_score = get_d4rl_normalized_score(results['eval/avg_reward'], eval_env_name) * 100
        print("normalized d4rl score: " + format(norm_score, ".5f"))

        all_scores.append(norm_score)

    print("=" * 60)
    all_scores = np.array(all_scores)
    print("evaluated on env: " + eval_env_name)
    print("total num of checkpoints evaluated: " + str(len(eval_chk_pt_list)))
    print("d4rl score mean: " + format(all_scores.mean(), ".5f"))
    print("d4rl score std: " + format(all_scores.std(), ".5f"))
    print("d4rl score var: " + format(all_scores.var(), ".5f"))
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1000)

    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=10)

    parser.add_argument("--render", action="store_true", default=False)

    parser.add_argument('--chk_pt_dir', type=str, default='dt_runs/')
    parser.add_argument('--chk_pt_name', type=str,
            default='dt_halfcheetah-medium-v2_model_22-02-13-09-03-10_best.pt')

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    test(args)
