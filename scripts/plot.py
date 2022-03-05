import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt


def plot(args):

    # env_d4rl_name = 'halfcheetah-medium-v2'
    # log_dir = 'dt_runs/'
    # x_key = "num_updates"
    # y_key = "eval_d4rl_score"
    # y_smoothing_win = 5
    # plot_avg = False
    # save_fig = False

    env_d4rl_name = args.env_d4rl_name
    log_dir = args.log_dir
    x_key = args.x_key
    y_key = args.y_key
    y_smoothing_win = args.smoothing_window
    plot_avg = args.plot_avg
    save_fig = args.save_fig

    if plot_avg:
        save_fig_path = env_d4rl_name + "_avg.png"
    else:
        save_fig_path = env_d4rl_name + ".png"

    all_files = glob.glob(log_dir + f'/dt_{env_d4rl_name}*.csv')

    ax = plt.gca()
    ax.set_title(env_d4rl_name)

    if plot_avg:
        name_list = []
        df_list = []
        for filename in all_files:
            frame = pd.read_csv(filename, index_col=None, header=0)
            print(filename, frame.shape)
            frame['y_smooth'] = frame[y_key].rolling(window=y_smoothing_win).mean()
            df_list.append(frame)

        df_concat = pd.concat(df_list)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()

        data_avg.plot(x=x_key, y='y_smooth', ax=ax)

        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.legend(['avg of all runs'], loc='lower right')

        if save_fig:
            plt.savefig(save_fig_path)

        plt.show()

    else:
        name_list = []
        for filename in all_files:
            frame = pd.read_csv(filename, index_col=None, header=0)
            print(filename, frame.shape)
            frame['y_smooth'] = frame[y_key].rolling(window=y_smoothing_win).mean()
            frame.plot(x=x_key, y='y_smooth', ax=ax)
            name_list.append(filename.split('/')[-1])

        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.legend(name_list, loc='lower right')

        if save_fig:
            plt.savefig(save_fig_path)

        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_d4rl_name', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')
    parser.add_argument('--x_key', type=str, default='num_updates')
    parser.add_argument('--y_key', type=str, default='eval_d4rl_score')
    parser.add_argument('--smoothing_window', type=int, default=1)
    parser.add_argument("--plot_avg", action="store_true", default=False,
                    help="plot avg of all logs else plot separately")
    parser.add_argument("--save_fig", action="store_true", default=False,
                    help="save figure if true")

    args = parser.parse_args()

    plot(args)
