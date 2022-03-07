import pickle
import matplotlib.pyplot as plt

def plot_following(files):
    print(files)
    f,ax = plt.subplots(2,1,sharex=True,squeeze=True,figsize=(5,5))
    colors=colors = ['#e41a1c', '#265285', '#4daf4a', '#984ea3', '#6e0178', '#ff7f00']
    for i, file in enumerate(files):
        (feedback_list, setpoint_list, time_list, outputs, alg_name) = pickle.load(open(file, "rb"))
        if i == 0:
            ax[0].plot(time_list, setpoint_list, color='black', label='Set-point')
        ax[0].plot(time_list, feedback_list, color=colors[i-1], alpha=0.8, label=alg_name)
        ax[0].set_ylabel('State')
        ax[0].grid(True)
        ax[0].legend()

        ax[1].plot(time_list, outputs, color=colors[i-1], alpha=0.8, label=alg_name)
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Action')
        ax[1].grid(True)
        # ax[1].legend()
    plt.show()

if __name__ == "__main__":
    import sys
    plot_following(sys.argv[1:])
