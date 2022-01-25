
#include <math.h>
#include <string>
#include <torch/script.h>
#include <monopod_sdk/monopod.hpp>

#include <signal.h>
#include <atomic>

/**
 * @brief This boolean is here to kill cleanly the application upon ctrl+c
 */
std::atomic_bool StopDemos(false);

/**
 * @brief This function is the callback upon a ctrl+c call from the terminal.
 *
 * @param s
 */
void my_handler(int)
{
    StopDemos = true;
}

std::vector<double> merge (std::vector<double> a, std::vector<double> b)
{
    std::vector<double> result;

    auto v1 = a.begin();
    auto v2 = b.begin();

    while (v1 != a.end() && v2 != b.end ())
    {
        result.push_back(*v1);
        result.push_back(*v2);
        ++v1;
        ++v2;
    }
    return result;
}

int main(int, char**)
{
    // make sure we catch the ctrl+c signal to kill the application properly.
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
    StopDemos = false;

    // Load policy network
    std::string path_to_network = "/home/capstone/capstone/rl-algorithm-exploration/exported_policy_net.pt"; //ENTER PATH HERE
    torch::jit::script::Module policy_net;
    try
    {
        policy_net = torch::jit::load(path_to_network);
    }
    catch (const c10::Error& e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // Monopod startup/
    monopod_drivers::Monopod monopod;
    monopod.initialize();
    monopod.start_loop();
    // rt_printf("loops have started \n");
    double x = 0;


    while (!StopDemos)
    {
        // real_time_tools::Timer::sleep_sec(1);

        // // Get data from monopod. The model inputs are in the following
        // // order: hip_joint, knee_joint, planarizer_pitch_joint,
        // // planarizer_yaw_joint, boom_connector_joint
        std::vector<double> vels = monopod.get_velocities().value();
        std::vector<double> poss = monopod.get_positions().value();
        std::vector<double> input_data = merge(poss, vels);

        // Create inputs for model
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor input_data = torch::from_blob(input_data.data(), {input_data.size()}).to(torch::kInt64);
       
        // inputs.push_back(input_data);
        // inputs.push_back(torch::ones({1, 8}));

        // Print output of model
        auto output = policy_net.forward(inputs);
        // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';   
        std::cout << output;
        /**
         * TODO: Set and send ouput torques to the robot
         */
    }

    return 0;
}
