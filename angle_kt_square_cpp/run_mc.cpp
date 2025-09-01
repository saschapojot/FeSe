#include "./mc_subroutine/mc_read_load_compute.hpp"



int main(int argc, char *argv[])
{

    if (argc != 2) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }

    auto mcObj=mc_computation(std::string(argv[1]));
    mcObj.init_and_run();


}
