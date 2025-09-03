//
// Created by adada on 9/1/2025.
//

#ifndef MC_READ_LOAD_COMPUTE_HPP
#define MC_READ_LOAD_COMPUTE_HPP

#include <boost/filesystem.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <cfenv> // for floating-point exceptions
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = boost::filesystem;
namespace py = boost::python;
namespace np = boost::python::numpy;

constexpr double PI = M_PI;

class mc_computation
{
public:
    mc_computation(const std::string& cppInParamsFileName)
    {
        std::ifstream file(cppInParamsFileName);
        if (!file.is_open())
        {
            std::cerr << "Failed to open the file." << std::endl;
            std::exit(20);
        }
        std::string line;
        int paramCounter = 0;
        while (std::getline(file, line))
        {
            // Check if the line is empty
            if (line.empty())
            {
                continue; // Skip empty lines
            }
            std::istringstream iss(line);
            //read T
            if (paramCounter == 0)
            {
                iss >> T;
                if (T <= 0)
                {
                    std::cerr << "T must be >0" << std::endl;
                    std::exit(1);
                } //end if
                std::cout << "T=" << T << std::endl;
                this->beta = 1.0 / T;
                std::cout << "beta=" << beta << std::endl;
                paramCounter++;
                continue;
            } //end T

            //read J11
            if (paramCounter == 1)
            {
                iss >> J11;
                std::cout << "J11=" << J11 << std::endl;
                paramCounter++;
                continue;
            } //end J11
            //read J12
            if (paramCounter == 2)
            {
                iss >> J12;
                std::cout << "J12=" << J12 << std::endl;
                paramCounter++;
                continue;
            } //end J12

            //read J21
            if (paramCounter == 3)
            {
                iss >> J21;
                std::cout << "J21=" << J21 << std::endl;
                paramCounter++;
                continue;
            } //end J21

            //read J22
            if (paramCounter == 4)
            {
                iss >> J22;
                std::cout << "J22=" << J22 << std::endl;
                paramCounter++;
                continue;
            } //end J22
            //read K
            if (paramCounter == 5)
            {
                iss >> K;
                std::cout << "K=" << K << std::endl;
                paramCounter++;
                continue;
            } //end K

            //read N
            if (paramCounter == 6)
            {
                iss >> N0;
                N1 = N0;
                if (N0 <= 0)
                {
                    std::cerr << "N must be >0" << std::endl;
                    std::exit(1);
                }
                if (N0 %2!= 0)
                {
                    std::cerr << "N must be even" << std::endl;
                    std::exit(1);
                }
                std::cout << "N0=N1=" << N0 << std::endl;
                this->lattice_num=N0*N1;
                this->total_components_num=3*N0*N1;
                this->tot_angle_components_num=2*N0*N1;
                std::cout<<"total_components_num="<<total_components_num<<std::endl;
                std::cout<<"tot_angle_components_num="<<tot_angle_components_num<<std::endl;
                std::cout<<"lattice_num="<<lattice_num<<std::endl;
                paramCounter++;
                continue;
            } //end N

            //read sweepToWrite
            if (paramCounter == 7)
            {
                iss >> sweepToWrite;
                if (sweepToWrite <= 0)
                {
                    std::cerr << "sweepToWrite must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "sweepToWrite=" << sweepToWrite << std::endl;
                paramCounter++;
                continue;
            } //end sweepToWrite

            //read newFlushNum
            if (paramCounter == 8)
            {
                iss >> newFlushNum;
                if (newFlushNum <= 0)
                {
                    std::cerr << "newFlushNum must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "newFlushNum=" << newFlushNum << std::endl;
                paramCounter++;
                continue;
            } //end newFlushNum

            //read flushLastFile
            if (paramCounter == 9)
            {
                iss >> flushLastFile;
                std::cout << "flushLastFile=" << flushLastFile << std::endl;
                paramCounter++;
                continue;
            } //end flushLastFile

            //read TDirRoot
            if (paramCounter == 10)
            {
                iss >> TDirRoot;
                std::cout << "TDirRoot=" << TDirRoot << std::endl;
                paramCounter++;
                continue;
            } //end TDirRoot

            //read U_s_dataDir
            if (paramCounter == 11)
            { iss >> U_s_dataDir;
                std::cout << "U_s_dataDir=" << U_s_dataDir << std::endl;
                paramCounter++;
                continue;

            } //end U_s_dataDir

            //read sweep_multiple
            if (paramCounter == 12)
            {
                iss >> sweep_multiple;
                if (sweep_multiple <= 0)
                {
                    std::cerr << "sweep_multiple must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "sweep_multiple=" << sweep_multiple << std::endl;
                paramCounter++;
                continue;
            }//end sweep_multiple

            //read num_parallel
            if (paramCounter == 13)
            {
                iss>>this->num_parallel;
                std::cout<<"num_parallel="<<num_parallel<<std::endl;
                paramCounter++;
                continue;
            }//end num_parallel

        }//end while
        //allocate memory for data
        try
        {
            this->U_data_all_ptr = new double[sweepToWrite];
            this->s_all_ptr = new double[sweepToWrite * total_components_num];

            this->s_init=new double[total_components_num];
            this->s_angle_init=new double[tot_angle_components_num];
            this->s_angle_all_ptr= new double[sweepToWrite * tot_angle_components_num];

            this->M_all_ptr=new double[sweepToWrite*3];

        }
        catch (const std::bad_alloc& e)
        {
            std::cerr << "Memory allocation error: " << e.what() << std::endl;
            std::exit(2);
        } catch (const std::exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            std::exit(2);
        }
        this->out_U_path = this->U_s_dataDir + "/U/";
        if (!fs::is_directory(out_U_path) || !fs::exists(out_U_path))
        {
            fs::create_directories(out_U_path);
        }

        this->out_s_angle_path = this->U_s_dataDir + "/s_angles/";
        if (!fs::is_directory(out_s_angle_path) || !fs::exists(out_s_angle_path))
        {
            fs::create_directories(out_s_angle_path);
        }

        this->out_M_path= this->U_s_dataDir + "/M/";
        if (!fs::is_directory(out_M_path) || !fs::exists(out_M_path))
        {
            fs::create_directories(out_M_path);
        }

        this->theta_left_end=0;
        this->theta_right_end=2.1*PI;

        this->phi_left_end=0;
        this->phi_right_end=1.1*PI;
        this->h=0.1;//step size
        std::cout<<"theta_left_end="<<theta_left_end<<", theta_right_end="<<theta_right_end<<std::endl;
        std::cout<<"phi_left_end="<<phi_left_end<<", phi_right_end="<<phi_right_end<<std::endl;
        std::cout<<"h="<<h<<std::endl;
    }//end constructor

    // Destructor
    ~mc_computation() {
        delete[] U_data_all_ptr;  // Use delete[] for arrays!
        delete []s_all_ptr;
        delete []s_init;
        delete [] M_all_ptr;
        delete[]s_angle_all_ptr;
        delete []s_angle_init;
    } //end Destructor
public:

    void init_and_run();

    ///
    /// @param s_vec_init all spin components in 1 configuration
    /// @param s_angle_vec_init angles corresponding to s_curr
    /// @param flushNum number of flushes
    void execute_mc(double * s_vec_init, double * s_angle_vec_init,const int& flushNum);

    ///
    ///compute M in parallel for all configurations
    void compute_all_magnetizations_parallel();
    ///
    /// @param Mx average value of M, x direction
    /// @param My average value of M, y direction
    /// @param Mz average value of M, z direction
    /// @param startInd tarting index of 1 configuration
    /// @param length 3*N0*N1
    void compute_M_avg_over_sites(double &Mx, double &My, double &Mz,const int &startInd, const int & length);

    ///
    /// @param s_curr all spin components in 1 configuration
    /// @param s_angle_curr angles corresponding to s_curr
    void update_spins_parallel_1_sweep(double *s_curr,double *s_angle_curr);



    ///
    /// @param s_vec_curr all current spins
    /// @param angle_vec_curr all current angles
    /// @param flattened_ind flattened index of the spin to update, this function updates phi
    void update_1_phi_1_site(double*s_vec_curr,double *angle_vec_curr,const int& flattened_ind);

    ///
    /// @param s_vec_curr all current spins
    /// @param angle_vec_curr all current angles
    /// @param flattened_ind flattened index of the spin to update, this function updates theta
    void update_1_theta_1_site(double*s_vec_curr,double *angle_vec_curr,const int& flattened_ind);

    ///
    /// @param phi_curr current phi value
    /// @param phi_next next phi value
    /// @param dE energy change
    /// @return acceptance ratio
    double acceptanceRatio_uni_phi(const double &phi_curr, const double &phi_next,const double& dE);

    ///
    /// @param theta_curr current theta value
    /// @param theta_next next theta value
    /// @param dE energy change
    /// @return acceptance ratio
    double acceptanceRatio_uni_theta(const double &theta_curr, const double &theta_next,const double& dE);


    ///
    /// @param x proposed value
    /// @param y current value
    /// @param a left end of interval
    /// @param b right end of interval
    /// @param epsilon half length
    /// @return proposal probability S(x|y)
    double S_uni(const double& x, const double& y, const double& a, const double& b, const double& epsilon);

    ///
    /// @param s_vec_curr  all current spin values
    /// @param angle_vec_curr all current angle values
    /// @param flattened_ind flattened index of the spin to update
    /// @param phi_next proposed phi value
    /// @return change in energy
    double delta_energy_update_phi(const double*s_vec_curr, const double* angle_vec_curr,const int& flattened_ind,const double & phi_next);

    ///
    /// @param s_vec_curr all current spin values
    /// @param angle_vec_curr all current angle values
    /// @param flattened_ind flattened index of the spin to update
    /// @param theta_next proposed theta value
    /// @return change in energy
    double delta_energy_update_theta(const double*s_vec_curr, const double* angle_vec_curr, const int& flattened_ind, const double & theta_next);

    ///
    /// @param sy_curr sy, current value
    /// @param sy_next sy, next value
    /// @param sy_neighbor sy, neighbor
    /// @return change in Kitaev energy, y term
    double delta_energy_kitaev_y(const double &sy_curr,const double &sy_next,const double &sy_neighbor);

    ///
    /// @param sx_curr sx, current value
    /// @param sx_next sx, next value
    /// @param sx_neighbor sx, neighbor
    /// @return change in Kitaev energy, x term
    double delta_energy_kitaev_x(const double & sx_curr,const double & sx_next,const double & sx_neighbor);

    /// @param sx_curr sx, current value
    /// @param sy_curr sy, current value
    /// @param sz_curr sz, current value
    /// @param sx_next sx, next value
    /// @param sy_next sy, next value
    /// @param sz_next sz, next value
    /// @param sx_neighbor sx, neighbor
    /// @param sy_neighbor sy, neighbor
    /// @param sz_neighbor sz, neighbor
    /// @return change in diagonal biquadratic energy
    double delta_energy_biquadratic_diagonal(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                           const double & sx_next, const double &sy_next, const double & sz_next,
                                           const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor);

    /// @param sx_curr sx, current value
    /// @param sy_curr sy, current value
    /// @param sz_curr sz, current value
    /// @param sx_next sx, next value
    /// @param sy_next sy, next value
    /// @param sz_next sz, next value
    /// @param sx_neighbor sx, neighbor
    /// @param sy_neighbor sy, neighbor
    /// @param sz_neighbor sz, neighbor
    /// @return change in nearest neighbor biquadratic energy
    double delta_energy_biquadratic_nearest_neighbor(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                           const double & sx_next, const double &sy_next, const double & sz_next,
                                           const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor);




    /// @param sx_curr sx, current value
    /// @param sy_curr sy, current value
    /// @param sz_curr sz, current value
    /// @param sx_next sx, next value
    /// @param sy_next sy, next value
    /// @param sz_next sz, next value
    /// @param sx_neighbor sx, neighbor
    /// @param sy_neighbor sy, neighbor
    /// @param sz_neighbor sz, neighbor
    /// @return change in diagonal Heisenberg energy
    double delta_energy_Heisenberg_diagonal(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                           const double & sx_next, const double &sy_next, const double & sz_next,
                                           const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor);
    ///
    /// @param sx_curr sx, current value
    /// @param sy_curr sy, current value
    /// @param sz_curr sz, current value
    /// @param sx_next sx, next value
    /// @param sy_next sy, next value
    /// @param sz_next sz, next value
    /// @param sx_neighbor sx, neighbor
    /// @param sy_neighbor sy, neighbor
    /// @param sz_neighbor sz, neighbor
    /// @return change in nearest neighbor Heisenberg energy
    double delta_energy_Heisenberg_nearest(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                           const double & sx_next, const double &sy_next, const double & sz_next,
                                           const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor);

    ///
    /// @param phi_curr current value of phi, 1 spin
    /// @param phi_next next value of phi, 1 spin
    void proposal_uni_phi(const double& phi_curr, double & phi_next);


    ///
    /// @param theta_curr current value of theta, 1 spin
    /// @param theta_next next value of theta, 1 spin
    void proposal_uni_theta(const double& theta_curr, double & theta_next);

    ///
    /// @param x
    /// @param leftEnd
    /// @param rightEnd
    /// @param eps
    /// @return return a value within distance eps from x, on the open interval (leftEnd, rightEnd)
    double generate_uni_open_interval(const double& x, const double& leftEnd, const double& rightEnd,
                                      const double& eps);

    double energy_tot(const double * s_vec);

    ///
    /// @param flattened_ind_center (flattened) index of spin to be updated
    /// @param ind_neighbor index of spin around the center spin (0..1)
    /// @param s_vec flattened s array
    /// @return Kitaev energy of flattened_ind_center and ind_neighbor, y neighbors
    double H_local_Kitaev_y(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec);

    ///
    /// @param flattened_ind_center (flattened) index of spin to be updated
    /// @param ind_neighbor index of spin around the center spin (0..1)
    /// @param s_vec flattened s array
    /// @return Kitaev energy of flattened_ind_center and ind_neighbor, x neighbors
    double H_local_Kitaev_x(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec);

    ///
    /// @param flattened_ind_center (flattened) index of spin to be updated
    /// @param ind_neighbor index of spin around the center spin (0..3)
    /// @param s_vec flattened s array
    /// @return biquadratic energy of flattened_ind_center and ind_neighbor, diagonal neighbors
    double H_local_biquadratic_diagonal(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec);


    ///
    /// @param flattened_ind_center (flattened) index of spin to be updated
    /// @param ind_neighbor index of spin around the center spin (0..3)
    /// @param s_vec flattened s array
    /// @return biquadratic energy of flattened_ind_center and ind_neighbor, nearest neighbors
    double H_local_biquadratic_nearest_neighbor(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec);

    ///
    /// @param flattened_ind_center (flattened) index of spin to be updated
    /// @param ind_neighbor index of spin around the center spin (0..3)
    /// @param s_vec flattened s array
    /// @return Heisenberg energy of flattened_ind_center and ind_neighbor, diagonal neighbors
    double H_local_Heisenberg_diagonal(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec);

    ///
    /// @param flattened_ind_center (flattened) index of spin to be updated
    /// @param ind_neighbor index of spin around the center spin (0..3)
    /// @param s_vec flattened s array
    /// @return Heisenberg energy of flattened_ind_center and ind_neighbor, nearest neighbors
    double H_local_Heisenberg_nearest(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec);

    ///
    /// @param angle_vec containing all angles
    /// @param flattened_ind flattened index of a spin
    /// @param theta parameter theta of the spin
    /// @param phi parameter phi of the spin
    inline void get_angle_components(const double* angle_vec, int flattened_ind, double &theta, double &phi)
    {
        const double * angle_ptr=angle_vec+(flattened_ind*2);
        theta=angle_ptr[0];
        phi=angle_ptr[1];

    }
    /// @param s_vec containing components of all spins
    /// @param flattened_ind flattened index of a spin
    /// @param s_x x component of this spin
    /// @param s_y y component of this spin
    /// @param s_z z component of this spin
    inline void get_spin_components(const double* s_vec, int flattened_ind,
                                   double& s_x, double& s_y, double& s_z)
    {
        const double* spin_ptr = s_vec + (flattened_ind * 3);
        s_x = spin_ptr[0];
        s_y = spin_ptr[1];
        s_z = spin_ptr[2];
    }



    //construct neighbors of each point, flattened index
    void init_flattened_ind_neighbors();
    //initialize A, B, C, D sublattices, flattened index
    void init_A_B_C_D_sublattices_flattened();

    ///construct nearest neighbors and diagonal neighbors around (0, 0)
    void construct_neighbors_origin();
    //initialize A, B, C, D sublattices for checkerboard update
    void init_A_B_C_D_sublattices();

    ///
    /// @param n0
    /// @param n1
    /// @return flatenned index
    int double_ind_to_flat_ind(const int& n0, const int& n1);

    void init_s();


    ///
    /// @param theta angle
    /// @param phi angle
    /// @param sx sx component of spin
    /// @param sy sy component of spin
    /// @param sz sz component of spin
    void angles_to_spin(const double &theta, const double &phi, double &sx, double &sy, double &sz);
    int mod_direction0(const int&m0);

    int mod_direction1(const int&m1);
    void save_array_to_pickle(const double* ptr, int size, const std::string& filename);
    void load_pickle_data(const std::string& filename,  double* data_ptr, std::size_t size);
    // Template function to print the contents of a std::vector<T>
    template <typename T>
    void print_vector(const std::vector<T>& vec)
    {
        // Check if the vector is empty
        if (vec.empty())
        {
            std::cout << "Vector is empty." << std::endl;
            return;
        }

        // Print each element with a comma between them
        for (size_t i = 0; i < vec.size(); ++i)
        {
            // Print a comma before all elements except the first one
            if (i > 0)
            {
                std::cout << ", ";
            }
            std::cout << vec[i];
        }
        std::cout << std::endl;
    }

public:
    double T; // temperature
    double beta;
    double J11, J12, J21, J22;
    double K;
    int N0;
    int N1;
    int num_parallel;
    int sweepToWrite;
    int newFlushNum;
    int flushLastFile;
    std::string TDirRoot;
    std::string U_s_dataDir;
    int sweep_multiple;
    std::string out_U_path;
    std::string out_s_angle_path;
    std::string out_M_path;
    int lattice_num;//N0*N1
    int total_components_num;//3*N0*N1
    int tot_angle_components_num;//2*N0*N1
    double h;//step size

    double theta_left_end;
    double theta_right_end;
    double phi_left_end;
    double phi_right_end;

    // std::ranlux24_base e2;
    // std::uniform_real_distribution<> distUnif01;

    //data in 1 flush
    double * U_data_all_ptr; //all U data
    double * s_all_ptr; //all s data
    double * M_all_ptr; //all M data
    double * s_angle_all_ptr;//all angle data, corresponds to s

    //initial value
    double * s_init;
    double *s_angle_init;

    //sublattices for checkerboard update
    std::vector<std::vector<int>> A_sublattice,B_sublattice,C_sublattice,D_sublattice;
    std::vector<std::vector<int>> nearest_neigbors;//around (0,0)
    std::vector<std::vector<int>> diagonal_neighbors;//around (0,0)

    std::vector<std::vector<int>> neighbors_x;//around (0,0), for Kitaev x

    std::vector<std::vector<int>> neighbors_y;//around (0,0), for Kitaev y


    //flatten index for lattice points
    std::vector<int> flattened_A_points;
    std::vector<int> flattened_B_points;
    std::vector<int> flattened_C_points;
    std::vector<int> flattened_D_points;

    std::vector<std::vector<int>> flattened_ind_nearest_neighbors;
    std::vector<std::vector<int>> flattened_ind_diagonal_neighbors;

    std::vector<std::vector<int>> flattened_ind_x_neighbors;//for Kitaev x
    std::vector<std::vector<int>> flattened_ind_y_neighbors;//for Kitaev y
};











#endif //MC_READ_LOAD_COMPUTE_HPP
