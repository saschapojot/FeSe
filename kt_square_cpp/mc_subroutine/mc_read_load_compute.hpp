//
// Created by adada on 7/14/2025.
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
                this->total_components_num=3*N0*N1;
                std::cout<<"total_components_num="<<total_components_num<<std::endl;
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
        } //end while

        //allocate memory for data
        try
        {
            this->U_data_all_ptr = new double[sweepToWrite];
            this->s_all_ptr = new double[sweepToWrite * total_components_num];

            this->s_init=new double[total_components_num];

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
        this->out_s_path = this->U_s_dataDir + "/s/";
        if (!fs::is_directory(out_s_path) || !fs::exists(out_s_path))
        {
            fs::create_directories(out_s_path);
        }
        this->out_M_path= this->U_s_dataDir + "/M/";
        if (!fs::is_directory(out_M_path) || !fs::exists(out_M_path))
        {
            fs::create_directories(out_M_path);
        }

    } //end constructor

    // Destructor
    ~mc_computation() {
        delete[] U_data_all_ptr;  // Use delete[] for arrays!
        delete []s_all_ptr;
        delete []s_init;
        delete [] M_all_ptr;
    } //end Destructor

public:
    void init_and_run();

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

    /// @param s_vec containing 3 components of all spins
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
    template <class T>
   void print_shared_ptr(std::shared_ptr<T> ptr, const int& size)
    {
        if (!ptr)
        {
            std::cout << "Pointer is null." << std::endl;
            return;
        }

        for (int i = 0; i < size; i++)
        {
            if (i < size - 1)
            {
                std::cout << ptr[i] << ",";
            }
            else
            {
                std::cout << ptr[i] << std::endl;
            }
        }
    } //end print_shared_ptr

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
    std::string out_s_path;
    std::string out_M_path;
    int total_components_num;//3*N0*N1

    //data in 1 flush
    double * U_data_all_ptr; //all U data
      double * s_all_ptr; //all s data
     double * M_all_ptr; //all M data

    //initial value
     double * s_init;

    //sublattices for checkerboard update
    std::vector<std::vector<int>> A_sublattice,B_sublattice,C_sublattice,D_sublattice;

    std::vector<std::vector<int>> nearest_neigbors;//around (0,0)
    std::vector<std::vector<int>> diagonal_neighbors;//around (0,0)

    std::vector<std::vector<int>> neighbors_x;//around (0,0), for Kitaev x

    std::vector<std::vector<int>> neighbors_y;//around (0,0), for Kitaev y

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
