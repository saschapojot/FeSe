//
// Created by adada on 9/1/2025.
//

#ifndef MC_READ_LOAD_COMPUTE_HPP
#define MC_READ_LOAD_COMPUTE_HPP
// Include necessary libraries for file system operations, Python integration, and numerical computations

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
// Namespace aliases for convenience
namespace fs = boost::filesystem;
namespace py = boost::python;
namespace np = boost::python::numpy;
// Mathematical constant PI
constexpr double PI = M_PI;


/**
 * @class mc_computation
 * @brief Main class for Monte Carlo simulation of a spin system on a 2D lattice
 *
 * This class implements a Monte Carlo simulation for a spin system with:
 * - Heisenberg interactions (nearest neighbor and diagonal)
 * - Biquadratic interactions (nearest neighbor and diagonal)
 * - Kitaev interactions (directional x and y)
 *
 * The lattice is divided into 4 sublattices (A, B, C, D) for checkerboard updates
 * to enable parallel processing without conflicts.
 */
class mc_computation
{
public:
    /**
     * @brief Constructor - reads parameters from file and initializes the simulation
     * @param cppInParamsFileName Path to input parameter file
     *
     * The parameter file should contain (one per line):
     * 1. Temperature (T)
     * 2. J11 - Heisenberg coupling, nearest neighbor
     * 3. J12 - Biquadratic coupling, nearest neighbor
     * 4. J21 - Heisenberg coupling, diagonal
     * 5. J22 - Biquadratic coupling, diagonal
     * 6. K - Kitaev coupling
     * 7. N - Lattice size (must be even, lattice will be N×N)
     * 8. sweepToWrite - writing data for each sweepToWrite sweeps
     * 10. flushLastFile - Index of last flush file (-1 for new simulation)
     * 11. TDirRoot - Root directory for temperature data
     * 12. U_s_dataDir - Directory for energy and spin data
     * 13. sweep_multiple - Number of sweeps between saved configurations
     * 14. num_parallel - Number of parallel threads to use
     */
    mc_computation(const std::string& cppInParamsFileName)
    {
        // Open the parameter file
        std::ifstream file(cppInParamsFileName);
        if (!file.is_open())
        {
            std::cerr << "Failed to open the file." << std::endl;
            std::exit(20);
        }
        std::string line;
        int paramCounter = 0; // Tracks which parameter we're reading
        // Read parameters line by line
        while (std::getline(file, line))
        {
            // Skip empty lines
            if (line.empty())
            {
                continue;
            }
            std::istringstream iss(line);
            // Read Temperature (T)
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

            // Read J11 - Heisenberg coupling for nearest neighbors
            if (paramCounter == 1)
            {
                iss >> J11;
                std::cout << "J11=" << J11 << std::endl;
                paramCounter++;
                continue;
            } //end J11
            // Read J12 - Biquadratic coupling for nearest neighbors
            if (paramCounter == 2)
            {
                iss >> J12;
                std::cout << "J12=" << J12 << std::endl;
                paramCounter++;
                continue;
            } //end J12

            // Read J21 - Heisenberg coupling for diagonal neighbors
            if (paramCounter == 3)
            {
                iss >> J21;
                std::cout << "J21=" << J21 << std::endl;
                paramCounter++;
                continue;
            } //end J21

            // Read J22 - Biquadratic coupling for diagonal neighbors
            if (paramCounter == 4)
            {
                iss >> J22;
                std::cout << "J22=" << J22 << std::endl;
                paramCounter++;
                continue;
            } //end J22
            // Read K - Kitaev coupling strength
            if (paramCounter == 5)
            {
                iss >> K;
                std::cout << "K=" << K << std::endl;
                paramCounter++;
                continue;
            } //end K

            // Read N - Lattice dimension (creates N×N lattice)
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
                // Calculate total number of lattice sites
                this->lattice_num=N0*N1;
                // Total spin components (3 per site: sx, sy, sz)
                this->total_components_num=3*N0*N1;
                // Total angle components (2 per site: theta, phi)
                this->tot_angle_components_num=2*N0*N1;
                std::cout<<"total_components_num="<<total_components_num<<std::endl;
                std::cout<<"tot_angle_components_num="<<tot_angle_components_num<<std::endl;
                std::cout<<"lattice_num="<<lattice_num<<std::endl;
                paramCounter++;
                continue;
            } //end N

            // Read sweepToWrite - number of Monte Carlo sweeps before writing data
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

            // Read newFlushNum - number of times to flush data to disk
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

            // Read flushLastFile - index of last saved flush (-1 means start from scratch)
            if (paramCounter == 9)
            {
                iss >> flushLastFile;
                std::cout << "flushLastFile=" << flushLastFile << std::endl;
                paramCounter++;
                continue;
            } //end flushLastFile

            // Read TDirRoot - root directory for temperature-dependent data
            if (paramCounter == 10)
            {
                iss >> TDirRoot;
                std::cout << "TDirRoot=" << TDirRoot << std::endl;
                paramCounter++;
                continue;
            } //end TDirRoot

            // Read U_s_dataDir - directory for energy (U) and spin (s) data
            if (paramCounter == 11)
            { iss >> U_s_dataDir;
                std::cout << "U_s_dataDir=" << U_s_dataDir << std::endl;
                paramCounter++;
                continue;

            } //end U_s_dataDir

            // Read sweep_multiple - perform sweep_multiple*sweepToWrite sweeps between saved configurations
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

            // Read num_parallel - number of parallel threads for computation
            if (paramCounter == 13)
            {
                iss>>this->num_parallel;
                std::cout<<"num_parallel="<<num_parallel<<std::endl;
                paramCounter++;
                continue;
            }//end num_parallel

        }// end while - finished reading all parameters
        // Allocate memory for data storage
        try
        {
            // Energy values for each saved configuration
            this->U_data_all_ptr = new double[sweepToWrite];

            // Spin components for all configurations (sx, sy, sz for each site)
            this->s_all_ptr = new double[sweepToWrite * total_components_num];

            // Initial spin components
            this->s_init=new double[total_components_num];

            // Initial angle values
            this->s_angle_init=new double[tot_angle_components_num];

            // Angle values for all configurations
            this->s_angle_all_ptr= new double[sweepToWrite * tot_angle_components_num];
            // Magnetization values (Mx, My, Mz) for each configuration
            this->M_all_ptr=new double[sweepToWrite*3];

            //order parameter values (val_x,val_y,val_z) for each configuration
            this->order_parameter_all_ptr=new double[sweepToWrite*3];

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

        // Create output directories if they don't exist
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

        this->out_order_parameter_path= this->U_s_dataDir + "/out_order/";
        if (!fs::is_directory(out_order_parameter_path) || !fs::exists(out_order_parameter_path))
        {
            fs::create_directories(out_order_parameter_path);
        }


        // Set bounds for angle proposals
        // theta ranges from 0 to slightly more than 2π
        this->theta_left_end=0;
        this->theta_right_end=2.0*PI;
        // phi ranges from 0 to slightly more than π
        this->phi_left_end=0;
        this->phi_right_end=1.0*PI;

        // Step size for proposal distribution
        this->h=0.1;
        std::cout<<"theta_left_end="<<theta_left_end<<", theta_right_end="<<theta_right_end<<std::endl;
        std::cout<<"phi_left_end="<<phi_left_end<<", phi_right_end="<<phi_right_end<<std::endl;
        std::cout<<"h="<<h<<std::endl;
    }//end constructor

    // Destructor

    /**
     * @brief Destructor - frees allocated memory
     */
    ~mc_computation() {
        delete[] U_data_all_ptr;  // Use delete[] for arrays!
        delete []s_all_ptr;
        delete []s_init;
        delete [] M_all_ptr;
        delete[]s_angle_all_ptr;
        delete []s_angle_init;
        delete[] order_parameter_all_ptr;
    } //end Destructor
public:

    /**
    * @brief Initialize the simulation and run Monte Carlo
    *
    * This function:
    * 1. Initializes spin configuration
    * 2. Sets up sublattices for checkerboard updates
    * 3. Constructs neighbor lists
    * 4. Executes the Monte Carlo simulation
    */
    void init_and_run();

    /**
    * @brief Execute Monte Carlo simulation
    * @param s_vec_init Initial spin components (sx, sy, sz for each site)
    * @param s_angle_vec_init Initial angles (theta, phi for each site)
    * @param flushNum Number of data flushes to perform
    *
    * Performs Monte Carlo sweeps, updates spins, computes energies,
    * and periodically saves data to disk.
    */
    void execute_mc(double * s_vec_init, double * s_angle_vec_init,const int& flushNum);

    /**
     * @brief Compute magnetization for all saved configurations in parallel
     *
     * For each configuration, computes average magnetization:
     * M = (1/N) * sum of all spins
     * Stores Mx, My, Mz for each configuration in M_all_ptr
     */
    void compute_all_magnetizations_parallel();
    /**
     * @brief Compute average magnetization over all sites for one configuration
     * @param Mx Output: x-component of magnetization
     * @param My Output: y-component of magnetization
     * @param Mz Output: z-component of magnetization
     * @param startInd Starting index in s_all_ptr for this configuration
     * @param length Number of components (should be 3*N0*N1)
     */
    void compute_M_avg_over_sites(double &Mx, double &My, double &Mz,const int &startInd, const int & length);


    /** by qyc
 * @brief Compute order parameter for all saved configurations in parallel
 *
 * For each configuration, computes order parameter:
 * val_α = (1/N) * Σ_i phase_i * s_α^i  for α = x, y, z
 * where phase_i = (-1)^(n0 mod 2)
 *
 * Stores val_x, val_y, val_z for each configuration in order_parameter_all_ptr
 */
    void compute_all_order_parameters_parallel();
    /// by qyc
    /// @param val_x x-component of order_parameter
    /// @param val_y y-component of order_parameter
    /// @param val_z z-component of order_parameter
    /// @param startInd Starting index in s_all_ptr for this configuration
    /// @param length Number of components (should be 3*N0*N1)
    void compute_order_parameter(double &val_x, double& val_y,double &val_z,const int &startInd, const int & length);

    /**
      * @brief Perform one Monte Carlo sweep updating all spins in parallel
      * @param s_curr Current spin configuration (all sx, sy, sz values)
      * @param s_angle_curr Current angles (all theta, phi values)
      *
      * Updates spins in checkerboard pattern:
      * 1. Update A sublattice (theta then phi)
      * 2. Update B sublattice (theta then phi)
      * 3. Update C sublattice (theta then phi)
      * 4. Update D sublattice (theta then phi)
      *
      * This ordering ensures no two neighboring spins are updated simultaneously.
      */
    void update_spins_parallel_1_sweep(double *s_curr,double *s_angle_curr);



    /**
     * @brief Update phi angle for one spin using Metropolis algorithm
     * @param s_vec_curr Current spin configuration
     * @param angle_vec_curr Current angle configuration
     * @param flattened_ind Index of spin to update (0 to N0*N1-1)
     *
     * Process:
     * 1. Propose new phi value
     * 2. Calculate energy change
     * 3. Accept/reject based on Metropolis criterion
     * 4. If accepted, update both angle and spin components
     */
    void update_1_phi_1_site(double*s_vec_curr,double *angle_vec_curr,const int& flattened_ind);

    /**
     * @brief Update theta angle for one spin using Metropolis algorithm
     * @param s_vec_curr Current spin configuration
     * @param angle_vec_curr Current angle configuration
     * @param flattened_ind Index of spin to update (0 to N0*N1-1)
     *
     * Process:
     * 1. Propose new theta value
     * 2. Calculate energy change
     * 3. Accept/reject based on Metropolis criterion
     * 4. If accepted, update both angle and spin components
     */
    void update_1_theta_1_site(double*s_vec_curr,double *angle_vec_curr,const int& flattened_ind);


    /**
     * @brief Calculate acceptance ratio for phi update
     * @param phi_curr Current phi value
     * @param phi_next Proposed phi value
     * @param dE Energy change (E_new - E_old)
     * @return Acceptance probability (between 0 and 1)
     *
     * Uses detailed balance with uniform proposal distribution:
     * A = min(1, exp(-β*dE) * S(phi_curr|phi_next) / S(phi_next|phi_curr))
     */
    double acceptanceRatio_uni_phi(const double &phi_curr, const double &phi_next,const double& dE);

    /**
      * @brief Calculate acceptance ratio for theta update
      * @param theta_curr Current theta value
      * @param theta_next Proposed theta value
      * @param dE Energy change (E_new - E_old)
      * @return Acceptance probability (between 0 and 1)
      *
      * Uses detailed balance with uniform proposal distribution:
      * A = min(1, exp(-β*dE) * S(theta_curr|theta_next) / S(theta_next|theta_curr))
      */
    double acceptanceRatio_uni_theta(const double &theta_curr, const double &theta_next,const double& dE);


    /**
     * @brief Proposal probability for uniform distribution
     * @param x Proposed value
     * @param y Current value
     * @param a Left boundary of allowed interval
     * @param b Right boundary of allowed interval
     * @param epsilon Half-width of proposal window
     * @return Probability S(x|y) of proposing x given current value y
     *
     * Proposal is uniform in window [y-ε, y+ε] intersected with (a, b)
     */
    double S_uni(const double& x, const double& y, const double& a, const double& b, const double& epsilon);


    /**
     * @brief Calculate energy change when updating phi
     * @param s_vec_curr Current spin configuration
     * @param angle_vec_curr Current angle configuration
     * @param flattened_ind Index of spin being updated
     * @param phi_next Proposed new phi value
     * @return Change in total energy (E_new - E_old)
     *
     * Computes change in:
     * - Nearest neighbor Heisenberg energy
     * - Diagonal Heisenberg energy
     * - Nearest neighbor biquadratic energy
     * - Diagonal biquadratic energy
     * - Kitaev x energy
     * - Kitaev y energy
     */
    double delta_energy_update_phi(const double*s_vec_curr, const double* angle_vec_curr,const int& flattened_ind,const double & phi_next);


    /**
     * @brief Calculate energy change when updating theta
     * @param s_vec_curr Current spin configuration
     * @param angle_vec_curr Current angle configuration
     * @param flattened_ind Index of spin being updated
     * @param theta_next Proposed new theta value
     * @return Change in total energy (E_new - E_old)
     *
     * Computes change in all energy terms (same as delta_energy_update_phi)
     */
    double delta_energy_update_theta(const double*s_vec_curr, const double* angle_vec_curr, const int& flattened_ind, const double & theta_next);

    /**
    * @brief Change in Kitaev y-bond energy
    * @param sy_curr Current sy component
    * @param sy_next Proposed sy component
    * @param sy_neighbor sy component of neighbor
    * @return K * (sy_next - sy_curr) * sy_neighbor
    */
    double delta_energy_kitaev_y(const double &sy_curr,const double &sy_next,const double &sy_neighbor);

    /**
     * @brief Change in Kitaev x-bond energy
     * @param sx_curr Current sx component
     * @param sx_next Proposed sx component
     * @param sx_neighbor sx component of neighbor
     * @return K * (sx_next - sx_curr) * sx_neighbor
     */
    double delta_energy_kitaev_x(const double & sx_curr,const double & sx_next,const double & sx_neighbor);

    /**
    * @brief Change in diagonal biquadratic energy
    * @param sx_curr, sy_curr, sz_curr Current spin components
    * @param sx_next, sy_next, sz_next Proposed spin components
    * @param sx_neighbor, sy_neighbor, sz_neighbor Neighbor spin components
    * @return J22 * [(S_next · S_neighbor)² - (S_curr · S_neighbor)²]
    */
    double delta_energy_biquadratic_diagonal(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                           const double & sx_next, const double &sy_next, const double & sz_next,
                                           const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor);

    /// @param sx_curr sx, current value

    /**
     * @brief Change in nearest neighbor biquadratic energy
     * @param sx_curr, sy_curr, sz_curr Current spin components
     * @param sx_next, sy_next, sz_next Proposed spin components
     * @param sx_neighbor, sy_neighbor, sz_neighbor Neighbor spin components
     * @return J12 * [(S_next · S_neighbor)² - (S_curr · S_neighbor)²]
     */
    double delta_energy_biquadratic_nearest_neighbor(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                           const double & sx_next, const double &sy_next, const double & sz_next,
                                           const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor);




    /**
    * @brief Change in diagonal Heisenberg energy
    * @param sx_curr, sy_curr, sz_curr Current spin components
    * @param sx_next, sy_next, sz_next Proposed spin components
    * @param sx_neighbor, sy_neighbor, sz_neighbor Neighbor spin components
    * @return J21 * [(S_next - S_curr) · S_neighbor]
    */
    double delta_energy_Heisenberg_diagonal(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                           const double & sx_next, const double &sy_next, const double & sz_next,
                                           const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor);
    ///
    /// @param sx_curr sx, current value
    /**
     * @brief Change in nearest neighbor Heisenberg energy
     * @param sx_curr, sy_curr, sz_curr Current spin components
     * @param sx_next, sy_next, sz_next Proposed spin components
     * @param sx_neighbor, sy_neighbor, sz_neighbor Neighbor spin components
     * @return J11 * [(S_next - S_curr) · S_neighbor]
     */
    double delta_energy_Heisenberg_nearest(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                           const double & sx_next, const double &sy_next, const double & sz_next,
                                           const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor);

    /**
     * @brief Propose new phi value using uniform distribution
     * @param phi_curr Current phi value
     * @param phi_next Output: proposed phi value
     *
     * Samples uniformly from interval [phi_curr - h, phi_curr + h]
     * intersected with (phi_left_end, phi_right_end)
     */
    void proposal_uni_phi(const double& phi_curr, double & phi_next);



    /**
     * @brief Propose new theta value using uniform distribution
     * @param theta_curr Current theta value
     * @param theta_next Output: proposed theta value
     *
     * Samples uniformly from interval [theta_curr - h, theta_curr + h]
     * intersected with (theta_left_end, theta_right_end)
     */
    void proposal_uni_theta(const double& theta_curr, double & theta_next);

    /**
    * @brief Generate random value from uniform distribution on open interval
    * @param x Center point
    * @param leftEnd Left boundary of interval
    * @param rightEnd Right boundary of interval
    * @param eps Maximum distance from x
    * @return Random value in (max(leftEnd, x-eps), min(rightEnd, x+eps))
    */
    double generate_uni_open_interval(const double& x, const double& leftEnd, const double& rightEnd,
                                      const double& eps);

    /**
     * @brief Calculate total energy of a configuration
     * @param s_vec Spin configuration (all sx, sy, sz values)
     * @return Total energy summing all interaction terms
     *
     * Computes:
     * E = 0.5 * (E_Heisenberg_nn + E_Heisenberg_diag +
     *            E_biquadratic_nn + E_biquadratic_diag +
     *            E_Kitaev_x + E_Kitaev_y)
     *
     * Factor of 0.5 corrects for double-counting of bonds
     */
    double energy_tot(const double * s_vec);

    /**
     * @brief Local Kitaev y-bond energy
     * @param flattened_ind_center Index of central spin
     * @param ind_neighbor Index in neighbor list (0 or 1)
     * @param s_vec Spin configuration
     * @return K * sy_center * sy_neighbor
     */
    double H_local_Kitaev_y(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec);

    /**
     * @brief Local Kitaev x-bond energy
     * @param flattened_ind_center Index of central spin
     * @param ind_neighbor Index in neighbor list (0 or 1)
     * @param s_vec Spin configuration
     * @return K * sx_center * sx_neighbor
     */
    double H_local_Kitaev_x(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec);

    /**
      * @brief Local diagonal biquadratic energy
      * @param flattened_ind_center Index of central spin
      * @param ind_neighbor Index in neighbor list (0..3)
      * @param s_vec Spin configuration
      * @return J22 * (S_center · S_neighbor)²
      */
    double H_local_biquadratic_diagonal(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec);


    /**
     * @brief Local nearest neighbor biquadratic energy
     * @param flattened_ind_center Index of central spin
     * @param ind_neighbor Index in neighbor list (0..3)
     * @param s_vec Spin configuration
     * @return J12 * (S_center · S_neighbor)²
     */
    double H_local_biquadratic_nearest_neighbor(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec);

    /**
     * @brief Local diagonal Heisenberg energy
     * @param flattened_ind_center Index of central spin
     * @param ind_neighbor Index in neighbor list (0..3)
     * @param s_vec Spin configuration
     * @return J21 * (S_center · S_neighbor)
     */
    double H_local_Heisenberg_diagonal(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec);

    /**
     * @brief Local nearest neighbor Heisenberg energy
     * @param flattened_ind_center Index of central spin
     * @param ind_neighbor Index in neighbor list (0..3)
     * @param s_vec Spin configuration
     * @return J11 * (S_center · S_neighbor)
     */
    double H_local_Heisenberg_nearest(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec);

    /**
     * @brief Extract angle components for a specific spin
     * @param angle_vec Array containing all angles
     * @param flattened_ind Index of spin
     * @param theta Output: theta value for this spin
     * @param phi Output: phi value for this spin
     */
    inline void get_angle_components(const double* angle_vec, int flattened_ind, double &theta, double &phi)
    {
        // Each spin has 2 angles, so multiply index by 2
        const double * angle_ptr=angle_vec+(flattened_ind*2);
        theta=angle_ptr[0];// First angle is theta
        phi=angle_ptr[1]; // Second angle is phi

    }
    /**
     * @brief Extract spin components for a specific spin
     * @param s_vec Array containing all spin components
     * @param flattened_ind Index of spin
     * @param s_x Output: x-component
     * @param s_y Output: y-component
     * @param s_z Output: z-component
     */
    inline void get_spin_components(const double* s_vec, int flattened_ind,
                                   double& s_x, double& s_y, double& s_z)
    {
        // Each spin has 3 components, so multiply index by 3
        const double* spin_ptr = s_vec + (flattened_ind * 3);
        s_x = spin_ptr[0];// First component is sx
        s_y = spin_ptr[1];// Second component is sy
        s_z = spin_ptr[2];// Third component is sz
    }



    /**
    * @brief Initialize neighbor lists for all lattice sites
    *
    * For each site, stores flattened indices of:
    * - Nearest neighbors (up, down, left, right)
    * - Diagonal neighbors (4 diagonal directions)
    * - x-direction neighbors (for Kitaev x bonds)
    * - y-direction neighbors (for Kitaev y bonds)
    */
    void init_flattened_ind_neighbors();
    /**
     * @brief Initialize flattened indices for A, B, C, D sublattices
     *
     * Converts 2D lattice coordinates to flattened 1D indices
     * for efficient array access during updates
     */
    void init_A_B_C_D_sublattices_flattened();

    /**
    * @brief Construct neighbor offset patterns around origin (0,0)
    *
    * Defines relative positions of neighbors in 2D lattice:
    * - nearest_neighbors: [(−1,0), (1,0), (0,−1), (0,1)]
    * - diagonal_neighbors: [(−1,−1), (−1,1), (1,−1), (1,1)]
    * - neighbors_x: [(−1,0), (1,0)]
    * - neighbors_y: [(0,−1), (0,1)]
    */
    void construct_neighbors_origin();

    /**
  * @brief Initialize A, B, C, D sublattices for checkerboard pattern
  *
  * Divides lattice into 4 sublattices:
  * - A: even i, even j
  * - B: even i, odd j
  * - C: odd i, even j
  * - D: odd i, odd j
  *
  * This ensures no two spins in the same sublattice are neighbors
  */
    void init_A_B_C_D_sublattices();

    /**
     * @brief Convert 2D lattice coordinates to flattened 1D index
     * @param n0 Index in first dimension (0 to N0-1)
     * @param n1 Index in second dimension (0 to N1-1)
     * @return Flattened index = n0 * N1 + n1
     */
    int double_ind_to_flat_ind(const int& n0, const int& n1);


    /**
     * @brief Initialize spin configuration
     *
     * Either:
     * - Loads from saved file if continuing simulation (flushLastFile ≥ 0)
     * - Loads initial configuration if starting new simulation (flushLastFile = -1)
     *
     * Converts angles (theta, phi) to spin components (sx, sy, sz)
     */
    void init_s();


    /**
   * @brief Convert spherical angles to Cartesian spin components
   * @param theta Azimuthal angle (0 to 2π)
   * @param phi Polar angle (0 to π)
   * @param sx Output: x-component = cos(θ)sin(φ)
   * @param sy Output: y-component = sin(θ)sin(φ)
   * @param sz Output: z-component = cos(φ)
   */
    void angles_to_spin(const double &theta, const double &phi, double &sx, double &sy, double &sz);

    /**
     * @brief Apply periodic boundary conditions in direction 0
     * @param m0 Index in direction 0 (may be outside [0, N0))
     * @return Wrapped index in range [0, N0)
     */
    int mod_direction0(const int&m0);

    /**
     * @brief Apply periodic boundary conditions in direction 1
     * @param m1 Index in direction 1 (may be outside [0, N1))
     * @return Wrapped index in range [0, N1)
     */
    int mod_direction1(const int&m1);

    /**
     * @brief Save array to pickle file for Python compatibility
     * @param ptr Pointer to data array
     * @param size Number of elements in array
     * @param filename Output file path
     *
     * Uses Boost.Python to serialize NumPy array to pickle format
     */
    void save_array_to_pickle(const double* ptr, int size, const std::string& filename);


    /**
     * @brief Load array from pickle file
     * @param filename Input file path
     * @param data_ptr Pointer to array where data will be stored
     * @param size Expected number of elements
     *
     * Uses Boost.Python to deserialize pickle file to C++ array
     */
    void load_pickle_data(const std::string& filename,  double* data_ptr, std::size_t size);


    /**
     * @brief Print contents of a vector (template function for any type)
     * @param vec Vector to print
     *
     * Prints elements separated by commas
     */
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
    // ===== Physical Parameters =====
    double T;      ///< Temperature
    double beta;   ///< Inverse temperature (1/T) for Boltzmann factor
    double J11;    ///< Heisenberg coupling, nearest neighbor
    double J12;    ///< Biquadratic coupling, nearest neighbor
    double J21;    ///< Heisenberg coupling, diagonal
    double J22;    ///< Biquadratic coupling, diagonal
    double K;      ///< Kitaev coupling strength

    // ===== Lattice Parameters =====
    int N0;        ///< Lattice size in direction 0
    int N1;        ///< Lattice size in direction 1 (equals N0 for square lattice)
    int lattice_num;   ///< Total number of sites (N0 × N1)
    int total_components_num;      ///< Total spin components (3 × N0 × N1)
    int tot_angle_components_num;  ///< Total angle components (2 × N0 × N1)

    // ===== Monte Carlo Parameters =====
    int sweepToWrite;    ///< Number of sweeps before writing data
    int newFlushNum;     ///< Number of data flushes
    int flushLastFile;   ///< Index of last saved flush (-1 for new simulation)
    int sweep_multiple;  ///< Sweeps between saved configurations
    double h;            ///< Step size for angle proposals
    int num_parallel;  ///< Number of parallel threads


    // ===== Angle Bounds =====
    double theta_left_end;   ///< Minimum theta value (0)
    double theta_right_end;  ///< Maximum theta value (slightly > 2π)
    double phi_left_end;     ///< Minimum phi value (0)
    double phi_right_end;    ///< Maximum phi value (slightly > π)

    // ===== Directory Paths =====
    std::string TDirRoot;       ///< Root directory for temperature data
    std::string U_s_dataDir;    ///< Directory for energy and spin data
    std::string out_U_path;     ///< Output path for energy data
    std::string out_s_angle_path;  ///< Output path for angle data
    std::string out_M_path;     ///< Output path for magnetization data
    std::string out_order_parameter_path; ///< Output path for order parameter data

    // ===== Data Storage =====
    double* U_data_all_ptr;      ///< Energy for each saved configuration
    double* s_all_ptr;           ///< Spin components for all configurations
    double* M_all_ptr;           ///< Magnetization (Mx, My, Mz) for all configurations
    double* s_angle_all_ptr;     ///< Angles for all configurations
    double* s_init;              ///< Initial spin configuration
    double* s_angle_init;        ///< Initial angle configuration
    double* order_parameter_all_ptr; ///< order parameter (val_x, va_y, val_z) for all configurations

    // ===== Sublattice Structure =====
    std::vector<std::vector<int>> A_sublattice;  ///< Sites with even i, even j
    std::vector<std::vector<int>> B_sublattice;  ///< Sites with even i, odd j
    std::vector<std::vector<int>> C_sublattice;  ///< Sites with odd i, even j
    std::vector<std::vector<int>> D_sublattice;  ///< Sites with odd i, odd j

    // ===== Neighbor Patterns (relative to origin) =====
    std::vector<std::vector<int>> nearest_neigbors;    ///< 4 nearest neighbors
    std::vector<std::vector<int>> diagonal_neighbors;  ///< 4 diagonal neighbors
    std::vector<std::vector<int>> neighbors_x;         ///< 2 neighbors for Kitaev x
    std::vector<std::vector<int>> neighbors_y;         ///< 2 neighbors for Kitaev y

    // ===== Flattened Sublattice Indices =====
    std::vector<int> flattened_A_points;  ///< Flattened indices for A sublattice
    std::vector<int> flattened_B_points;  ///< Flattened indices for B sublattice
    std::vector<int> flattened_C_points;  ///< Flattened indices for C sublattice
    std::vector<int> flattened_D_points;  ///< Flattened indices for D sublattice

    // ===== Neighbor Lists (for each site) =====
    std::vector<std::vector<int>> flattened_ind_nearest_neighbors;   ///< Nearest neighbor indices
    std::vector<std::vector<int>> flattened_ind_diagonal_neighbors;  ///< Diagonal neighbor indices
    std::vector<std::vector<int>> flattened_ind_x_neighbors;         ///< x-direction neighbor indices
    std::vector<std::vector<int>> flattened_ind_y_neighbors;         ///< y-direction neighbor indices



};











#endif //MC_READ_LOAD_COMPUTE_HPP
