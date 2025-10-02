//
// Created by adada on 9/1/2025.
//

#include "mc_read_load_compute.hpp"


/**
 * @brief Load data from pickle file into C++ array
 * @param filename Path to pickle file
 * @param data_ptr Pointer to array where data will be stored
 * @param size Expected size of data
 *
 * Process:
 * 1. Initialize Python interpreter and NumPy
 * 2. Open pickle file using Python's io module
 * 3. Load pickled NumPy array
 * 4. Convert to Python list
 * 5. Copy data to C++ array
 */
void mc_computation::load_pickle_data(const std::string& filename, double* data_ptr,
                                      std::size_t size)
{
    // Initialize Python interpreter (required for Boost.Python)
    Py_Initialize();

    // Initialize NumPy C API
    np::initialize();


    try
    {
        // Import Python's 'io' module for file operations
        py::object io_module = py::import("io");
        // Open file in binary read mode
        py::object file = io_module.attr("open")(filename, "rb");

        // Import pickle module for deserialization
        py::object pickle_module = py::import("pickle");

        // Deserialize the pickle file to Python object
        py::object loaded_data = pickle_module.attr("load")(file);

        // Close the file
        file.attr("close")();

        // Check if loaded object is a NumPy array
        if (py::extract<np::ndarray>(loaded_data).check())
        {
            // Extract as NumPy array
            np::ndarray np_array = py::extract<np::ndarray>(loaded_data);

            // Convert NumPy array to Python list for easier element access
            py::object py_list = np_array.attr("tolist")();

            // Get size of loaded data
            ssize_t list_size = py::len(py_list);

            // Verify size matches expected size
            if (static_cast<std::size_t>(list_size) > size)
            {
                throw std::runtime_error("The provided shared_ptr array size is smaller than the list size.");
            }

            // Copy data from Python list to C++ array
            for (ssize_t i = 0; i < list_size; ++i)
            {
                data_ptr[i] = py::extract<double>(py_list[i]);
            }
        }
        else
        {
            throw std::runtime_error("Loaded data is not a NumPy array.");
        }
    }
    catch (py::error_already_set&)
    {
        // Print Python error traceback
        PyErr_Print();
        throw std::runtime_error("Python error occurred.");
    }
}


/**
 * @brief Save C++ array to pickle file for Python compatibility
 * @param ptr Pointer to data array
 * @param size Number of elements in array
 * @param filename Output file path
 *
 * Process:
 * 1. Initialize Python/NumPy if needed
 * 2. Convert C++ array to NumPy array
 * 3. Serialize using pickle.dumps
 * 4. Write binary data to file
 */
void mc_computation::save_array_to_pickle(const double* ptr, int size, const std::string& filename)
{
    using namespace boost::python;
    namespace np = boost::python::numpy;

    // Initialize Python interpreter if not already initialized
    if (!Py_IsInitialized())
    {
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            throw std::runtime_error("Failed to initialize Python interpreter");
        }
        np::initialize(); // Initialize NumPy
    }

    try
    {
        // Import the pickle module
        object pickle = import("pickle");
        object pickle_dumps = pickle.attr("dumps");

        // Convert C++ array to NumPy array
        // np::from_data creates a NumPy array that views the C++ data
        np::ndarray numpy_array = np::from_data(
            ptr, // Pointer to data
            np::dtype::get_builtin<double>(), // Data type (double)
            boost::python::make_tuple(size), // Shape of the array (1D array)
            boost::python::make_tuple(sizeof(double)), // Strides
            object() // Optional base object
        );

        // Serialize the NumPy array using pickle.dumps
        object serialized_array = pickle_dumps(numpy_array);

        // Extract the serialized data as a string
        std::string serialized_str = extract<std::string>(serialized_array);

        // Write the serialized data to a file
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Failed to open file for writing");
        }
        file.write(serialized_str.data(), serialized_str.size());
        file.close();

        // Debug output (optional)
        // std::cout << "Array serialized and written to file successfully." << std::endl;
    }
    catch (const error_already_set&)
    {
        PyErr_Print();
        std::cerr << "Boost.Python error occurred." << std::endl;
    } catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}


/**
 * @brief Apply periodic boundary conditions in direction 0
 * @param m0 Index (can be negative or >= N0)
 * @return Index wrapped to [0, N0)
 *
 * Uses modulo arithmetic: ((m0 % N0) + N0) % N0
 * This handles negative indices correctly
 */
int mc_computation::mod_direction0(const int& m0)
{
    return ((m0 % N0) + N0) % N0;
}


/**
 * @brief Apply periodic boundary conditions in direction 1
 * @param m1 Index (can be negative or >= N1)
 * @return Index wrapped to [0, N1)
 */
int mc_computation::mod_direction1(const int& m1)
{
    return ((m1 % N1) + N1) % N1;
}

///
/**
 * @brief Convert spherical angles to Cartesian spin components
 * @param theta Azimuthal angle
 * @param phi Polar angle
 * @param sx Output: x-component
 * @param sy Output: y-component
 * @param sz Output: z-component
 *
 * Standard spherical to Cartesian conversion:
 * sx = cos(θ) * sin(φ)
 * sy = sin(θ) * sin(φ)
 * sz = cos(φ)
 */
void mc_computation::angles_to_spin(const double &theta, const double &phi, double &sx, double &sy, double &sz)
{
    sx=std::cos(theta)*std::sin(phi);
    sy=std::sin(theta)*sin(phi);
    sz=std::cos(phi);
}


/**
 * @brief Initialize spin configuration from file or previous run
 *
 * If flushLastFile == -1:
 *   Load initial configuration from "s_angle_init.pkl"
 * Else:
 *   Load final configuration from previous flush
 *
 * Then convert all angles to spin components
 */
void mc_computation::init_s()
{
    std::string name;
    std::string s_angle_inFileName;

    // Determine which file to load based on whether this is a new or continued simulation
    if (this->flushLastFile == -1)
    {
        // New simulation - load initial configuration
        name = "init";
        s_angle_inFileName=this->out_s_angle_path+"/s_angle_"+name+".pkl";
        this->load_pickle_data(s_angle_inFileName,s_angle_init,tot_angle_components_num);
    } //end flushLastFile==-1
    else
    {
        // Continue from previous simulation - load last saved configuration
        name = "flushEnd" + std::to_string(this->flushLastFile);
        s_angle_inFileName = this->out_s_angle_path + "/" + name + ".s_angle_final.pkl";
        // Load angle data
        this->load_pickle_data(s_angle_inFileName,s_angle_init,tot_angle_components_num);


    }//end else

    // Convert angles to spin components for all lattice sites
    for (int j=0;j<lattice_num;j++)
    {
        // Get angles for this site (2 values per site: theta and phi)
        double theta_tmp=s_angle_init[2*j];
        double phi_tmp=s_angle_init[2*j+1];

        // Convert to spin components
        double sx_tmp,sy_tmp,sz_tmp;
        this->angles_to_spin(theta_tmp,phi_tmp,sx_tmp,sy_tmp,sz_tmp);

        // Store spin components (3 values per site: sx, sy, sz)
        s_init[3*j]=sx_tmp;
        s_init[3*j+1]=sy_tmp;
        s_init[3*j+2]=sz_tmp;
    }//end for
}

/**
 * @brief Convert 2D lattice coordinates to 1D flattened index
 * @param n0 Index in direction 0 (row)
 * @param n1 Index in direction 1 (column)
 * @return Flattened index = n0 * N1 + n1
 *
 * This is standard row-major ordering for 2D arrays
 */
int mc_computation::double_ind_to_flat_ind(const int& n0, const int& n1)
{
    return n0 * N1 + n1;
}

/**
 * @brief Initialize A, B, C, D sublattices for checkerboard updates
 *
 * Divides N0×N1 lattice into 4 sublattices based on parity of indices:
 * - A sublattice: i even, j even
 * - B sublattice: i even, j odd
 * - C sublattice: i odd, j even
 * - D sublattice: i odd, j odd
 *
 * Each sublattice contains N0*N1/4 sites
 * No two sites in the same sublattice are nearest neighbors
 */
void mc_computation::init_A_B_C_D_sublattices()
{
    int sublat_num = static_cast<int>(N0*N1/4);

    // Reserve space for efficiency (avoid repeated reallocations)
    this->A_sublattice.reserve(sublat_num);
    this->B_sublattice.reserve(sublat_num);
    this->C_sublattice.reserve(sublat_num);
    this->D_sublattice.reserve(sublat_num);

    // Iterate through all lattice sites
    for (int i = 0; i < N0; i++)
    {
        for (int j = 0; j < N1; j++)
        {
            // Assign to sublattice based on parity of both indices
            if (i%2 == 0 && j%2 == 0)
            {
                A_sublattice.push_back({i,j});
            }
            else if (i%2 == 0 && j%2 == 1)
            {
                B_sublattice.push_back({i,j});
            }
            else if (i%2 == 1 && j%2 == 0)
            {
                C_sublattice.push_back({i,j});
            }
            else // i%2 == 1 && j%2 == 1
            {
                D_sublattice.push_back({i,j});
            }
        }//end for j
    }//end for i
}

/**
 * @brief Main initialization and execution function
 *
 * Performs full simulation setup:
 * 1. Initialize spins from file
 * 2. Set up sublattices
 * 3. Construct neighbor patterns
 * 4. Initialize flattened neighbor lists
 * 5. Run Monte Carlo simulation
 */
void mc_computation::init_and_run()
{
    // Load initial or previous spin configuration
    this->init_s();

    // Create checkerboard sublattices
    this->init_A_B_C_D_sublattices();

    // std::cout<<"D:"<<std::endl;
    // for (int n=0;n<D_sublattice.size();n++)
    // {
    //     print_vector(D_sublattice[n]);
    // }

    // Define neighbor offset patterns
    this->construct_neighbors_origin();

    // Convert sublattices to flattened indices
    this->init_A_B_C_D_sublattices_flattened();

    //print flattened index
    // std::cout<<"A: "<<std::endl;
    // print_vector(flattened_A_points);
    //
    // std::cout<<"B: "<<std::endl;
    // print_vector(flattened_B_points);
    //
    // std::cout<<"C: "<<std::endl;
    // print_vector(flattened_C_points);
    //
    // std::cout<<"D: "<<std::endl;
    // print_vector(flattened_D_points);

    // Build neighbor lists for all sites
    this->init_flattened_ind_neighbors();

    // Execute Monte Carlo simulation
    this->execute_mc(s_init,s_angle_init,newFlushNum);

    // for (int j=0;j<flattened_ind_nearest_neighbors.size();j++)
    // {
    //     std::cout<<"nearest neighbors of "<<j<<": \n";
    //     print_vector(flattened_ind_nearest_neighbors[j]);
    // }

    // for (int j=0;j<flattened_ind_diagonal_neighbors.size();j++)
    // {
    //     std::cout<<"diagonal neighbors of "<<j<<": \n";
    //     print_vector(flattened_ind_diagonal_neighbors[j]);
    // }

    // for (int j=0;j<flattened_ind_x_neighbors.size();j++)
    // {
    //     std::cout<<"x neighbors of "<<j<<": \n";
    //     print_vector(flattened_ind_x_neighbors[j]);
    // }

    // for (int j=0;j<flattened_ind_y_neighbors.size();j++)
    // {
    //     std::cout<<"y neighbors of "<<j<<": \n";
    //     print_vector(flattened_ind_y_neighbors[j]);
    // }

}



/**
 * @brief Construct neighbor offset patterns relative to origin (0,0)
 *
 * Defines displacement vectors for different types of neighbors:
 * - Nearest neighbors: 4 cardinal directions
 * - Diagonal neighbors: 4 diagonal directions
 * - x-neighbors: 2 horizontal neighbors (for Kitaev x-bonds)
 * - y-neighbors: 2 vertical neighbors (for Kitaev y-bonds)
 */
void mc_computation::construct_neighbors_origin()
{
    // Nearest neighbors:
    this->nearest_neigbors={{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    //print neighbors
    std::cout << "nearest neighbors:" << std::endl;
    for (const auto& vec:nearest_neigbors)
    {
        print_vector(vec);
    }//end for
    // Diagonal neighbors: 4 diagonal directions
    this->diagonal_neighbors={{-1,-1},{-1,1},{1,-1},{1,1}};
    std::cout << "diagonal neighbors:" << std::endl;
    for (const auto& vec:diagonal_neighbors)
    {
        print_vector(vec);
    }//end for

    // x-direction neighbors for Kitaev x-bonds
    this->neighbors_x={{-1,0},{1,0}};
    std::cout << "neighbors_x:" << std::endl;
    for (const auto& vec:neighbors_x)
    {
        print_vector(vec);
    }//end for

    // y-direction neighbors for Kitaev y-bonds
    this->neighbors_y={{0,-1},{0,1}};


    std::cout << "neighbors_y:" << std::endl;
    for (const auto& vec:neighbors_y)
    {
        print_vector(vec);
    }//end for

}




/**
 * @brief Convert sublattice coordinates to flattened indices
 *
 * For each sublattice (A, B, C, D), convert 2D coordinates [i, j]
 * to flattened 1D index for efficient array access
 */
void mc_computation::init_A_B_C_D_sublattices_flattened()
{
    int sublat_num = static_cast<int>(N0*N1/4);

    // Reserve space for all sublattice index vectors
    this->flattened_A_points.reserve(sublat_num);
    this->flattened_B_points.reserve(sublat_num);
    this->flattened_C_points.reserve(sublat_num);
    this->flattened_D_points.reserve(sublat_num);
    int ind0;
    int ind1;
    int flat_ind;
    // Convert A sublattice coordinates to flattened indices
    for (const auto& vec: this->A_sublattice)
    {
        ind0=vec[0];
        ind1=vec[1];
        flat_ind=this->double_ind_to_flat_ind(ind0,ind1);
        flattened_A_points.push_back(flat_ind);
    }//end for A

    // Convert B sublattice coordinates to flattened indices
    for (const auto& vec: this->B_sublattice)
    {
        ind0=vec[0];
        ind1=vec[1];
        flat_ind=this->double_ind_to_flat_ind(ind0,ind1);
        flattened_B_points.push_back(flat_ind);
    }//end for B

    // Convert C sublattice coordinates to flattened indices
    for (const auto& vec: this->C_sublattice)
    {
        ind0=vec[0];
        ind1=vec[1];
        flat_ind=this->double_ind_to_flat_ind(ind0,ind1);
        flattened_C_points.push_back(flat_ind);
    }//end for C

    // Convert D sublattice coordinates to flattened indices
    for (const auto& vec: this->D_sublattice)
    {
        ind0=vec[0];
        ind1=vec[1];
        flat_ind=this->double_ind_to_flat_ind(ind0,ind1);
        flattened_D_points.push_back(flat_ind);
    }//end for D

}




/**
 * @brief Build neighbor lists for all lattice sites
 *
 * For each site, creates lists of flattened indices for:
 * - Nearest neighbors (4 sites)
 * - Diagonal neighbors (4 sites)
 * - x-direction neighbors (2 sites)
 * - y-direction neighbors (2 sites)
 *
 * Applies periodic boundary conditions using mod_direction0/1
 */
void mc_computation::init_flattened_ind_neighbors()
{
    // Initialize neighbor lists (one list per site)
    this->flattened_ind_nearest_neighbors=std::vector<std::vector<int>>(N0*N1,std::vector<int>());
    this->flattened_ind_diagonal_neighbors=std::vector<std::vector<int>>(N0*N1,std::vector<int>());
    this->flattened_ind_x_neighbors=std::vector<std::vector<int>>(N0*N1,std::vector<int>());
    this->flattened_ind_y_neighbors=std::vector<std::vector<int>>(N0*N1,std::vector<int>());

    // Build nearest neighbor lists
    for (int n0 = 0; n0 < N0; n0++)
    {
        for (int n1 = 0; n1 < N1; n1++)
        {
            // Get flattened index of current site
            int point_curr_flattened = this->double_ind_to_flat_ind(n0, n1);
            // Add each nearest neighbor
            for (const auto& vec_nghbrs:this->nearest_neigbors)
            {
                // Calculate neighbor coordinates
                int diff_direc0 = vec_nghbrs[0];
                int diff_direc1 = vec_nghbrs[1];
                int m0 = n0 + diff_direc0;
                int m1 = n1 + diff_direc1;

                // Apply periodic boundary conditions
                int m0_mod = mod_direction0(m0);
                int m1_mod = mod_direction1(m1);

                // Get flattened index of neighbor
                int flattened_ngb = double_ind_to_flat_ind(m0_mod, m1_mod);
                flattened_ind_nearest_neighbors[point_curr_flattened].push_back(flattened_ngb);
            }//end neighbors
        }//end n1
    }//end n0

    // Build diagonal neighbor lists (same process as above)
    for (int n0 = 0; n0 < N0; n0++)
    {
        for (int n1 = 0; n1 < N1; n1++)
        {
            int point_curr_flattened = this->double_ind_to_flat_ind(n0, n1);

            for (const auto& vec_nghbrs:this->diagonal_neighbors)
            {
                int diff_direc0 = vec_nghbrs[0];
                int diff_direc1 = vec_nghbrs[1];
                int m0 = n0 + diff_direc0;
                int m1 = n1 + diff_direc1;
                int m0_mod = mod_direction0(m0);
                int m1_mod = mod_direction1(m1);
                int flattened_ngb = double_ind_to_flat_ind(m0_mod, m1_mod);
                flattened_ind_diagonal_neighbors[point_curr_flattened].push_back(flattened_ngb);
            }//end neighbors

        }//end for n1
    }//end for n0

    // Build x-direction neighbor lists for Kitaev x-bonds
    for (int n0 = 0; n0 < N0; n0++)
    {
        for (int n1 = 0; n1 < N1; n1++)
        {
            int point_curr_flattened = this->double_ind_to_flat_ind(n0, n1);
            for (const auto& vec_nghbrs:this->neighbors_x)
            {
                int diff_direc0 = vec_nghbrs[0];
                int diff_direc1 = vec_nghbrs[1];
                int m0 = n0 + diff_direc0;
                int m1 = n1 + diff_direc1;
                int m0_mod = mod_direction0(m0);
                int m1_mod = mod_direction1(m1);
                int flattened_ngb = double_ind_to_flat_ind(m0_mod, m1_mod);
                flattened_ind_x_neighbors[point_curr_flattened].push_back(flattened_ngb);
            }//end neighbors
        }//end n1
    }//end n0
    // Build y-direction neighbor lists for Kitaev y-bonds
    for (int n0 = 0; n0 < N0; n0++)
    {
        for (int n1 = 0; n1 < N1; n1++)
        {
            int point_curr_flattened = this->double_ind_to_flat_ind(n0, n1);
            for (const auto& vec_nghbrs : this->neighbors_y)
            {
                int diff_direc0 = vec_nghbrs[0];
                int diff_direc1 = vec_nghbrs[1];
                int m0 = n0 + diff_direc0;
                int m1 = n1 + diff_direc1;
                int m0_mod = mod_direction0(m0);
                int m1_mod = mod_direction1(m1);
                int flattened_ngb = double_ind_to_flat_ind(m0_mod, m1_mod);
                flattened_ind_y_neighbors[point_curr_flattened].push_back(flattened_ngb);
            }//end neighbors

        }//end n1
    }//end n0
}


/**
 * @brief Calculate local Heisenberg energy for nearest neighbor pair
 * @param flattened_ind_center Index of central spin
 * @param ind_neighbor Which neighbor (0..3)
 * @param s_vec Spin configuration
 * @return J11 * (S_center · S_neighbor)
 */
double mc_computation::H_local_Heisenberg_nearest(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec)
{
    double center_s_x,center_s_y,center_s_z;
    double neighbor_s_x,neighbor_s_y,neighbor_s_z;
    // Get flattened index of this neighbor
    int flattened_ind_one_neighbor =this->flattened_ind_nearest_neighbors[flattened_ind_center][ind_neighbor];

    // Get spin components of central spin
    this->get_spin_components(s_vec,flattened_ind_center,center_s_x,center_s_y,center_s_z);

    // Get spin components of neighbor
    this->get_spin_components(s_vec, flattened_ind_one_neighbor, neighbor_s_x, neighbor_s_y, neighbor_s_z);
    // Calculate dot product
    double dot_product = center_s_x * neighbor_s_x +
                       center_s_y * neighbor_s_y +
                       center_s_z * neighbor_s_z;

    // Return Heisenberg energy for this bond
    return this->J11*dot_product;
}



/**
 * @brief Calculate local Heisenberg energy for diagonal neighbor pair
 * @param flattened_ind_center Index of central spin
 * @param ind_neighbor Which neighbor (0..3)
 * @param s_vec Spin configuration
 * @return J21 * (S_center · S_neighbor)
 */
double mc_computation::H_local_Heisenberg_diagonal(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec)
{
    double center_s_x,center_s_y,center_s_z;
    double neighbor_s_x,neighbor_s_y,neighbor_s_z;

    // Get flattened index of diagonal neighbor
    int flattened_ind_one_neighbor =this->flattened_ind_diagonal_neighbors[flattened_ind_center][ind_neighbor];
    //get spin values of center
    this->get_spin_components(s_vec,flattened_ind_center,center_s_x,center_s_y,center_s_z);

    // Get spin components of neighbor spin
    this->get_spin_components(s_vec, flattened_ind_one_neighbor, neighbor_s_x, neighbor_s_y, neighbor_s_z);
    // Calculate dot product
    double dot_product = center_s_x * neighbor_s_x +
                       center_s_y * neighbor_s_y +
                       center_s_z * neighbor_s_z;

    // Return diagonal Heisenberg energy for this bond
    return this->J21*dot_product;
}



/**
 * @brief Calculate local biquadratic energy for nearest neighbor pair
 * @param flattened_ind_center Index of central spin
 * @param ind_neighbor Which neighbor (0..3)
 * @param s_vec Spin configuration
 * @return J12 * (S_center · S_neighbor)²
 */
double mc_computation::H_local_biquadratic_nearest_neighbor(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec)
{
    double center_s_x,center_s_y,center_s_z;
    double neighbor_s_x,neighbor_s_y,neighbor_s_z;

    // Get neighbor index
    int flattened_ind_one_neighbor =this->flattened_ind_nearest_neighbors[flattened_ind_center][ind_neighbor];

    //get spin values of center
    this->get_spin_components(s_vec,flattened_ind_center,center_s_x,center_s_y,center_s_z);

    // Get spin components of neighbor spin
    this->get_spin_components(s_vec, flattened_ind_one_neighbor, neighbor_s_x, neighbor_s_y, neighbor_s_z);

    double dot_product = center_s_x * neighbor_s_x +
                       center_s_y * neighbor_s_y +
                       center_s_z * neighbor_s_z;

    // Return biquadratic energy (dot product squared) for this bond
    return this->J12*std::pow(dot_product,2.0);


}



/**
 * @brief Calculate local biquadratic energy for diagonal neighbor pair
 * @param flattened_ind_center Index of central spin
 * @param ind_neighbor Which neighbor (0..3)
 * @param s_vec Spin configuration
 * @return J22 * (S_center · S_neighbor)²
 */
double mc_computation::H_local_biquadratic_diagonal(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec)
{
    double center_s_x,center_s_y,center_s_z;
    double neighbor_s_x,neighbor_s_y,neighbor_s_z;
    // Get diagonal neighbor index
    int flattened_ind_one_neighbor =this->flattened_ind_diagonal_neighbors[flattened_ind_center][ind_neighbor];
    //get spin values of center
    this->get_spin_components(s_vec,flattened_ind_center,center_s_x,center_s_y,center_s_z);

    // Get spin components of neighbor spin
    this->get_spin_components(s_vec, flattened_ind_one_neighbor, neighbor_s_x, neighbor_s_y, neighbor_s_z);
    // Calculate dot product
    double dot_product = center_s_x * neighbor_s_x +
                       center_s_y * neighbor_s_y +
                       center_s_z * neighbor_s_z;

    // Return diagonal biquadratic energy  for this bond
    return this->J22*std::pow(dot_product,2.0);
}


/**
 * @brief Calculate local Kitaev x-bond energy
 * @param flattened_ind_center Index of central spin
 * @param ind_neighbor Which x-neighbor (0 or 1)
 * @param s_vec Spin configuration
 * @return K * sx_center * sx_neighbor
 */
double mc_computation::H_local_Kitaev_x(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec)
{
    double center_s_x,center_s_y,center_s_z;
    double neighbor_s_x,neighbor_s_y,neighbor_s_z;

    // Get x-direction neighbor index
    int flattened_ind_one_neighbor =this->flattened_ind_x_neighbors[flattened_ind_center][ind_neighbor];
    //get spin values of center
    this->get_spin_components(s_vec,flattened_ind_center,center_s_x,center_s_y,center_s_z);

    // Get spin components of neighbor spin
    this->get_spin_components(s_vec, flattened_ind_one_neighbor, neighbor_s_x, neighbor_s_y, neighbor_s_z);
    // Calculate product of x-components only
    double prod=center_s_x*neighbor_s_x;

    // Return Kitaev x-bond energy
    return this->K*prod;
}


/**
 * @brief Calculate local Kitaev y-bond energy
 * @param flattened_ind_center Index of central spin
 * @param ind_neighbor Which y-neighbor (0 or 1)
 * @param s_vec Spin configuration
 * @return K * sy_center * sy_neighbor
 */
double mc_computation::H_local_Kitaev_y(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec)
{
    double center_s_x,center_s_y,center_s_z;
    double neighbor_s_x,neighbor_s_y,neighbor_s_z;

    // Get y-direction neighbor index
    int flattened_ind_one_neighbor =this->flattened_ind_y_neighbors[flattened_ind_center][ind_neighbor];
    //get spin values of center
    this->get_spin_components(s_vec,flattened_ind_center,center_s_x,center_s_y,center_s_z);

    // Get spin components of neighbor spin
    this->get_spin_components(s_vec, flattened_ind_one_neighbor, neighbor_s_x, neighbor_s_y, neighbor_s_z);
    // Calculate product of y-components only
    double prod=center_s_y*neighbor_s_y;

    // Return Kitaev y-bond energy
    return this->K*prod;

}


/**
 * @brief Generate random value uniformly on open interval
 * @param x Center point
 * @param leftEnd Left boundary
 * @param rightEnd Right boundary
 * @param eps Maximum distance from x
 * @return Random value in (max(leftEnd, x-eps), min(rightEnd, x+eps))
 *
 * Uses thread-local random number generator for thread safety
 */
double mc_computation::generate_uni_open_interval(const double& x, const double& leftEnd, const double& rightEnd,
                                                  const double& eps)
{
    // Thread-local random number generator (one per thread)
    thread_local std::random_device rd;
    thread_local std::ranlux24_base e2_local(rd());

    // Calculate proposal interval
    double xMinusEps = x - eps;
    double xPlusEps = x + eps;

    // Constrain to allowed interval
    double unif_left_end = xMinusEps < leftEnd ? leftEnd : xMinusEps;
    double unif_right_end = xPlusEps > rightEnd ? rightEnd : xPlusEps;

    //    std::random_device rd;
    //    std::ranlux24_base e2(rd());

    // Get next representable double to ensure open interval
    double unif_left_end_double_on_the_right = std::nextafter(unif_left_end, std::numeric_limits<double>::infinity());


    // Create uniform distribution on open interval
    std::uniform_real_distribution<> distUnif(unif_left_end_double_on_the_right, unif_right_end);
    //(unif_left_end_double_on_the_right, unif_right_end)

    // Sample and return
    double xNext = distUnif(e2_local);
    return xNext;
}



/**
 * @brief Propose new theta value
 * @param theta_curr Current theta
 * @param theta_next Output: proposed theta
 *
 * Samples uniformly from [theta_curr - h, theta_curr + h] ∩ (theta_left_end, theta_right_end)
 */
void mc_computation::proposal_uni_theta(const double& theta_curr, double & theta_next)
{

    theta_next=this->generate_uni_open_interval(theta_curr,theta_left_end,theta_right_end,h);
}


/**
 * @brief Propose new phi value
 * @param phi_curr Current phi
 * @param phi_next Output: proposed phi
 *
 * Samples uniformly from [phi_curr - h, phi_curr + h] ∩ (phi_left_end, phi_right_end)
 */
void mc_computation::proposal_uni_phi(const double& phi_curr, double & phi_next)
{
    phi_next=this->generate_uni_open_interval(phi_curr,phi_left_end,phi_right_end,h);
}


/**
 * @brief Calculate energy change when updating theta
 * @param s_vec_curr Current spin configuration
 * @param angle_vec_curr Current angle configuration
 * @param flattened_ind Index of spin to update
 * @param theta_next Proposed theta value
 * @return Total energy change ΔE = E_new - E_old
 *
 * Computes change in all energy terms involving this spin:
 * - Nearest neighbor Heisenberg
 * - Diagonal Heisenberg
 * - Nearest neighbor biquadratic
 * - Diagonal biquadratic
 * - Kitaev x
 * - Kitaev y
 */
double mc_computation::delta_energy_update_theta(const double*s_vec_curr, const double* angle_vec_curr, const int& flattened_ind, const double & theta_next)
{
    double sx_curr=0,sy_curr=0,sz_curr=0;
    double theta_curr=0,phi_curr=0;
    // Get current spin and angle values
    this->get_spin_components(s_vec_curr,flattened_ind,sx_curr,sy_curr,sz_curr);
    this->get_angle_components(angle_vec_curr,flattened_ind,theta_curr,phi_curr);

    double sx_next=0,sy_next=0,sz_next=0;
    // Calculate proposed spin components (phi unchanged)
    this->angles_to_spin(theta_next,phi_curr,sx_next,sy_next,sz_next);

    // Change in nearest neighbor Heisenberg energy
    double E_delta_Heisengerg_nn=0;
    double sx_neighbor, sy_neighbor, sz_neighbor;
    for (const int& ind: this->flattened_ind_nearest_neighbors[flattened_ind])
    {
        this->get_spin_components(s_vec_curr,ind,sx_neighbor,sy_neighbor,sz_neighbor);
        E_delta_Heisengerg_nn+=this->delta_energy_Heisenberg_nearest(sx_curr,sy_curr,sz_curr,sx_next,sy_next,sz_next,
           sx_neighbor,sy_neighbor, sz_neighbor);

    }//end for ind

    //change in diagonal Heisenberg energy
    double E_delta_Heisenberg_diag=0;
    for (const int & ind: this->flattened_ind_diagonal_neighbors[flattened_ind])
    {
        this->get_spin_components(s_vec_curr,ind,sx_neighbor,sy_neighbor,sz_neighbor);
        E_delta_Heisenberg_diag+=this->delta_energy_Heisenberg_diagonal(sx_curr,sy_curr,sz_curr,
            sx_next,sy_next,sz_next,
           sx_neighbor, sy_neighbor,sz_neighbor);
    }//end for ind

    //change in nearest neighbor biquadratic energy
    double E_delta_bq_nn=0;
    for (const int& ind:this->flattened_ind_nearest_neighbors[flattened_ind])
    {
        this->get_spin_components(s_vec_curr,ind,sx_neighbor,sy_neighbor,sz_neighbor);
        E_delta_bq_nn+=this->delta_energy_biquadratic_nearest_neighbor(sx_curr,sy_curr,sz_curr,
            sx_next,sy_next,sz_next,
            sx_neighbor,sy_neighbor,sz_neighbor);

    }//end for ind

    //change in diagonal biquadratic energy
    double E_delta_bq_diag=0;
    for (const int& ind:this->flattened_ind_diagonal_neighbors[flattened_ind])
    {
        this->get_spin_components(s_vec_curr,ind,sx_neighbor,sy_neighbor,sz_neighbor);
        E_delta_bq_diag+=this->delta_energy_biquadratic_diagonal(sx_curr,sy_curr,sz_curr,
            sx_next,sy_next,sz_next,
            sx_neighbor,sy_neighbor,sz_neighbor);
    }//end for ind

    //change in x Kitaev  energy
    double E_delta_kt_x=0;
    for (const int& ind:this->flattened_ind_x_neighbors[flattened_ind])
    {
        this->get_spin_components(s_vec_curr,ind,sx_neighbor,sy_neighbor,sz_neighbor);
        E_delta_kt_x+=this->delta_energy_kitaev_x(sx_curr,sx_next,sx_neighbor);
    }//end for ind

    //change in y Kitaev  energy
    double E_delta_kt_y=0;
    for (const int& ind: this->flattened_ind_y_neighbors[flattened_ind])
    {
        this->get_spin_components(s_vec_curr,ind,sx_neighbor,sy_neighbor,sz_neighbor);

        E_delta_kt_y+=this->delta_energy_kitaev_y(sy_curr,sy_next,sy_neighbor);

    }//end for ind

    // Return total energy change
    return E_delta_Heisengerg_nn+E_delta_Heisenberg_diag+E_delta_bq_nn+E_delta_bq_diag+E_delta_kt_x+E_delta_kt_y;

}//end delta_energy_update_theta





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
/// @return change in nearest neighbor Heisenberg energy (J11 * [(S_next - S_curr) · S_neighbor])
double  mc_computation::delta_energy_Heisenberg_nearest(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                       const double & sx_next, const double &sy_next, const double & sz_next,
                                       const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor)
{


    // Energy with current spin
    double E_curr=this->J11*(sx_curr*sx_neighbor+sy_curr*sy_neighbor+sz_curr*sz_neighbor);

    // Energy with proposed spin
    double E_next=this->J11*(sx_next*sx_neighbor+sy_next*sy_neighbor+sz_next*sz_neighbor);

    // Return change
    return E_next-E_curr;
}



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
/// @return change in diagonal Heisenberg energy, J21 * [(S_next - S_curr) · S_neighbor]
double mc_computation::delta_energy_Heisenberg_diagonal(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                       const double & sx_next, const double &sy_next, const double & sz_next,
                                       const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor)
{

    // Energy with current spin
    double E_curr=this->J21*(sx_curr*sx_neighbor+sy_curr*sy_neighbor+sz_curr*sz_neighbor);

    // Energy with proposed spin
    double E_next=this->J21*(sx_next*sx_neighbor+sy_next*sy_neighbor+sz_next*sz_neighbor);

    // Return change
    return E_next-E_curr;
}



/// @param sx_curr sx, current value
/// @param sy_curr sy, current value
/// @param sz_curr sz, current value
/// @param sx_next sx, next value
/// @param sy_next sy, next value
/// @param sz_next sz, next value
/// @param sx_neighbor sx, neighbor
/// @param sy_neighbor sy, neighbor
/// @param sz_neighbor sz, neighbor
/// @return change in diagonal biquadratic energy, J12 * [(S_next · S_neighbor)² - (S_curr · S_neighbor)²]
double  mc_computation::delta_energy_biquadratic_nearest_neighbor(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                       const double & sx_next, const double &sy_next, const double & sz_next,
                                       const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor)
{

    // Dot product with current spin
    double prod_curr=sx_curr*sx_neighbor+sy_curr*sy_neighbor+sz_curr*sz_neighbor;

    // Dot product with proposed spin
    double prod_next=sx_next*sx_neighbor+sy_next*sy_neighbor+sz_next*sz_neighbor;

    // Energy with current spin (squared dot product)
    double E_curr=this->J12*std::pow(prod_curr,2.0);

    // Energy with proposed spin
    double E_next=this->J12*std::pow(prod_next,2.0);

    // Return change
    return E_next-E_curr;


}




/// @param sx_curr sx, current value
/// @param sy_curr sy, current value
/// @param sz_curr sz, current value
/// @param sx_next sx, next value
/// @param sy_next sy, next value
/// @param sz_next sz, next value
/// @param sx_neighbor sx, neighbor
/// @param sy_neighbor sy, neighbor
/// @param sz_neighbor sz, neighbor
/// @return change in diagonal biquadratic energy, J22 * [(S_next · S_neighbor)² - (S_curr · S_neighbor)²]
double mc_computation::delta_energy_biquadratic_diagonal(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                       const double & sx_next, const double &sy_next, const double & sz_next,
                                       const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor)
{
    // Dot product with current spin
    double prod_curr=sx_curr*sx_neighbor+sy_curr*sy_neighbor+sz_curr*sz_neighbor;


    // Dot product with proposed spin
    double prod_next=sx_next*sx_neighbor+sy_next*sy_neighbor+sz_next*sz_neighbor;

    // Energy with current spin
    double E_curr=this->J22*std::pow(prod_curr,2.0);

    // Energy with proposed spin
    double E_next=this->J22*std::pow(prod_next,2.0);

    // Return change
    return E_next-E_curr;

}


///
/// @param sx_curr sx, current value
/// @param sx_next sx, next value
/// @param sx_neighbor sx, neighbor
/// @return change in Kitaev energy, x term, K * (sx_next - sx_curr) * sx_neighbor
double mc_computation::delta_energy_kitaev_x(const double & sx_curr,const double & sx_next,const double & sx_neighbor)
{

    // Energy with current x-component
    double E_curr=this->K*sx_curr*sx_neighbor;

    // Energy with proposed x-component
    double E_next=this->K*sx_next*sx_neighbor;

    // Return change
    return E_next-E_curr;
}


///
/// @param sy_curr sy, current value
/// @param sy_next sy, next value
/// @param sy_neighbor sy, neighbor
/// @return change in Kitaev energy, y term, K * (sy_next - sy_curr) * sy_neighbor
double mc_computation::delta_energy_kitaev_y(const double &sy_curr,const double &sy_next,const double &sy_neighbor)
{

    // Energy with current y-component
    double E_curr=this->K*sy_curr*sy_neighbor;

    // Energy with proposed y-component
    double E_next=this->K*sy_next*sy_neighbor;

    // Return change
    return E_next-E_curr;
}



/**
 * @brief Calculate energy change when updating phi
 * @param s_vec_curr Current spin configuration
 * @param angle_vec_curr Current angle configuration
 * @param flattened_ind Index of spin to update
 * @param phi_next Proposed phi value
 * @return Total energy change ΔE = E_new - E_old
 *
 * Same structure as delta_energy_update_theta, but updates phi instead of theta
 */
double mc_computation::delta_energy_update_phi(const double*s_vec_curr, const double* angle_vec_curr,const int& flattened_ind,const double & phi_next)
{
    double sx_curr=0,sy_curr=0,sz_curr=0;
    double theta_curr=0,phi_curr=0;
    // Get current values
    //s_vec_curr corresponds to  angle_vec_curr
    this->get_spin_components(s_vec_curr,flattened_ind,sx_curr,sy_curr,sz_curr);

    this->get_angle_components(angle_vec_curr,flattened_ind,theta_curr,phi_curr);
    double sx_next=0,sy_next=0,sz_next=0;
    // Calculate proposed spin (theta unchanged)
    this->angles_to_spin(theta_curr,phi_next,sx_next,sy_next,sz_next);
    // Calculate energy changes (same process as in delta_energy_update_theta)

    double E_delta_Heisengerg_nn=0;
    double sx_neighbor, sy_neighbor, sz_neighbor;
    for (const int& ind: this->flattened_ind_nearest_neighbors[flattened_ind])
    {
        this->get_spin_components(s_vec_curr,ind,sx_neighbor,sy_neighbor,sz_neighbor);
        E_delta_Heisengerg_nn+=this->delta_energy_Heisenberg_nearest(sx_curr,sy_curr,sz_curr,sx_next,sy_next,sz_next,
           sx_neighbor,sy_neighbor, sz_neighbor);

    }//end for ind

    //change in diagonal Heisenberg energy
    double E_delta_Heisenberg_diag=0;
    for (const int & ind: this->flattened_ind_diagonal_neighbors[flattened_ind])
    {
        this->get_spin_components(s_vec_curr,ind,sx_neighbor,sy_neighbor,sz_neighbor);
        E_delta_Heisenberg_diag+=this->delta_energy_Heisenberg_diagonal(sx_curr,sy_curr,sz_curr,
            sx_next,sy_next,sz_next,
           sx_neighbor, sy_neighbor,sz_neighbor);
    }//end for ind

    //change in nearest neighbor biquadratic energy
    double E_delta_bq_nn=0;
    for (const int& ind:this->flattened_ind_nearest_neighbors[flattened_ind])
    {
        this->get_spin_components(s_vec_curr,ind,sx_neighbor,sy_neighbor,sz_neighbor);
        E_delta_bq_nn+=this->delta_energy_biquadratic_nearest_neighbor(sx_curr,sy_curr,sz_curr,
            sx_next,sy_next,sz_next,
            sx_neighbor,sy_neighbor,sz_neighbor);

    }//end for ind
    //change in diagonal biquadratic energy
    double E_delta_bq_diag=0;
    for (const int& ind:this->flattened_ind_diagonal_neighbors[flattened_ind])
    {
        this->get_spin_components(s_vec_curr,ind,sx_neighbor,sy_neighbor,sz_neighbor);
        E_delta_bq_diag+=this->delta_energy_biquadratic_diagonal(sx_curr,sy_curr,sz_curr,
            sx_next,sy_next,sz_next,
            sx_neighbor,sy_neighbor,sz_neighbor);
    }//end for ind

    //change in x Kitaev  energy
    double E_delta_kt_x=0;
    for (const int& ind:this->flattened_ind_x_neighbors[flattened_ind])
    {
        this->get_spin_components(s_vec_curr,ind,sx_neighbor,sy_neighbor,sz_neighbor);
        E_delta_kt_x+=this->delta_energy_kitaev_x(sx_curr,sx_next,sx_neighbor);
    }//end for ind


    //change in y Kitaev  energy
    double E_delta_kt_y=0;
    for (const int& ind: this->flattened_ind_y_neighbors[flattened_ind])
    {
        this->get_spin_components(s_vec_curr,ind,sx_neighbor,sy_neighbor,sz_neighbor);

        E_delta_kt_y+=this->delta_energy_kitaev_y(sy_curr,sy_next,sy_neighbor);

    }//end for ind
    return E_delta_Heisengerg_nn+E_delta_Heisenberg_diag+E_delta_bq_nn+E_delta_bq_diag+E_delta_kt_x+E_delta_kt_y;

}

/**
 * @brief Calculate proposal probability for uniform distribution
 * @param x Proposed value (unused in calculation, kept for interface)
 * @param y Current value
 * @param a Left boundary
 * @param b Right boundary
 * @param epsilon Half-width of proposal window
 * @return Probability density S(x|y)
 *
 * The proposal distribution is uniform on [y-ε, y+ε] ∩ (a, b)
 * Probability = 1 / (width of proposal interval)
 */
double mc_computation::S_uni(const double& x, const double& y, const double& a, const double& b, const double& epsilon)
{
    // Case 1: y is close to left boundary
    if (a < y and y < a + epsilon)
    {
        // Proposal interval: (a, y+ε)
        return 1.0 / (y - a + epsilon);
    }
    // Case 2: y is in middle region
    else if (a + epsilon <= y and y < b - epsilon)
    {
        // Proposal interval: (y-ε, y+ε)
        return 1.0 / (2.0 * epsilon);
    }
    // Case 3: y is close to right boundary
    else if (b - epsilon <= y and y < b)
    {
        // Proposal interval: (y-ε, b)
        return 1.0 / (b - y + epsilon);
    }
    else
    {
        std::cerr << "value out of range." << std::endl;
        std::exit(10);
    }
}


/**
 * @brief Calculate acceptance ratio for theta update
 * @param theta_curr Current theta value
 * @param theta_next Proposed theta value
 * @param dE Energy change
 * @return Acceptance probability A = min(1, R) where
 *         R = exp(-β*ΔE) * S(θ_curr|θ_next) / S(θ_next|θ_curr)
 *
 * Uses Metropolis-Hastings with detailed balance
 */
double mc_computation::acceptanceRatio_uni_theta(const double &theta_curr, const double &theta_next,const double& dE)
{

    // Boltzmann factor
    double R = std::exp(-beta*dE);
    // std::cout<<"theta_curr="<<theta_curr<<", theta_next="<<theta_next<<std::endl;

    // Proposal probability ratio (for detailed balance)
    double S_curr_next = S_uni(theta_curr,theta_next,theta_left_end,theta_right_end,h);
    // std::cout<<"S_curr_next="<<S_curr_next<<std::endl;
    double S_next_curr=S_uni(theta_next,theta_curr,theta_left_end,theta_right_end,h);
    double ratio = S_curr_next / S_next_curr;
    // std::cout<<"S_next_curr="<<S_next_curr<<std::endl;

    // Check for division by zero
    if (std::fetestexcept(FE_DIVBYZERO))
    {
        std::cout << "Division by zero exception caught." << std::endl;
        std::exit(15);
    }
    if (std::isnan(ratio))
    {
        std::cout << "The result is NaN." << std::endl;
        std::exit(15);
    }

    // Multiply Boltzmann factor by proposal ratio
    R *= ratio;

    // Return min(1, R)
    return std::min(1.0, R);
}

/**
 * @brief Calculate acceptance ratio for phi update
 * @param phi_curr Current phi value
 * @param phi_next Proposed phi value
 * @param dE Energy change
 * @return Acceptance probability (same formula as theta)
 */
double mc_computation::acceptanceRatio_uni_phi(const double &phi_curr, const double &phi_next,const double& dE)
{


    // Boltzmann factor
    double R = std::exp(-beta*dE);
    // Proposal probability ratio
    double S_curr_next = S_uni(phi_curr,phi_next,phi_left_end,phi_right_end,h);

    double S_next_curr=S_uni(phi_next,phi_curr,phi_left_end,phi_right_end,h);
    double ratio = S_curr_next / S_next_curr;

    // Error checking
    if (std::fetestexcept(FE_DIVBYZERO))
    {
        std::cout << "Division by zero exception caught." << std::endl;
        std::exit(15);
    }
    if (std::isnan(ratio))
    {
        std::cout << "The result is NaN." << std::endl;
        std::exit(15);
    }
    R *= ratio;

    return std::min(1.0, R);

}

/**
 * @brief Update theta for one spin using Metropolis algorithm
 * @param s_vec_curr Current spin configuration (will be modified if accepted)
 * @param angle_vec_curr Current angle configuration (will be modified if accepted)
 * @param flattened_ind Index of spin to update
 *
 * Process:
 * 1. Get current theta and phi
 * 2. Propose new theta
 * 3. Calculate energy change
 * 4. Calculate acceptance ratio
 * 5. Accept or reject based on random number
 * 6. If accepted, update both angles and spin components
 */
void  mc_computation::update_1_theta_1_site(double*s_vec_curr,double *angle_vec_curr,const int& flattened_ind)
{
double theta_curr=0,phi_curr=0, theta_next=0;

    // Thread-local random number generator
    thread_local std::random_device rd;
    thread_local std::ranlux24_base e2_local(rd());
    thread_local std::uniform_real_distribution<> distUnif01_local;

    //get angle values
    this->get_angle_components(angle_vec_curr,flattened_ind,theta_curr,phi_curr);
    //propose next theta value
    this->proposal_uni_theta(theta_curr,theta_next);

    // Calculate energy change
    double dE=delta_energy_update_theta(s_vec_curr,angle_vec_curr,flattened_ind,theta_next);

    // Calculate acceptance probability
    double r=this->acceptanceRatio_uni_theta(theta_curr,theta_next,dE);

    // Generate random number for accept/reject decision
    double u = distUnif01_local(e2_local);
    // Accept if u <= r
    if (u<=r)
    {
        // Update angle in angle array
        int theta_ind=2*flattened_ind;
        angle_vec_curr[theta_ind]=theta_next;

        // Calculate new spin components
        double sx=0,sy=0,sz=0;
        this->angles_to_spin(theta_next,phi_curr,sx,sy,sz);
        // Update spin components in spin array
        int s_ind=3*flattened_ind;
        s_vec_curr[s_ind]=sx;
        s_vec_curr[s_ind+1]=sy;
        s_vec_curr[s_ind+2]=sz;


    }//end of accept-reject

}



/**
 * @brief Update phi for one spin using Metropolis algorithm
 * @param s_vec_curr Current spin configuration (will be modified if accepted)
 * @param angle_vec_curr Current angle configuration (will be modified if accepted)
 * @param flattened_ind Index of spin to update
 *
 * Same process as update_1_theta_1_site but for phi angle
 */
void mc_computation::update_1_phi_1_site(double*s_vec_curr,double *angle_vec_curr,const int& flattened_ind)
{

    // Thread-local RNG
    thread_local std::random_device rd;
    thread_local std::ranlux24_base e2_local(rd());
    thread_local std::uniform_real_distribution<> distUnif01_local;
    double theta_curr=0,phi_curr=0, phi_next=0;
    // Get current angles
    this->get_angle_components(angle_vec_curr,flattened_ind,theta_curr,phi_curr);
    //propose next phi value
    this->proposal_uni_phi(phi_curr,phi_next);
    // Calculate energy change
    double dE=delta_energy_update_phi(s_vec_curr,angle_vec_curr,flattened_ind,phi_next);

    // Calculate acceptance probability
    double r=this->acceptanceRatio_uni_phi(phi_curr,phi_next,dE);

    // Accept/reject
    double u =  distUnif01_local(e2_local);
    if (u<=r)
    {
        //update angle
        int phi_ind=2*flattened_ind+1;
        angle_vec_curr[phi_ind]=phi_next;

        // Calculate and update spin components
        double sx=0,sy=0,sz=0;
        this->angles_to_spin(theta_curr,phi_next,sx,sy,sz);
        //update spins
        int s_ind=3*flattened_ind;
        s_vec_curr[s_ind]=sx;
        s_vec_curr[s_ind+1]=sy;
        s_vec_curr[s_ind+2]=sz;
    }//end of accept-reject
}


/**
 * @brief Calculate total energy of configuration
 * @param s_vec Spin configuration
 * @return Total energy summing all interaction terms
 *
 * Computes:
 * - Sum of all Heisenberg nearest neighbor bonds
 * - Sum of all Heisenberg diagonal bonds
 * - Sum of all biquadratic nearest neighbor bonds
 * - Sum of all biquadratic diagonal bonds
 * - Sum of all Kitaev x bonds
 * - Sum of all Kitaev y bonds
 *
 * Factor of 0.5 corrects for double-counting (each bond counted from both ends)
 */
double mc_computation::energy_tot(const double * s_vec)
{
    //Heisenberg energy, nearest neighbor
    double E_Heisenberg_nn=0;
    for (int center_ind=0;center_ind<lattice_num;center_ind++)
    {
        for (int neighbor_idx = 0; neighbor_idx < flattened_ind_nearest_neighbors[center_ind].size(); neighbor_idx++)
        {
            E_Heisenberg_nn += this->H_local_Heisenberg_nearest(center_ind, neighbor_idx, s_vec);
        }//end for neighbor_idx
    }//end for center_ind

    //Heisenberg energy, diagonal
    double E_Heisenberg_diag=0;
    for (int center_ind=0;center_ind<lattice_num;center_ind++)
    {
        for (int neighbor_idx = 0; neighbor_idx < flattened_ind_diagonal_neighbors[center_ind].size(); neighbor_idx++)
        {
            E_Heisenberg_diag+=this->H_local_Heisenberg_diagonal(center_ind,neighbor_idx,s_vec);
        }//end for neighbor_idx
    }//end for center_ind

    //biquadratic energy, nearest neighbor
    double E_bq_nn=0;
    for (int center_ind=0;center_ind<lattice_num;center_ind++)
    {
        for (int neighbor_idx = 0; neighbor_idx < flattened_ind_nearest_neighbors[center_ind].size(); neighbor_idx++)
        {
            E_bq_nn+=this->H_local_biquadratic_nearest_neighbor(center_ind,neighbor_idx,s_vec);
        }//end for neighbor_idx
    }//end for center_ind

    //biquadratic energy, diagonal
    double E_bq_diag=0;
    for (int center_ind=0;center_ind<lattice_num;center_ind++)
    {
        for (int neighbor_idx = 0; neighbor_idx < flattened_ind_diagonal_neighbors[center_ind].size(); neighbor_idx++)
        {
            E_bq_diag+=this->H_local_biquadratic_diagonal(center_ind,neighbor_idx,s_vec);
        }//end for neighbor_idx
    }//end for center_ind

    // Kitaev x energy
    double E_kt_x=0;
    for (int center_ind=0;center_ind<lattice_num;center_ind++)
    {
        for (int neighbor_idx = 0; neighbor_idx < flattened_ind_x_neighbors[center_ind].size(); neighbor_idx++)
        {
            E_kt_x+=this->H_local_Kitaev_x(center_ind,neighbor_idx,s_vec);
        }//end for neighbor_idx
    }//end for center_ind

    // Kitaev y energy
    double E_kt_y=0;
    for (int center_ind=0;center_ind<lattice_num;center_ind++)
    {
        for (int neighbor_idx = 0; neighbor_idx < flattened_ind_y_neighbors[center_ind].size(); neighbor_idx++)
        {
            E_kt_y+=this->H_local_Kitaev_y(center_ind,neighbor_idx,s_vec);
        }//end for neighbor_idx
    }//end for center_ind

    // Return total energy (factor of 0.5 corrects for double-counting)
    return 0.5*(E_Heisenberg_nn+E_Heisenberg_diag+E_bq_nn+E_bq_diag+E_kt_x+E_kt_y);
}


/**
 * @brief Perform one Monte Carlo sweep with parallel updates
 * @param s_curr Current spin configuration
 * @param s_angle_curr Current angle configuration
 *
 * Updates all spins in checkerboard pattern:
 * 1. Update A sublattice theta (parallel)
 * 2. Update A sublattice phi (parallel)
 * 3. Update B sublattice theta (parallel)
 * 4. Update B sublattice phi (parallel)
 * 5. Update C sublattice theta (parallel)
 * 6. Update C sublattice phi (parallel)
 * 7. Update D sublattice theta (parallel)
 * 8. Update D sublattice phi (parallel)
 *
 * This order ensures no two neighboring spins are updated simultaneously
 */
void mc_computation::update_spins_parallel_1_sweep( double *s_curr,double *s_angle_curr)
{
    std::vector<std::thread> threads;


    ///////////////////////////////////////////////////////////////////////
    // Update A sublattice, theta
    ///////////////////////////////////////////////////////////////////////
    if (flattened_A_points.size() > 0) {
        // Determine number of threads to use
        int actual_threads_A = std::min(this->num_parallel, static_cast<int>(flattened_A_points.size()));
        int chunk_size = flattened_A_points.size() / actual_threads_A;

        // Ensure minimum chunk size of 1
        if (chunk_size == 0) {
            chunk_size = 1;
            actual_threads_A = flattened_A_points.size();
        }

        // Launch threads
        for (int t = 0; t < actual_threads_A; ++t) {
            int start_idx = t * chunk_size;
            int end_idx = (t == actual_threads_A - 1) ? flattened_A_points.size() : (t + 1) * chunk_size;

            // Each thread updates a chunk of A sublattice spins
            threads.emplace_back([this, start_idx, end_idx, s_curr, s_angle_curr]() {
                for (int i = start_idx; i < end_idx; ++i) {
                    int flattened_ind = this->flattened_A_points[i];
                    this->update_1_theta_1_site(s_curr, s_angle_curr, flattened_ind);
                }
            });
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
    }
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    // Update A sublattice, phi
    ///////////////////////////////////////////////////////////////////////
    if (flattened_A_points.size() > 0) {
        int actual_threads_A = std::min(this->num_parallel, static_cast<int>(flattened_A_points.size()));
        int chunk_size = flattened_A_points.size() / actual_threads_A;

        // Ensure minimum chunk size
        if (chunk_size == 0) {
            chunk_size = 1;
            actual_threads_A = flattened_A_points.size();
        }

        for (int t = 0; t < actual_threads_A; ++t) {
            int start_idx = t * chunk_size;
            int end_idx = (t == actual_threads_A - 1) ? flattened_A_points.size() : (t + 1) * chunk_size;

            threads.emplace_back([this, start_idx, end_idx, s_curr, s_angle_curr]() {
                for (int i = start_idx; i < end_idx; ++i) {
                    int flattened_ind = this->flattened_A_points[i];
                    this->update_1_phi_1_site(s_curr, s_angle_curr, flattened_ind);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
    }
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    // Update B sublattice, theta
    ///////////////////////////////////////////////////////////////////////
    if (flattened_B_points.size() > 0) {
        int actual_threads_B = std::min(this->num_parallel, static_cast<int>(flattened_B_points.size()));
        int chunk_size_B = flattened_B_points.size() / actual_threads_B;

        // Ensure minimum chunk size
        if (chunk_size_B == 0) {
            chunk_size_B = 1;
            actual_threads_B = flattened_B_points.size();
        }

        for (int t = 0; t < actual_threads_B; ++t) {
            int start_idx = t * chunk_size_B;
            int end_idx = (t == actual_threads_B - 1) ? flattened_B_points.size() : (t + 1) * chunk_size_B;

            threads.emplace_back([this, start_idx, end_idx, s_curr, s_angle_curr]() {
                for (int i = start_idx; i < end_idx; ++i) {
                    int flattened_ind = this->flattened_B_points[i];
                    this->update_1_theta_1_site(s_curr, s_angle_curr, flattened_ind);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
    }
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    // Update B sublattice, phi
    ///////////////////////////////////////////////////////////////////////
    if (flattened_B_points.size() > 0) {
        int actual_threads_B = std::min(this->num_parallel, static_cast<int>(flattened_B_points.size()));
        int chunk_size_B = flattened_B_points.size() / actual_threads_B;

        // Ensure minimum chunk size
        if (chunk_size_B == 0) {
            chunk_size_B = 1;
            actual_threads_B = flattened_B_points.size();
        }

        for (int t = 0; t < actual_threads_B; ++t) {
            int start_idx = t * chunk_size_B;
            int end_idx = (t == actual_threads_B - 1) ? flattened_B_points.size() : (t + 1) * chunk_size_B;

            threads.emplace_back([this, start_idx, end_idx, s_curr, s_angle_curr]() {
                for (int i = start_idx; i < end_idx; ++i) {
                    int flattened_ind = this->flattened_B_points[i];
                    this->update_1_phi_1_site(s_curr, s_angle_curr, flattened_ind);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
    }
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    // Update C sublattice, theta
    ///////////////////////////////////////////////////////////////////////
    if (flattened_C_points.size() > 0) {
        int actual_threads_C = std::min(this->num_parallel, static_cast<int>(flattened_C_points.size()));
        int chunk_size_C = flattened_C_points.size() / actual_threads_C;

        // Ensure minimum chunk size
        if (chunk_size_C == 0) {
            chunk_size_C = 1;
            actual_threads_C = flattened_C_points.size();
        }

        for (int t = 0; t < actual_threads_C; ++t) {
            int start_idx = t * chunk_size_C;
            int end_idx = (t == actual_threads_C - 1) ? flattened_C_points.size() : (t + 1) * chunk_size_C;

            threads.emplace_back([this, start_idx, end_idx, s_curr, s_angle_curr]() {
                for (int i = start_idx; i < end_idx; ++i) {
                    int flattened_ind = this->flattened_C_points[i];
                    this->update_1_theta_1_site(s_curr, s_angle_curr, flattened_ind);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
    }
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    // Update C sublattice, phi
    ///////////////////////////////////////////////////////////////////////
    if (flattened_C_points.size() > 0) {
        int actual_threads_C = std::min(this->num_parallel, static_cast<int>(flattened_C_points.size()));
        int chunk_size_C = flattened_C_points.size() / actual_threads_C;

        // Ensure minimum chunk size
        if (chunk_size_C == 0) {
            chunk_size_C = 1;
            actual_threads_C = flattened_C_points.size();
        }

        for (int t = 0; t < actual_threads_C; ++t) {
            int start_idx = t * chunk_size_C;
            int end_idx = (t == actual_threads_C - 1) ? flattened_C_points.size() : (t + 1) * chunk_size_C;

            threads.emplace_back([this, start_idx, end_idx, s_curr, s_angle_curr]() {
                for (int i = start_idx; i < end_idx; ++i) {
                    int flattened_ind = this->flattened_C_points[i];
                    this->update_1_phi_1_site(s_curr, s_angle_curr, flattened_ind);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
    }
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    // Update D sublattice, theta
    ///////////////////////////////////////////////////////////////////////
    if (flattened_D_points.size() > 0) {
        int actual_threads_D = std::min(this->num_parallel, static_cast<int>(flattened_D_points.size()));
        int chunk_size_D = flattened_D_points.size() / actual_threads_D;

        // Ensure minimum chunk size
        if (chunk_size_D == 0) {
            chunk_size_D = 1;
            actual_threads_D = flattened_D_points.size();
        }

        for (int t = 0; t < actual_threads_D; ++t) {
            int start_idx = t * chunk_size_D;
            int end_idx = (t == actual_threads_D - 1) ? flattened_D_points.size() : (t + 1) * chunk_size_D;

            threads.emplace_back([this, start_idx, end_idx, s_curr, s_angle_curr]() {
                for (int i = start_idx; i < end_idx; ++i) {
                    int flattened_ind = this->flattened_D_points[i];
                    this->update_1_theta_1_site(s_curr, s_angle_curr, flattened_ind);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
    }
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    // Update D sublattice, phi
    ///////////////////////////////////////////////////////////////////////
    if (flattened_D_points.size() > 0) {
        int actual_threads_D = std::min(this->num_parallel, static_cast<int>(flattened_D_points.size()));
        int chunk_size_D = flattened_D_points.size() / actual_threads_D;

        // Ensure minimum chunk size
        if (chunk_size_D == 0) {
            chunk_size_D = 1;
            actual_threads_D = flattened_D_points.size();
        }

        for (int t = 0; t < actual_threads_D; ++t) {
            int start_idx = t * chunk_size_D;
            int end_idx = (t == actual_threads_D - 1) ? flattened_D_points.size() : (t + 1) * chunk_size_D;

            threads.emplace_back([this, start_idx, end_idx, s_curr, s_angle_curr]() {
                for (int i = start_idx; i < end_idx; ++i) {
                    int flattened_ind = this->flattened_D_points[i];
                    this->update_1_phi_1_site(s_curr, s_angle_curr, flattened_ind);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }
    ///////////////////////////////////////////////////////////////////////

    // U_base_value = energy_tot(s_curr);
}

/**
 * @brief Execute full Monte Carlo simulation
 * @param s_vec_init Initial spin configuration (modified in-place)
 * @param s_angle_vec_init Initial angle configuration (modified in-place)
 * @param flushNum Number of data flushes to perform
 *
 * Main simulation loop:
 * For each flush:
 *   For each sweep:
 *     - Update all spins (one full sweep)
 *     - Every sweep_multiple sweeps, save configuration
 *   - Compute magnetizations for all saved configurations
 *   - Save energy, magnetization, and final angles to disk
 *   - Print timing information
 */
void mc_computation::execute_mc(double * s_vec_init, double * s_angle_vec_init,const int& flushNum)
{
    // Calculate starting flush number (continues from previous if applicable)
    int flushThisFileStart=this->flushLastFile+1;

    // Main loop over flushes
    for (int fls = 0; fls < flushNum; fls++)
    {
        // Start timing this flush
        const auto tMCStart{std::chrono::steady_clock::now()};

        // Perform Monte Carlo sweeps
        for (int swp = 0; swp < sweepToWrite*sweep_multiple; swp++)
        {
            // Update all spins (one complete sweep)
            this->update_spins_parallel_1_sweep(s_vec_init,s_angle_vec_init);

            // Save configuration every sweep_multiple sweeps
            if(swp%sweep_multiple==0)
            {
                int swp_out=swp/sweep_multiple;
                // Compute and save total energy
                double energy_tot=this->energy_tot(s_vec_init);
                this->U_data_all_ptr[swp_out]=energy_tot;
                // Copy spin configuration to storage array
                std::memcpy(s_all_ptr+swp_out*total_components_num,s_vec_init,total_components_num*sizeof(double));
                // Copy angle configuration to storage array
                std::memcpy(s_angle_all_ptr+swp_out*tot_angle_components_num,s_angle_vec_init,tot_angle_components_num*sizeof(double));


            }//end save to array
        }//end sweep for
        // Calculate flush number for this file
        int flushEnd=flushThisFileStart+fls;
        std::string fileNameMiddle =  "flushEnd" + std::to_string(flushEnd);

        // Save energy data to pickle file
        std::string out_U_PickleFileName = out_U_path+"/" + fileNameMiddle + ".U.pkl";
        this->save_array_to_pickle(U_data_all_ptr,sweepToWrite,out_U_PickleFileName);

        // Compute magnetization for all saved configurations
        this->compute_all_magnetizations_parallel();
        // Save magnetization data
        std::string out_M_PickleFileName=this->out_M_path+"/" + fileNameMiddle + ".M.pkl";
        //save M
        this->save_array_to_pickle(M_all_ptr,3*sweepToWrite,out_M_PickleFileName);


        // Save final angle configuration (for continuing simulation later)
        std::string out_s_angle_final_PickleFileName = this->out_s_angle_path + "/" + fileNameMiddle + ".s_angle_final.pkl";
        this->save_array_to_pickle(s_angle_vec_init, tot_angle_components_num, out_s_angle_final_PickleFileName);

        // Print timing information
        const auto tMCEnd{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_secondsAll{tMCEnd - tMCStart};
        std::cout << "flush " + std::to_string(flushEnd)  + ": "
                  << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;
    }//end flush for loop

}


/**
 * @brief Compute average magnetization for one configuration
 * @param Mx Output: x-component of magnetization
 * @param My Output: y-component of magnetization
 * @param Mz Output: z-component of magnetization
 * @param startInd Starting index in s_all_ptr
 * @param length Number of spin components (3*N0*N1)
 *
 * Computes: M_α = (1/N) * Σ_i s_α^i  for α = x, y, z
 */
void mc_computation::compute_M_avg_over_sites(double &Mx, double &My, double &Mz,const int &startInd, const int & length)
{
double sum_x=0, sum_y=0,sum_z=0;

    // Sum x-components (at indices 0, 3, 6, ...)
    for (int j=startInd;j<startInd+length;j+=3)
    {
        sum_x+=this->s_all_ptr[j];
    }//end for j

    Mx=sum_x/static_cast<double>(lattice_num);

    // Sum y-components (at indices 1, 4, 7, ...)
    for (int j=startInd+1;j<startInd+length;j+=3)
    {
        sum_y+=this->s_all_ptr[j];
    }
    My=sum_y/static_cast<double>(lattice_num);

    // Sum z-components (at indices 2, 5, 8, ...)
    for (int j=startInd+2;j<startInd+length;j+=3)
    {
        sum_z+=this->s_all_ptr[j];
    }
    Mz=sum_z/static_cast<double>(lattice_num);
}



/**
 * @brief Compute magnetizations for all saved configurations in parallel
 *
 * Divides configurations among threads for parallel processing
 * Each thread computes magnetization for a subset of configurations
 */
void mc_computation::compute_all_magnetizations_parallel()
{
    int num_threads = num_parallel;
    int config_size=total_components_num;// 3*N0*N1
    int  num_configs=sweepToWrite;
    std::vector<std::thread> threads;

    // Calculate how many configurations each thread will process
    int configs_per_thread = num_configs / num_threads;
    int remainder = num_configs % num_threads;

    // Launch threads
    for (int t = 0; t < num_threads; ++t) {
        // Calculate range of configurations for this thread
        int start_config = t * configs_per_thread;
        int end_config = (t == num_threads - 1) ? start_config + configs_per_thread + remainder
                                                : start_config + configs_per_thread;

        // Each thread processes a range of configurations
        threads.emplace_back([this, start_config, end_config, config_size]() {
            for (int config_idx = start_config; config_idx < end_config; ++config_idx) {
                double Mx, My, Mz;
                int startInd = config_idx * config_size;

                // Calculate magnetization for this configuration
                this->compute_M_avg_over_sites(Mx, My, Mz, startInd, config_size);

                // Store results (3 values per configuration: Mx, My, Mz)
                int M_idx = config_idx * 3;
                this->M_all_ptr[M_idx] = Mx;
                this->M_all_ptr[M_idx + 1] = My;
                this->M_all_ptr[M_idx + 2] = Mz;
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}