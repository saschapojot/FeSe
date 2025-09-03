//
// Created by adada on 7/14/2025.
//

#include "mc_read_load_compute.hpp"


void mc_computation::load_pickle_data(const std::string& filename, double* data_ptr,
                                      std::size_t size)
{
    // Initialize Python and NumPy
    Py_Initialize();
    np::initialize();


    try
    {
        // Use Python's 'io' module to open the file directly in binary mode
        py::object io_module = py::import("io");
        py::object file = io_module.attr("open")(filename, "rb"); // Open file in binary mode

        // Import the 'pickle' module
        py::object pickle_module = py::import("pickle");

        // Use pickle.load to deserialize from the Python file object
        py::object loaded_data = pickle_module.attr("load")(file);

        // Close the file
        file.attr("close")();

        // Check if the loaded object is a NumPy array
        if (py::extract<np::ndarray>(loaded_data).check())
        {
            np::ndarray np_array = py::extract<np::ndarray>(loaded_data);

            // Convert the NumPy array to a Python list using tolist()
            py::object py_list = np_array.attr("tolist")();

            // Ensure the list size matches the expected size
            ssize_t list_size = py::len(py_list);
            if (static_cast<std::size_t>(list_size) > size)
            {
                throw std::runtime_error("The provided shared_ptr array size is smaller than the list size.");
            }

            // Copy the data from the Python list to the shared_ptr array
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
        PyErr_Print();
        throw std::runtime_error("Python error occurred.");
    }
}


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

        // Convert C++ array to NumPy array using shared_ptr
        np::ndarray numpy_array = np::from_data(
            ptr, // Use shared_ptr's raw pointer
            np::dtype::get_builtin<double>(), // NumPy data type (double)
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



int mc_computation::mod_direction0(const int& m0)
{
    return ((m0 % N0) + N0) % N0;
}

int mc_computation::mod_direction1(const int& m1)
{
    return ((m1 % N1) + N1) % N1;
}

void mc_computation::init_s()
{
    std::string name;
    std::string s_inFileName;
    if (this->flushLastFile == -1)
    {
        name = "init";
        s_inFileName = this->out_s_path + "/s_" + name + ".pkl";
        this->load_pickle_data(s_inFileName, s_init, total_components_num);
    } //end flushLastFile==-1
    else
    {
        name = "flushEnd" + std::to_string(this->flushLastFile);
        s_inFileName = this->out_s_path + "/" + name + ".s.pkl";
        //load s
        this->load_pickle_data(s_inFileName, s_all_ptr, sweepToWrite * total_components_num);
        //copy last total_components_num elements of to s_init
        std::memcpy(this->s_init, s_all_ptr + total_components_num * (sweepToWrite - 1),
                    total_components_num * sizeof(double));
    } //end else
}


///
/// @param n0
/// @param n1
/// @return flatenned index
int mc_computation::double_ind_to_flat_ind(const int& n0, const int& n1)
{
    return n0 * N1 + n1;
}

//initialize A, B, C, D sublattices for checkerboard update
void mc_computation::init_A_B_C_D_sublattices()
{
    int sublat_num = static_cast<int>(N0*N1/4);

    // Reserve space for all sublattices
    this->A_sublattice.reserve(sublat_num);
    this->B_sublattice.reserve(sublat_num);
    this->C_sublattice.reserve(sublat_num);
    this->D_sublattice.reserve(sublat_num);

    // Initialize all sublattices in a single pass
    for (int i = 0; i < N0; i++)
    {
        for (int j = 0; j < N1; j++)
        {
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


void mc_computation::init_and_run()
{
    this->init_s();
    this->init_A_B_C_D_sublattices();
    // std::cout<<"D:"<<std::endl;
    // for (int n=0;n<D_sublattice.size();n++)
    // {
    //     print_vector(D_sublattice[n]);
    // }

    this->construct_neighbors_origin();
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
    this->init_flattened_ind_neighbors();

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


///construct nearest neighbors and diagonal neighbors around (0, 0)
void mc_computation::construct_neighbors_origin()
{
    this->nearest_neigbors={{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    //print neighbors
    std::cout << "nearest neighbors:" << std::endl;
    for (const auto& vec:nearest_neigbors)
    {
        print_vector(vec);
    }//end for
    this->diagonal_neighbors={{-1,-1},{-1,1},{1,-1},{1,1}};
    std::cout << "diagonal neighbors:" << std::endl;
    for (const auto& vec:diagonal_neighbors)
    {
        print_vector(vec);
    }//end for

    this->neighbors_x={{-1,0},{1,0}};
    std::cout << "neighbors_x:" << std::endl;
    for (const auto& vec:neighbors_x)
    {
        print_vector(vec);
    }//end for

    this->neighbors_y={{0,-1},{0,1}};


    std::cout << "neighbors_y:" << std::endl;
    for (const auto& vec:neighbors_y)
    {
        print_vector(vec);
    }//end for

}





//initialize A, B, C, D sublattices, flattened index
void mc_computation::init_A_B_C_D_sublattices_flattened()
{
    int sublat_num = static_cast<int>(N0*N1/4);
    this->flattened_A_points.reserve(sublat_num);

    this->flattened_B_points.reserve(sublat_num);
    this->flattened_C_points.reserve(sublat_num);
    this->flattened_D_points.reserve(sublat_num);
    int ind0;
    int ind1;
    int flat_ind;
    for (const auto& vec: this->A_sublattice)
    {
        ind0=vec[0];
        ind1=vec[1];
        flat_ind=this->double_ind_to_flat_ind(ind0,ind1);
        flattened_A_points.push_back(flat_ind);
    }//end for A

    for (const auto& vec: this->B_sublattice)
    {
        ind0=vec[0];
        ind1=vec[1];
        flat_ind=this->double_ind_to_flat_ind(ind0,ind1);
        flattened_B_points.push_back(flat_ind);
    }//end for B

    for (const auto& vec: this->C_sublattice)
    {
        ind0=vec[0];
        ind1=vec[1];
        flat_ind=this->double_ind_to_flat_ind(ind0,ind1);
        flattened_C_points.push_back(flat_ind);
    }//end for C

    for (const auto& vec: this->D_sublattice)
    {
        ind0=vec[0];
        ind1=vec[1];
        flat_ind=this->double_ind_to_flat_ind(ind0,ind1);
        flattened_D_points.push_back(flat_ind);
    }//end for D

}


//construct neighbors of each point, flattened index
void mc_computation::init_flattened_ind_neighbors()
{
    this->flattened_ind_nearest_neighbors=std::vector<std::vector<int>>(N0*N1,std::vector<int>());
    this->flattened_ind_diagonal_neighbors=std::vector<std::vector<int>>(N0*N1,std::vector<int>());
    this->flattened_ind_x_neighbors=std::vector<std::vector<int>>(N0*N1,std::vector<int>());
    this->flattened_ind_y_neighbors=std::vector<std::vector<int>>(N0*N1,std::vector<int>());

//flattened_ind_nearest_neighbors
    for (int n0 = 0; n0 < N0; n0++)
    {
        for (int n1 = 0; n1 < N1; n1++)
        {
            int point_curr_flattened = this->double_ind_to_flat_ind(n0, n1);
            for (const auto& vec_nghbrs:this->nearest_neigbors)
            {
                int diff_direc0 = vec_nghbrs[0];
                int diff_direc1 = vec_nghbrs[1];
                int m0 = n0 + diff_direc0;
                int m1 = n1 + diff_direc1;
                int m0_mod = mod_direction0(m0);
                int m1_mod = mod_direction1(m1);
                int flattened_ngb = double_ind_to_flat_ind(m0_mod, m1_mod);
                flattened_ind_nearest_neighbors[point_curr_flattened].push_back(flattened_ngb);
            }//end neighbors
        }//end n1
    }//end n0

    // diagonal neighbors of each point, flattened index
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

    //x neighbors for Kitaev, flattened index
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
    //y neighbors for Kitaev, flattened index
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



///
/// @param flattened_ind_center (flattened) index of spin to be updated
/// @param ind_neighbor index of spin around the center spin (0..3)
/// @param s_vec flattened s array
/// @return Heisenberg energy of flattened_ind_center and ind_neighbor
double mc_computation::H_local_Heisenberg_nearest(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec)
{
    double center_s_x,center_s_y,center_s_z;
    double neighbor_s_x,neighbor_s_y,neighbor_s_z;
    int flattened_ind_one_neighbor =this->flattened_ind_nearest_neighbors[flattened_ind_center][ind_neighbor];

    //get spin values of center
    this->get_spin_components(s_vec,flattened_ind_center,center_s_x,center_s_y,center_s_z);

    // Get spin components of neighbor spin
    this->get_spin_components(s_vec, flattened_ind_one_neighbor, neighbor_s_x, neighbor_s_y, neighbor_s_z);
    double dot_product = center_s_x * neighbor_s_x +
                       center_s_y * neighbor_s_y +
                       center_s_z * neighbor_s_z;

    return this->J11*dot_product;
}



///
/// @param flattened_ind_center flattened) index of spin to be updated
/// @param ind_neighbor index of spin around the center spin (0..3)
/// @param s_vec flattened s array
/// @return Heisenberg energy of flattened_ind_center and ind_neighbor, diagonal neighbors
double mc_computation::H_local_Heisenberg_diagonal(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec)
{
    double center_s_x,center_s_y,center_s_z;
    double neighbor_s_x,neighbor_s_y,neighbor_s_z;
    int flattened_ind_one_neighbor =this->flattened_ind_diagonal_neighbors[flattened_ind_center][ind_neighbor];
    //get spin values of center
    this->get_spin_components(s_vec,flattened_ind_center,center_s_x,center_s_y,center_s_z);

    // Get spin components of neighbor spin
    this->get_spin_components(s_vec, flattened_ind_one_neighbor, neighbor_s_x, neighbor_s_y, neighbor_s_z);
    double dot_product = center_s_x * neighbor_s_x +
                       center_s_y * neighbor_s_y +
                       center_s_z * neighbor_s_z;

    return this->J21*dot_product;
}


///
/// @param flattened_ind_center (flattened) index of spin to be updated
/// @param ind_neighbor index of spin around the center spin (0..3)
/// @param s_vec flattened s array
/// @return biquadratic energy of flattened_ind_center and ind_neighbor, nearest neighbors
double mc_computation::H_local_biquadratic_nearest_neighbor(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec)
{
    double center_s_x,center_s_y,center_s_z;
    double neighbor_s_x,neighbor_s_y,neighbor_s_z;
    int flattened_ind_one_neighbor =this->flattened_ind_nearest_neighbors[flattened_ind_center][ind_neighbor];

    //get spin values of center
    this->get_spin_components(s_vec,flattened_ind_center,center_s_x,center_s_y,center_s_z);

    // Get spin components of neighbor spin
    this->get_spin_components(s_vec, flattened_ind_one_neighbor, neighbor_s_x, neighbor_s_y, neighbor_s_z);

    double dot_product = center_s_x * neighbor_s_x +
                       center_s_y * neighbor_s_y +
                       center_s_z * neighbor_s_z;

    return this->J12*std::pow(dot_product,2.0);


}



///
/// @param flattened_ind_center (flattened) index of spin to be updated
/// @param ind_neighbor index of spin around the center spin (0..3)
/// @param s_vec flattened s array
/// @return biquadratic energy of flattened_ind_center and ind_neighbor, diagonal neighbors
double mc_computation::H_local_biquadratic_diagonal(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec)
{
    double center_s_x,center_s_y,center_s_z;
    double neighbor_s_x,neighbor_s_y,neighbor_s_z;
    int flattened_ind_one_neighbor =this->flattened_ind_diagonal_neighbors[flattened_ind_center][ind_neighbor];
    //get spin values of center
    this->get_spin_components(s_vec,flattened_ind_center,center_s_x,center_s_y,center_s_z);

    // Get spin components of neighbor spin
    this->get_spin_components(s_vec, flattened_ind_one_neighbor, neighbor_s_x, neighbor_s_y, neighbor_s_z);
    double dot_product = center_s_x * neighbor_s_x +
                       center_s_y * neighbor_s_y +
                       center_s_z * neighbor_s_z;

    return this->J22*std::pow(dot_product,2.0);
}



///
/// @param flattened_ind_center (flattened) index of spin to be updated
/// @param ind_neighbor index of spin around the center spin (0..1)
/// @param s_vec flattened s array
/// @return Kitaev energy of flattened_ind_center and ind_neighbor, x neighbors
double mc_computation::H_local_Kitaev_x(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec)
{
    double center_s_x,center_s_y,center_s_z;
    double neighbor_s_x,neighbor_s_y,neighbor_s_z;
    int flattened_ind_one_neighbor =this->flattened_ind_x_neighbors[flattened_ind_center][ind_neighbor];
    //get spin values of center
    this->get_spin_components(s_vec,flattened_ind_center,center_s_x,center_s_y,center_s_z);

    // Get spin components of neighbor spin
    this->get_spin_components(s_vec, flattened_ind_one_neighbor, neighbor_s_x, neighbor_s_y, neighbor_s_z);
    double prod=center_s_x*neighbor_s_x;

    return this->K*prod;
}


///
/// @param flattened_ind_center (flattened) index of spin to be updated
/// @param ind_neighbor index of spin around the center spin (0..1)
/// @param s_vec flattened s array
/// @return Kitaev energy of flattened_ind_center and ind_neighbor, y neighbors
double mc_computation::H_local_Kitaev_y(const int& flattened_ind_center,const int& ind_neighbor,const double * s_vec)
{
    double center_s_x,center_s_y,center_s_z;
    double neighbor_s_x,neighbor_s_y,neighbor_s_z;
    int flattened_ind_one_neighbor =this->flattened_ind_y_neighbors[flattened_ind_center][ind_neighbor];
    //get spin values of center
    this->get_spin_components(s_vec,flattened_ind_center,center_s_x,center_s_y,center_s_z);

    // Get spin components of neighbor spin
    this->get_spin_components(s_vec, flattened_ind_one_neighbor, neighbor_s_x, neighbor_s_y, neighbor_s_z);
    double prod=center_s_y*neighbor_s_y;

    return this->K*prod;

}



///
/// @param flattened_ind flattened [n0, n1], for A, B, C, D
/// @return energy changed if spin [n0,n1] is flipped
double mc_computation::delta_energy(const int &flattened_ind)
{

    // nearest neighbor, Heisenberg

}