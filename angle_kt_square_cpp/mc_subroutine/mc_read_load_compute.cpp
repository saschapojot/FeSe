//
// Created by adada on 9/1/2025.
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

///
/// @param theta angle
/// @param phi angle
/// @param sx sx component of spin
/// @param sy sy component of spin
/// @param sz sz component of spin
void mc_computation::angles_to_spin(const double &theta, const double &phi, double &sx, double &sy, double &sz)
{
    sx=std::cos(theta)*std::sin(phi);
    sy=std::sin(theta)*sin(phi);
    sz=std::cos(phi);
}

void mc_computation::init_s()
{
    std::string name;
    std::string s_angle_inFileName;
    if (this->flushLastFile == -1)
    {
        name = "init";
        s_angle_inFileName=this->out_s_angle_path+"/s_angle_"+name+".pkl";
        this->load_pickle_data(s_angle_inFileName,s_angle_init,tot_angle_components_num);
    } //end flushLastFile==-1
    else
    {
        name = "flushEnd" + std::to_string(this->flushLastFile);
        s_angle_inFileName = this->out_s_angle_path + "/" + name + ".s_angle.pkl";
        //load angle
        this->load_pickle_data(s_angle_inFileName,s_angle_all_ptr,sweepToWrite*tot_angle_components_num);
        //copy  last tot_angle_components_num elements to s_angle_init
        std::memcpy(s_angle_init,s_angle_all_ptr+tot_angle_components_num * (sweepToWrite - 1),tot_angle_components_num*sizeof(double));


    }//end else

    for (int j=0;j<lattice_num;j++)
    {
        // convert angles to spin
        double theta_tmp=s_angle_init[2*j];
        double phi_tmp=s_angle_init[2*j+1];
        double sx_tmp,sy_tmp,sz_tmp;
        this->angles_to_spin(theta_tmp,phi_tmp,sx_tmp,sy_tmp,sz_tmp);
        s_init[3*j]=sx_tmp;
        s_init[3*j+1]=sy_tmp;
        s_init[3*j+2]=sz_tmp;
    }//end for
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


//
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
/// @param x
/// @param leftEnd
/// @param rightEnd
/// @param eps
/// @return return a value within distance eps from x, on the open interval (leftEnd, rightEnd)
double mc_computation::generate_uni_open_interval(const double& x, const double& leftEnd, const double& rightEnd,
                                                  const double& eps)
{
    thread_local std::random_device rd;
    thread_local std::ranlux24_base e2_local(rd());

    double xMinusEps = x - eps;
    double xPlusEps = x + eps;

    double unif_left_end = xMinusEps < leftEnd ? leftEnd : xMinusEps;
    double unif_right_end = xPlusEps > rightEnd ? rightEnd : xPlusEps;

    //    std::random_device rd;
    //    std::ranlux24_base e2(rd());

    double unif_left_end_double_on_the_right = std::nextafter(unif_left_end, std::numeric_limits<double>::infinity());


    std::uniform_real_distribution<> distUnif(unif_left_end_double_on_the_right, unif_right_end);
    //(unif_left_end_double_on_the_right, unif_right_end)

    double xNext = distUnif(e2_local);
    return xNext;
}



///
/// @param theta_curr current value of theta, 1 spin
/// @param theta_next next value of theta, 1 spin
void mc_computation::proposal_uni_theta(const double& theta_curr, double & theta_next)
{

    theta_next=this->generate_uni_open_interval(theta_curr,theta_left_end,theta_right_end,h);
}


///
/// @param phi_curr current value of phi, 1 spin
/// @param phi_next next value of phi, 1 spin
void mc_computation::proposal_uni_phi(const double& phi_curr, double & phi_next)
{
    phi_next=this->generate_uni_open_interval(phi_curr,phi_left_end,phi_right_end,h);
}


///
/// @param flattened_ind flattened index of spin to update
/// @param theta_next proposed new theta value
double mc_computation::delta_energy_update_theta(const double*s_vec_curr, const double* angle_vec_curr, const int& flattened_ind, const double & theta_next)
{
    double sx_curr=0,sy_curr=0,sz_curr=0;
    double theta_curr=0,phi_curr=0;
    //s_vec_curr corresponds to  angle_vec_curr
    this->get_spin_components(s_vec_curr,flattened_ind,sx_curr,sy_curr,sz_curr);
    this->get_angle_components(angle_vec_curr,flattened_ind,theta_curr,phi_curr);

    double sx_next=0,sy_next=0,sz_next=0;
    //proposed new spin
    this->angles_to_spin(theta_next,phi_curr,sx_next,sy_next,sz_next);

    //change in nearest Heiserberg energy
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
/// @return change in nearest neighbor Heisenberg energy
double  mc_computation::delta_energy_Heisenberg_nearest(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                       const double & sx_next, const double &sy_next, const double & sz_next,
                                       const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor)
{


    double E_curr=this->J11*(sx_curr*sx_neighbor+sy_curr*sy_neighbor+sz_curr*sz_neighbor);

    double E_next=this->J11*(sx_next*sx_neighbor+sy_next*sy_neighbor+sz_next*sz_neighbor);

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
/// @return change in diagonal Heisenberg energy
double mc_computation::delta_energy_Heisenberg_diagonal(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                       const double & sx_next, const double &sy_next, const double & sz_next,
                                       const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor)
{

    double E_curr=this->J21*(sx_curr*sx_neighbor+sy_curr*sy_neighbor+sz_curr*sz_neighbor);

    double E_next=this->J21*(sx_next*sx_neighbor+sy_next*sy_neighbor+sz_next*sz_neighbor);

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
/// @return change in diagonal biquadratic energy
double  mc_computation::delta_energy_biquadratic_nearest_neighbor(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                       const double & sx_next, const double &sy_next, const double & sz_next,
                                       const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor)
{

    double prod_curr=sx_curr*sx_neighbor+sy_curr*sy_neighbor+sz_curr*sz_neighbor;

    double prod_next=sx_next*sx_neighbor+sy_next*sy_neighbor+sz_next*sz_neighbor;

    double E_curr=this->J12*std::pow(prod_curr,2.0);

    double E_next=this->J12*std::pow(prod_next,2.0);

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
/// @return change in diagonal biquadratic energy
double mc_computation::delta_energy_biquadratic_diagonal(const double & sx_curr, const double &sy_curr, const double& sz_curr,
                                       const double & sx_next, const double &sy_next, const double & sz_next,
                                       const double & sx_neighbor, const double &sy_neighbor, const double &sz_neighbor)
{
    double prod_curr=sx_curr*sx_neighbor+sy_curr*sy_neighbor+sz_curr*sz_neighbor;

    double prod_next=sx_next*sx_neighbor+sy_next*sy_neighbor+sz_next*sz_neighbor;

    double E_curr=this->J22*std::pow(prod_curr,2.0);

    double E_next=this->J22*std::pow(prod_next,2.0);

    return E_next-E_curr;

}


///
/// @param sx_curr sx, current value
/// @param sx_next sx, next value
/// @param sx_neighbor sx, neighbor
/// @return change in Kitaev energy, x term
double mc_computation::delta_energy_kitaev_x(const double & sx_curr,const double & sx_next,const double & sx_neighbor)
{

    double E_curr=this->K*sx_curr*sx_neighbor;

    double E_next=this->K*sx_next*sx_neighbor;

    return E_next-E_curr;
}


///
/// @param sy_curr sy, current value
/// @param sy_next sy, next value
/// @param sy_neighbor sy, neighbor
/// @return change in Kitaev energy, y term
double mc_computation::delta_energy_kitaev_y(const double &sy_curr,const double &sy_next,const double &sy_neighbor)
{

    double E_curr=this->K*sy_curr*sy_neighbor;

    double E_next=this->K*sy_next*sy_neighbor;

    return E_next-E_curr;
}



///
/// @param s_vec_curr  all current spin values
/// @param angle_vec_curr all current angle values
/// @param flattened_ind flattened index of the spin to update
/// @param phi_next proposed phi value
/// @return change in energy
double mc_computation::delta_energy_update_phi(const double*s_vec_curr, const double* angle_vec_curr,const int& flattened_ind,const double & phi_next)
{
    double sx_curr=0,sy_curr=0,sz_curr=0;
    double theta_curr=0,phi_curr=0;
    //s_vec_curr corresponds to  angle_vec_curr
    this->get_spin_components(s_vec_curr,flattened_ind,sx_curr,sy_curr,sz_curr);

    this->get_angle_components(angle_vec_curr,flattened_ind,theta_curr,phi_curr);
    double sx_next=0,sy_next=0,sz_next=0;
    //proposed new spin
    this->angles_to_spin(theta_curr,phi_next,sx_next,sy_next,sz_next);
    //change in nearest Heiserberg energy
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

///
/// @param x proposed value
/// @param y current value
/// @param a left end of interval
/// @param b right end of interval
/// @param epsilon half length
/// @return proposal probability S(x|y)
double mc_computation::S_uni(const double& x, const double& y, const double& a, const double& b, const double& epsilon)
{
    if (a < y and y < a + epsilon)
    {
        return 1.0 / (y - a + epsilon);
    }
    else if (a + epsilon <= y and y < b - epsilon)
    {
        return 1.0 / (2.0 * epsilon);
    }
    else if (b - epsilon <= y and y < b)
    {
        return 1.0 / (b - y + epsilon);
    }
    else
    {
        std::cerr << "value out of range." << std::endl;
        std::exit(10);
    }
}


///
/// @param theta_curr current theta value
/// @param theta_next next theta value
/// @param dE energy change
/// @return acceptance ratio
double mc_computation::acceptanceRatio_uni_theta(const double &theta_curr, const double &theta_next,const double& dE)
{

    double R = std::exp(-beta*dE);
    double S_curr_next = S_uni(theta_curr,theta_next,theta_left_end,theta_right_end,h);

    double S_next_curr=S_uni(theta_next,theta_curr,theta_left_end,theta_right_end,h);
    double ratio = S_curr_next / S_next_curr;

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

///
/// @param phi_curr current phi value
/// @param phi_next next phi value
/// @param dE energy change
/// @return acceptance ratio
double mc_computation::acceptanceRatio_uni_phi(const double &phi_curr, const double &phi_next,const double& dE)
{


    double R = std::exp(-beta*dE);
    double S_curr_next = S_uni(phi_curr,phi_next,phi_left_end,phi_right_end,h);

    double S_next_curr=S_uni(phi_next,phi_curr,phi_left_end,phi_right_end,h);
    double ratio = S_curr_next / S_next_curr;

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


void  mc_computation::update_1_theta_1_site(double*s_vec_curr,double *angle_vec_curr,const int& flattened_ind)
{
double theta_curr=0,phi_curr=0, theta_next=0;
    thread_local std::random_device rd;
    thread_local std::ranlux24_base e2_local(rd());
    thread_local std::uniform_real_distribution<> distUnif01_local;

    //get angle values
    this->get_angle_components(angle_vec_curr,flattened_ind,theta_curr,phi_curr);
    //propose next theta value
    this->proposal_uni_theta(theta_curr,theta_next);

    double dE=delta_energy_update_theta(s_vec_curr,angle_vec_curr,flattened_ind,theta_next);

    double r=this->acceptanceRatio_uni_theta(theta_curr,theta_next,dE);

    double u = distUnif01_local(e2_local);
    if (u<=r)
    {
        //update angle
        int theta_ind=2*flattened_ind;
        angle_vec_curr[theta_ind]=theta_next;
        double sx=0,sy=0,sz=0;
        this->angles_to_spin(theta_next,phi_curr,sx,sy,sz);
        //update spins
        int s_ind=3*flattened_ind;
        s_vec_curr[s_ind]=sx;
        s_vec_curr[s_ind+1]=sy;
        s_vec_curr[s_ind+2]=sz;


    }//end of accept-reject

}



///
/// @param s_vec_curr all current spins
/// @param angle_vec_curr all current angles
/// @param flattened_ind flattened index of the spin to update, this function updates phi
void mc_computation::update_1_phi_1_site(double*s_vec_curr,double *angle_vec_curr,const int& flattened_ind)
{

    thread_local std::random_device rd;
    thread_local std::ranlux24_base e2_local(rd());
    thread_local std::uniform_real_distribution<> distUnif01_local;
    double theta_curr=0,phi_curr=0, phi_next=0;
    this->get_angle_components(angle_vec_curr,flattened_ind,theta_curr,phi_curr);
    //propose next phi value
    this->proposal_uni_phi(phi_curr,phi_next);
    double dE=delta_energy_update_phi(s_vec_curr,angle_vec_curr,flattened_ind,phi_next);

    double r=this->acceptanceRatio_uni_phi(phi_curr,phi_next,dE);
    double u =  distUnif01_local(e2_local);
    if (u<=r)
    {
        //update angle
        int phi_ind=2*flattened_ind+1;
        angle_vec_curr[phi_ind]=phi_next;
        double sx=0,sy=0,sz=0;
        this->angles_to_spin(theta_curr,phi_next,sx,sy,sz);
        //update spins
        int s_ind=3*flattened_ind;
        s_vec_curr[s_ind]=sx;
        s_vec_curr[s_ind+1]=sy;
        s_vec_curr[s_ind+2]=sz;
    }//end of accept-reject
}



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

    return 0.5*(E_Heisenberg_nn+E_Heisenberg_diag+E_bq_nn+E_bq_diag+E_kt_x+E_kt_y);
}



void mc_computation::update_spins_parallel_1_sweep(double& U_base_value, double *s_curr,double *s_angle_curr)
{
    std::vector<std::thread> threads;

    ///////////////////////////////////////////////////////////////////////
    // update A, theta
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
    // Update A sublattice, phi
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

    U_base_value = energy_tot(s_curr);
}