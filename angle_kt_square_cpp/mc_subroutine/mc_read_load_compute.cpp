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