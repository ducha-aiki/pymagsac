#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "magsac_python.hpp"


namespace py = pybind11;


py::tuple findFundamentalMatrix(py::array_t<double>  x1y1_,
                                py::array_t<double>  x2y2_,
                                double sigma_th,
                                double conf,
                                int max_iters,
                                int partition_num) {
    py::buffer_info buf1 = x1y1_.request();
    size_t NUM_TENTS = buf1.shape[0];
    size_t DIM = buf1.shape[1];
    
    if (DIM != 2) {
        throw std::invalid_argument( "x1y1 should be an array with dims [n,2], n>=7" );
    }
    if (NUM_TENTS < 7) {
        throw std::invalid_argument( "x1y1 should be an array with dims [n,2], n>=7");
    }
    py::buffer_info buf1a = x2y2_.request();
    size_t NUM_TENTSa = buf1a.shape[0];
    size_t DIMa = buf1a.shape[1];
    
    if (DIMa != 2) {
        throw std::invalid_argument( "x2y2 should be an array with dims [n,2], n>=7" );
    }
    if (NUM_TENTSa != NUM_TENTS) {
        throw std::invalid_argument( "x1y1 and x2y2 should be the same size");
    }
    
    double *ptr1 = (double *) buf1.ptr;
    std::vector<double> x1y1;
    x1y1.assign(ptr1, ptr1 + buf1.size);
    
    double *ptr1a = (double *) buf1a.ptr;
    std::vector<double> x2y2;
    x2y2.assign(ptr1a, ptr1a + buf1a.size);
    std::vector<double> F(9);
    std::vector<bool> inliers(NUM_TENTS);
    
    int num_inl = findFundamentalMatrix_(x1y1,
                           x2y2,
                           inliers,
                           F,
                           sigma_th,
                           conf,
                           max_iters,
                           partition_num);
    
    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool *ptr3 = (bool *)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];   
    if (num_inl  == 0){
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None),inliers_);
    }
    py::array_t<double> F_ = py::array_t<double>({3,3});
    py::buffer_info buf2 = F_.request();
    double *ptr2 = (double *)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = F[i];
    return py::make_tuple(F_,inliers_);
}

py::tuple findEssentialMatrix(py::array_t<double>  x1y1_,
                                py::array_t<double>  x2y2_,
                                py::array_t<double>  K1_,
                                py::array_t<double>  K2_,
                                double sigma_th,
                                double conf,
                                int max_iters,
                                int partition_num,
                                int core_num,
                                int minimum_inlier_ratio_in_validity_check) {
    py::buffer_info buf1 = x1y1_.request();
    size_t NUM_TENTS = buf1.shape[0];
    size_t DIM = buf1.shape[1];
    
    if (DIM != 2) {
        throw std::invalid_argument( "x1y1 should be an array with dims [n,2], n>=7" );
    }
    if (NUM_TENTS < 7) {
        throw std::invalid_argument( "x1y1 should be an array with dims [n,2], n>=7");
    }
    py::buffer_info buf1a = x2y2_.request();
    size_t NUM_TENTSa = buf1a.shape[0];
    size_t DIMa = buf1a.shape[1];
    
    if (DIMa != 2) {
        throw std::invalid_argument( "x2y2 should be an array with dims [n,2], n>=7" );
    }
    if (NUM_TENTSa != NUM_TENTS) {
        throw std::invalid_argument( "x1y1 and x2y2 should be the same size");
    }

    py::buffer_info bufk1 = K1_.request();
    if (bufk1.shape[0]!= 3 || bufk1.shape[1] != 3){
        throw std::invalid_argument( "K1 should be an array with dims [3,3]" );
    }

    py::buffer_info bufk2 = K2_.request();
    if (bufk2.shape[0]!= 3 || bufk2.shape[1] != 3){
        throw std::invalid_argument( "K2 should be an array with dims [3,3]" );
    }

    
    double *ptr1 = (double *) buf1.ptr;
    std::vector<double> x1y1;
    x1y1.assign(ptr1, ptr1 + buf1.size);
    
    double *ptr1a = (double *) buf1a.ptr;
    std::vector<double> x2y2;
    x2y2.assign(ptr1a, ptr1a + buf1a.size);

    double *ptrk1 = (double *) bufk1.ptr;
    std::vector<double> K1;
    K1.assign(ptrk1, ptrk1 + bufk1.size);

    double *ptrk2 = (double *) bufk2.ptr;
    std::vector<double> K2;
    K2.assign(ptrk2, ptrk2 + bufk2.size);

    

    std::vector<double> E(9);
    std::vector<bool> inliers(NUM_TENTS);
    
    int num_inl = findEssentialMatrix_(x1y1,
                           x2y2,
                           inliers,
                           E,
                           K1,
                           K2,
                           sigma_th,
                           conf,
                           max_iters,
                           partition_num,
                           core_num,
                           minimum_inlier_ratio_in_validity_check);
    
    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool *ptr3 = (bool *)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];   
    if (num_inl  == 0){
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None),inliers_);
    }
    py::array_t<double> E_ = py::array_t<double>({3,3});
    py::buffer_info buf2 = E_.request();
    double *ptr2 = (double *)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = E[i];
    return py::make_tuple(E_,inliers_);
}
                                
py::tuple findHomography(py::array_t<double>  x1y1_,
                         py::array_t<double>  x2y2_,
                         double sigma_th,
                         double conf,
                         int max_iters,
                         int partition_num) {
    py::buffer_info buf1 = x1y1_.request();
    size_t NUM_TENTS = buf1.shape[0];
    size_t DIM = buf1.shape[1];
    
    if (DIM != 2) {
        throw std::invalid_argument( "x1y1 should be an array with dims [n,2], n>=4" );
    }
    if (NUM_TENTS < 4) {
        throw std::invalid_argument( "x1y1 should be an array with dims [n,2], n>=4");
    }
    py::buffer_info buf1a = x2y2_.request();
    size_t NUM_TENTSa = buf1a.shape[0];
    size_t DIMa = buf1a.shape[1];
    
    if (DIMa != 2) {
        throw std::invalid_argument( "x2y2 should be an array with dims [n,2], n>=4" );
    }
    if (NUM_TENTSa != NUM_TENTS) {
        throw std::invalid_argument( "x1y1 and x2y2 should be the same size");
    }
    
    double *ptr1 = (double *) buf1.ptr;
    std::vector<double> x1y1;
    x1y1.assign(ptr1, ptr1 + buf1.size);
    
    double *ptr1a = (double *) buf1a.ptr;
    std::vector<double> x2y2;
    x2y2.assign(ptr1a, ptr1a + buf1a.size);
    std::vector<double> H(9);
    std::vector<bool> inliers(NUM_TENTS);
    
    int num_inl = findHomography_(x1y1,
                    x2y2,
                    inliers,
                    H,
                    sigma_th,
                    conf,
                    max_iters,
                    partition_num);
    
    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool *ptr3 = (bool *)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];   
    
    if (num_inl  == 0){
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None),inliers_);
    }
    py::array_t<double> H_ = py::array_t<double>({3,3});
    py::buffer_info buf2 = H_.request();
    double *ptr2 = (double *)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = H[i];
    
    return py::make_tuple(H_,inliers_);
                         }
PYBIND11_PLUGIN(pymagsac) {
                                                                             
    py::module m("pymagsac", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pymagsac
        .. autosummary::
           :toctree: _generate
           
           findFundamentalMatrix,
           findEssentialMatrix,
           findHomography,

    )doc");

    m.def("findFundamentalMatrix", &findFundamentalMatrix, R"doc(some doc)doc",
          py::arg("x1y1"),
          py::arg("x2y2"),
          py::arg("sigma_th") = 1.0,
          py::arg("conf") = 0.99,
          py::arg("max_iters") = 10000,
          py::arg("partition_num") = 2);

    m.def("findEssentialMatrix", &findEssentialMatrix, R"doc(some doc)doc",
          py::arg("x1y1"),
          py::arg("x2y2"),
          py::arg("K1"),
          py::arg("K2"),
          py::arg("sigma_th") = 1.0,
          py::arg("conf") = 0.99,
          py::arg("max_iters") = 10000,
          py::arg("partition_num") = 2,
          py::arg("core_num") = 1),
          py::arg("minimum_inlier_ratio_in_validity_check") = 0.1;
    

  m.def("findHomography", &findHomography, R"doc(some doc)doc",
        py::arg("x1y1"),
        py::arg("x2y2"),
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("max_iters") = 10000,
        py::arg("partition_num") = 2); 


  return m.ptr();
}
