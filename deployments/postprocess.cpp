#include <pybind11/pybind11.h>
#include <iostream>

using namespace std;
namespace py = pybind11;

double postprocess(double heatmap[], double heatmax[]){
    return i+j;
}



PYBIND11_MODULE(pybind11_exp, m){
    m.doc() = "pybind11 postprocess of centernet.";
    m.def("add",
        &postprocess,
        "A function which input the output of the neural network."
        );
    }


