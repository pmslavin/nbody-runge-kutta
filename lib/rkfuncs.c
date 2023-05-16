/*
 *  Python extension module exposing funcs for RK
 *  evaluation.
 *
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>


double G = 6.67408313131313e-11;  // Default G, overwrite in funcs
PyObject *xa, *ya, *vxa, *vya, *ma = NULL;


static PyObject *gravity_first_order_py(PyObject *self, PyObject *bodies){
    /*
     *  System of first-order ODEs representing gravitation.
     *
     *    - Accepts a sequence of N <Body> instances
     *    - Returns a 4-tuple of length N arrays comprising (dv_x, dv_y, vx, vy)
     *
     */
    PyObject *lhs = NULL, *lhs_x = NULL, *lhs_y = NULL, *lhs_vx = NULL, *lhs_vy = NULL;
    PyObject *rhs = NULL, *rhs_x = NULL, *rhs_y = NULL;
    PyObject *dv_x = NULL, *dv_y = NULL, *vxr = NULL, *vyr = NULL;

    if(!PySequence_Check(bodies)){
        PyErr_SetString(PyExc_ValueError, "Argument must be a sequence of <Body> instances");
        return (PyObject *)NULL;
    }

    Py_ssize_t len = PySequence_Size(bodies);

    PyObject *ret = PyList_New((Py_ssize_t)4);   // dv_x, dv_y, vx, vy
    dv_x = PyList_New(len);
    dv_y = PyList_New(len);
    vxr = PyList_New(len);
    vyr = PyList_New(len);

    PyObject *g_dict = PyEval_GetGlobals();
    PyObject *G_ref = PyDict_GetItemString(g_dict, "G");
    G = PyFloat_AsDouble(G_ref);

    for(Py_ssize_t i=0; i<len; i++){
        double ret_dv_x = 0.0;
        double ret_dv_y = 0.0;

        lhs = PySequence_GetItem(bodies, i);
        lhs_vx = PyObject_GetAttr(lhs, vxa);
        lhs_vy = PyObject_GetAttr(lhs, vya);
        PyList_SetItem(vxr, i, lhs_vx);
        PyList_SetItem(vyr, i, lhs_vy);

        lhs_x = PyObject_GetAttr(lhs, xa);
        lhs_y = PyObject_GetAttr(lhs, ya);
        double lx = PyFloat_AsDouble(lhs_x);
        double ly = PyFloat_AsDouble(lhs_y);
        Py_DECREF(lhs_x);
        Py_DECREF(lhs_y);

        for(Py_ssize_t j=0; j<len; j++){
            if(i == j){
                continue;
            }
            rhs = PySequence_GetItem(bodies, j);
            PyObject *mp = PyObject_GetAttr(rhs, ma);
            double m = PyFloat_AsDouble(mp);
            double GM = -G*m;

            rhs_x = PyObject_GetAttr(rhs, xa);
            rhs_y = PyObject_GetAttr(rhs, ya);
            double rx = PyFloat_AsDouble(rhs_x);
            double ry = PyFloat_AsDouble(rhs_y);

            double rdiff_x = lx - rx;
            double rdiff_y = ly - ry;

            double rcubed = pow(pow(rx - lx, 2.0) + pow(ry - ly, 2.0), 3.0/2.0);

            ret_dv_x += GM*rdiff_x/rcubed;
            ret_dv_y += GM*rdiff_y/rcubed;

            Py_DECREF(mp);
            Py_DECREF(rhs);
            Py_DECREF(rhs_x);
            Py_DECREF(rhs_y);
        }
        PyObject *ret_dv_xp = Py_BuildValue("d", ret_dv_x);
        PyObject *ret_dv_yp = Py_BuildValue("d", ret_dv_y);

        PyList_SetItem(dv_x, i, ret_dv_xp);
        PyList_SetItem(dv_y, i, ret_dv_yp);
    }

    PyList_SetItem(ret, 0, dv_x);
    PyList_SetItem(ret, 1, dv_y);
    PyList_SetItem(ret, 2, vxr);
    PyList_SetItem(ret, 3, vyr);

    return ret;
}


static char gravity_first_order_doc[] =
    "System of first-order ODEs representing gravitation.\n"
    "  - Accepts a sequence of N <Body> instances\n"
    "  - Returns a 4-tuple of length N arrays comprising (dv_x, dv_y, vx, vy) per Body";

static PyMethodDef rkfuncs_module_methods[] = {
    {"gravity_first_order", (PyCFunction)gravity_first_order_py,
     METH_O, gravity_first_order_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef rkfuncs_module_def = {
    PyModuleDef_HEAD_INIT,
    "rkfuncs",
    "Miscellaneous functions for evaluation by Runge-Kutta solvers",
    -1,
    rkfuncs_module_methods
};

PyMODINIT_FUNC PyInit_rkfuncs(void){
    Py_Initialize();

    // Body instance attributes
    xa = Py_BuildValue("s", "x");
    ya = Py_BuildValue("s", "y");
    vxa = Py_BuildValue("s", "vx");
    vya = Py_BuildValue("s", "vy");
    ma = Py_BuildValue("s", "M");

    /*
    PyObject *main = PyImport_AddModule("__main__");
    PyObject *G_ref = PyObject_GetAttrString(main, "G");
    G = PyFloat_AsDouble(G_ref);
    */

    return PyModule_Create(&rkfuncs_module_def);
}
