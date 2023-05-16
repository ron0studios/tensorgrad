#include <python3.11/Python.h>
#include <python3.11/methodobject.h>
#include <python3.11/modsupport.h>
#include <python3.11/moduleobject.h>
#include <python3.11/pyport.h>
#include <python3.11/pytypedefs.h>

int Cfib(int n) {
  if(n<2) return n;
  if(n==2) return 1;
  int a = 1;
  int b = 1;
  for(int i = 3; i <= n; i++){
    int c = a + b;
    a = b;
    b = c;
  }
  return b;
}

static PyObject* fib(PyObject* self, PyObject* args) {
  int n;
  if(!PyArg_ParseTuple(args, "i", &n))
    return NULL;

  return Py_BuildValue("i", Cfib(n));
}

static PyObject* version(PyObject* self){
  return Py_BuildValue("s", "version 1.0");
}

static PyMethodDef myMethods[] = {
  {"fib", fib, METH_VARARGS, "Calculates the fibonacci numbers"},
  {"version", (PyCFunction)version, METH_NOARGS, "returns the version"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myModule = {
  PyModuleDef_HEAD_INIT, 
  "myModule",
  "fibonacci module",
  -1,
  myMethods
};

PyMODINIT_FUNC PyInit_myModule(void) {
  return PyModule_Create(&myModule);
}
