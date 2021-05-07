#include "input.hpp"
#include <iostream>


struct inputConfig executeConfiguration() {
  struct inputConfig cf;
  cf.nci = 14;
  cf.ncj = 14;
  cf.nck = 14;
  cf.nvt = 5;

  cf.ng  = 3;
  cf.ngi = cf.nci + 2 * cf.ng;
  cf.ngj = cf.ncj + 2 * cf.ng;
  cf.ngk = cf.nck + 2 * cf.ng;


  return cf;
};
