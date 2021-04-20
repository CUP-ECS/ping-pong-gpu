#include "input.hpp"
#include <iostream>

struct inputConfig executeConfiguration() {
  struct inputConfig cf;
  cf.ng  = 1;
  cf.ngi = 4;
  cf.ngj = 4;
  cf.ngk = 4;
  cf.nvt = 12;

  cf.nci = 2;
  cf.ncj = 2;
  cf.nck = 2;

  return cf;
};
