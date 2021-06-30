#include "input.hpp"
#include <iostream>

struct inputConfig executeConfiguration( int max_i ) {
  struct inputConfig cf;

  cf.ngi = max_i;
  cf.ngj = max_i;
  cf.ngk = max_i;
  cf.ng  = 3;

  cf.nci = cf.ngi - 2 * cf.ng;
  cf.ncj = cf.ngj - 2 * cf.ng;
  cf.nck = cf.ngk - 2 * cf.ng;
  cf.nvt = 5;

  return cf;
};
