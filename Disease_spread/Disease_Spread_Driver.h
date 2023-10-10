//!#####################################################################
//! \file Disease_Spread_Driver.h
//!#####################################################################
// Class Disease_Spread_Driver
//######################################################################
#ifndef __Disease_Spread_Driver__
#define __Disease_Spread_Driver__

#include <nova/Tools/Utilities/Driver.h>
#include "Disease_Spread_Example.h"

namespace Nova{
template<class T,int d>
class Disease_Spread_Driver: public Driver<T,d>
{
    using TV                = Vector<T,d>;
    using Base              = Driver<T,d>;

    using Base::time;
    using Base::Compute_Dt;using Base::Write_Output_Files;

  public:
    Disease_Spread_Example<T,d>& example;

    Disease_Spread_Driver(Disease_Spread_Example<T,d>& example_input);
    ~Disease_Spread_Driver() {}

//######################################################################
    void Initialize() override;
    void Advance_One_Time_Step_Explicit_Part(const T dt,const T time);
    void Advance_One_Time_Step_Implicit_Part(const T dt,const T time);
    void Advance_To_Target_Time(const T target_time) override;
    void Simulate_To_Frame(const int frame) override;
//######################################################################
};
}
#endif
