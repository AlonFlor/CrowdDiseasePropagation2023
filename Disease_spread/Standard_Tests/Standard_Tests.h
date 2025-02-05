//!#####################################################################
//! \file Standard_Tests.h
//!#####################################################################
// Class Standard_Tests
//######################################################################
#ifndef __Standard_Tests__
#define __Standard_Tests__

#include <nova/Geometry/Implicit_Objects/Box_Implicit_Object.h>
#include <nova/SPGrid/Tools/SPGrid_Clear.h>
#include <nova/Tools/Utilities/Range_Iterator.h>
#include "../Poisson_Data.h"
#include "../Disease_Spread_Example.h"
//#include "../../Rasterizers/Adaptive_Sphere_Rasterizer.h"

namespace Nova{
template<class T,int d>
class Standard_Tests: public Disease_Spread_Example<T,d>
{
    using TV                        = Vector<T,d>;
    using T_INDEX                   = Vector<int,d>;
    using Struct_type               = Poisson_Data<T>;
    using Hierarchy                 = Grid_Hierarchy<Struct_type,T,d>;
    using Flags_type                = typename Struct_type::Flags_type;
    using Base                      = Disease_Spread_Example<T,d>;
    using Allocator_type            = SPGrid::SPGrid_Allocator<Struct_type,d>;
    using Flag_array_mask           = typename Allocator_type::template Array_mask<unsigned>;

  public:
    using Base::output_directory;using Base::test_number;using Base::counts;using Base::levels;using Base::domain_walls;using Base::hierarchy;using Base::rasterizer;
    using Base::cfl;using Base::sources;using Base::density_channel;

    /****************************
     * example explanation:
     *
     * //1. Simple inflow/outflow.
     * 1. Not yet initialized
     ****************************/

    Standard_Tests()
        :Base()
    {}

//######################################################################
    void Parse_Options() override
    {
        Base::Parse_Options();
        output_directory="Test_"+std::to_string(test_number)+"_Resolution_"+std::to_string(counts(0))+"_Levels_"+std::to_string(levels)+"_CFL_"+std::to_string(cfl);

        for(int axis=0;axis<d;++axis) for(int side=0;side<2;++side) domain_walls(axis)(side)=true;
        domain_walls(1)(1)=false;           // open top

        TV min_corner,max_corner=TV(1);
        hierarchy=new Hierarchy(counts,Range<T,d>(min_corner,max_corner),levels);
    }
//######################################################################
    void Initialize_Rasterizer() override
    {
        //rasterizer=new Adaptive_Sphere_Rasterizer<Struct_type,T,d>(*hierarchy,TV(.5),(T).1);
    }
//######################################################################
    void Initialize_Fluid_State() override
    {
        /*// clear density channel
        for(int level=0;level<levels;++level)
            SPGrid::Clear<Struct_type,T,d>(hierarchy->Allocator(level),hierarchy->Blocks(level),density_channel);

        for(int level=0;level<levels;++level){auto blocks=hierarchy->Blocks(level);
            auto block_size=hierarchy->Allocator(level).Block_Size();
            auto data=hierarchy->Allocator(level).template Get_Array<Struct_type,T>(density_channel);
            auto flags=hierarchy->Allocator(level).template Get_Const_Array<Struct_type,unsigned>(&Struct_type::flags);

            for(unsigned block=0;block<blocks.second;++block){uint64_t offset=blocks.first[block];
                Range_Iterator<d> range_iterator(T_INDEX(),*reinterpret_cast<T_INDEX*>(&block_size)-1);
                T_INDEX base_index(Flag_array_mask::LinearToCoord(offset));

                for(int e=0;e<Flag_array_mask::elements_per_block;++e,offset+=sizeof(Flags_type)){
                    const T_INDEX index=base_index+range_iterator.Index();
                    if(flags(offset)&Cell_Type_Interior && sources(0)->Inside(hierarchy->Lattice(level).Center(index))) data(offset)=(T)1.;
                    range_iterator.Next();}}}*/
    }
//######################################################################
    void Initialize_Sources() override
    {
        /*TV min_corner=TV({.45,0}),max_corner=TV({.55,.05});
        Implicit_Object<T,d>* obj=new Box_Implicit_Object<T,d>(min_corner,max_corner);
        sources.Append(obj);*/
    }
//######################################################################
};
}
#endif
